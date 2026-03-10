"""Tests for verified Hugging Face model downloads."""

from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from huggingface_hub.utils import RepositoryNotFoundError

from services.model_download_service import (
    ModelVerificationEntry,
    ModelVerificationError,
    ModelVerificationManifest,
    _fetch_verification_manifest,
    _verify_model_snapshot,
    ensure_hf_repo_local_dir_verified,
    ensure_model_cached,
)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _repo_sibling(path: str, *, sha256: str | None = None, size: int | None = None):
    lfs = None if sha256 is None and size is None else SimpleNamespace(sha256=sha256, size=size)
    return SimpleNamespace(rfilename=path, size=size, lfs=lfs)


class ModelDownloadServiceTests(unittest.TestCase):
    def test_fetch_verification_manifest_collects_only_lfs_files(self) -> None:
        expected_sha = _sha256_bytes(b"weights")
        info = SimpleNamespace(
            sha="abc123",
            siblings=[
                _repo_sibling("config.json"),
                _repo_sibling("model.bin", sha256=expected_sha, size=7),
            ],
        )

        with patch("services.model_download_service.HfApi") as mock_api_cls:
            mock_api_cls.return_value.model_info.return_value = info
            manifest = _fetch_verification_manifest(
                repo_id="owner/repo",
                token=None,
                revision=None,
            )

        self.assertEqual(manifest.repo_id, "owner/repo")
        self.assertEqual(manifest.revision, "abc123")
        self.assertEqual(len(manifest.files), 1)
        self.assertEqual(manifest.files[0].relative_path, "model.bin")
        self.assertEqual(manifest.files[0].sha256, expected_sha)

    def test_fetch_verification_manifest_rejects_missing_lfs_sha(self) -> None:
        info = SimpleNamespace(
            sha="abc123",
            siblings=[_repo_sibling("model.bin", sha256=None, size=10)],
        )

        with patch("services.model_download_service.HfApi") as mock_api_cls:
            mock_api_cls.return_value.model_info.return_value = info
            with self.assertRaises(ModelVerificationError):
                _fetch_verification_manifest(
                    repo_id="owner/repo",
                    token=None,
                    revision=None,
                )

    def test_verify_model_snapshot_succeeds_for_matching_sha(self) -> None:
        data = b"verified-model"
        digest = _sha256_bytes(data)
        manifest = ModelVerificationManifest(
            repo_id="owner/repo",
            revision="rev1",
            files=(ModelVerificationEntry(relative_path="model.bin", sha256=digest, size_bytes=len(data)),),
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            snapshot_dir = Path(temp_dir)
            (snapshot_dir / "model.bin").write_bytes(data)
            result = _verify_model_snapshot(snapshot_dir, manifest)

        self.assertEqual(result.repo_id, "owner/repo")
        self.assertEqual(result.revision, "rev1")
        self.assertEqual(result.verified_files, 1)
        self.assertEqual(result.verified_bytes, len(data))

    def test_ensure_model_cached_downloads_revision_and_verifies(self) -> None:
        data = b"downloaded-model"
        digest = _sha256_bytes(data)
        info = SimpleNamespace(
            sha="rev-download",
            siblings=[_repo_sibling("model.bin", sha256=digest, size=len(data))],
        )
        statuses: list[str] = []
        progresses: list[float] = []

        with tempfile.TemporaryDirectory() as temp_dir:
            snapshot_dir = Path(temp_dir) / "rev-download"
            snapshot_dir.mkdir()
            (snapshot_dir / "model.bin").write_bytes(data)
            captured: dict[str, object] = {}

            def _mock_snapshot_download(*args, **kwargs):  # noqa: ANN002, ANN003
                captured["kwargs"] = dict(kwargs)
                return str(snapshot_dir)

            with patch("services.model_download_service._find_cached_snapshot_path", return_value=None), patch(
                "services.model_download_service.get_hf_token",
                return_value=None,
            ), patch("services.model_download_service.HfApi") as mock_api_cls, patch(
                "services.model_download_service.snapshot_download",
                side_effect=_mock_snapshot_download,
            ):
                mock_api_cls.return_value.model_info.return_value = info
                result = ensure_model_cached(
                    "tiny",
                    on_status=statuses.append,
                    on_progress=progresses.append,
                )

        self.assertEqual(result, str(snapshot_dir))
        self.assertEqual(captured["kwargs"]["revision"], "rev-download")
        self.assertFalse(bool(captured["kwargs"]["force_download"]))
        self.assertEqual(progresses[-1], 100.0)
        self.assertTrue(any("Fetching model metadata" in status for status in statuses))
        self.assertTrue(any("Verifying downloaded files" in status for status in statuses))

    def test_ensure_model_cached_rejects_checksum_mismatch(self) -> None:
        expected_digest = _sha256_bytes(b"expected")
        info = SimpleNamespace(
            sha="rev-bad",
            siblings=[_repo_sibling("model.bin", sha256=expected_digest, size=8)],
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            snapshot_dir = Path(temp_dir) / "rev-bad"
            snapshot_dir.mkdir()
            (snapshot_dir / "model.bin").write_bytes(b"unexpected")

            with patch("services.model_download_service._find_cached_snapshot_path", return_value=None), patch(
                "services.model_download_service.get_hf_token",
                return_value=None,
            ), patch("services.model_download_service.HfApi") as mock_api_cls, patch(
                "services.model_download_service.snapshot_download",
                return_value=str(snapshot_dir),
            ):
                mock_api_cls.return_value.model_info.return_value = info
                with self.assertRaises(ModelVerificationError):
                    ensure_model_cached("tiny")

    def test_ensure_model_cached_redownloads_when_cached_snapshot_fails_verification(self) -> None:
        good_data = b"good-model"
        digest = _sha256_bytes(good_data)
        revision = "rev-cached"
        info = SimpleNamespace(
            sha=revision,
            siblings=[_repo_sibling("model.bin", sha256=digest, size=len(good_data))],
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            snapshot_dir = Path(temp_dir) / revision
            snapshot_dir.mkdir()
            model_path = snapshot_dir / "model.bin"
            model_path.write_bytes(b"corrupt")
            captured: dict[str, object] = {}
            repaired_contents = {"value": b""}

            def _mock_snapshot_download(*args, **kwargs):  # noqa: ANN002, ANN003
                captured["kwargs"] = dict(kwargs)
                model_path.write_bytes(good_data)
                repaired_contents["value"] = model_path.read_bytes()
                return str(snapshot_dir)

            with patch(
                "services.model_download_service._find_cached_snapshot_path",
                return_value=str(snapshot_dir),
            ), patch(
                "services.model_download_service.get_hf_token",
                return_value=None,
            ), patch("services.model_download_service.HfApi") as mock_api_cls, patch(
                "services.model_download_service.snapshot_download",
                side_effect=_mock_snapshot_download,
            ):
                mock_api_cls.return_value.model_info.return_value = info
                result = ensure_model_cached("tiny")

        self.assertEqual(result, str(snapshot_dir))
        self.assertTrue(bool(captured["kwargs"]["force_download"]))
        self.assertEqual(repaired_contents["value"], good_data)

    def test_ensure_hf_repo_local_dir_verified_downloads_and_verifies(self) -> None:
        data = b"ocr-model"
        digest = _sha256_bytes(data)
        info = SimpleNamespace(
            sha="rev-ocr",
            siblings=[_repo_sibling("inference.pdiparams", sha256=digest, size=len(data))],
        )
        statuses: list[str] = []

        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir) / "PP-OCRv5_server_det"
            captured: dict[str, object] = {}

            def _mock_snapshot_download(*args, **kwargs):  # noqa: ANN002, ANN003
                captured["kwargs"] = dict(kwargs)
                model_dir.mkdir(parents=True, exist_ok=True)
                (model_dir / "inference.pdiparams").write_bytes(data)
                metadata_dir = model_dir / ".cache" / "huggingface" / "download"
                metadata_dir.mkdir(parents=True, exist_ok=True)
                (metadata_dir / "inference.pdiparams.metadata").write_text(
                    "rev-ocr\netag\n0\n",
                    encoding="utf-8",
                )
                return str(model_dir)

            with patch(
                "services.model_download_service.get_hf_token",
                return_value=None,
            ), patch("services.model_download_service.HfApi") as mock_api_cls, patch(
                "services.model_download_service.snapshot_download",
                side_effect=_mock_snapshot_download,
            ):
                mock_api_cls.return_value.model_info.return_value = info
                result = ensure_hf_repo_local_dir_verified(
                    "PaddlePaddle/PP-OCRv5_server_det",
                    model_dir,
                    on_status=statuses.append,
                )

        self.assertEqual(result, str(model_dir))
        self.assertEqual(captured["kwargs"]["revision"], "rev-ocr")
        self.assertEqual(captured["kwargs"]["local_dir"], str(model_dir))
        self.assertFalse(bool(captured["kwargs"]["force_download"]))
        self.assertTrue(any("Fetching model metadata" in status for status in statuses))
        self.assertTrue(any("Verifying downloaded files" in status for status in statuses))

    def test_ensure_hf_repo_local_dir_verified_redownloads_broken_cache(self) -> None:
        good_data = b"verified-ocr-model"
        digest = _sha256_bytes(good_data)
        info = SimpleNamespace(
            sha="rev-ocr",
            siblings=[_repo_sibling("inference.pdiparams", sha256=digest, size=len(good_data))],
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir) / "PP-OCRv5_server_det"
            model_dir.mkdir(parents=True)
            (model_dir / "inference.pdiparams").write_bytes(b"corrupt")
            metadata_dir = model_dir / ".cache" / "huggingface" / "download"
            metadata_dir.mkdir(parents=True)
            (metadata_dir / "inference.pdiparams.metadata").write_text(
                "rev-ocr\netag\n0\n",
                encoding="utf-8",
            )
            captured: dict[str, object] = {}

            def _mock_snapshot_download(*args, **kwargs):  # noqa: ANN002, ANN003
                captured["kwargs"] = dict(kwargs)
                (model_dir / "inference.pdiparams").write_bytes(good_data)
                return str(model_dir)

            with patch(
                "services.model_download_service.get_hf_token",
                return_value=None,
            ), patch("services.model_download_service.HfApi") as mock_api_cls, patch(
                "services.model_download_service.snapshot_download",
                side_effect=_mock_snapshot_download,
            ):
                mock_api_cls.return_value.model_info.return_value = info
                result = ensure_hf_repo_local_dir_verified(
                    "PaddlePaddle/PP-OCRv5_server_det",
                    model_dir,
                )

        self.assertEqual(result, str(model_dir))
        self.assertTrue(bool(captured["kwargs"]["force_download"]))

    def test_ensure_hf_repo_local_dir_verified_redownloads_when_revision_metadata_missing(self) -> None:
        data = b"verified-ocr-model"
        digest = _sha256_bytes(data)
        info = SimpleNamespace(
            sha="rev-ocr",
            siblings=[_repo_sibling("inference.pdiparams", sha256=digest, size=len(data))],
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir) / "PP-OCRv5_server_det"
            model_dir.mkdir(parents=True)
            (model_dir / "inference.pdiparams").write_bytes(b"stale")
            captured: dict[str, object] = {}

            def _mock_snapshot_download(*args, **kwargs):  # noqa: ANN002, ANN003
                captured["kwargs"] = dict(kwargs)
                metadata_dir = model_dir / ".cache" / "huggingface" / "download"
                metadata_dir.mkdir(parents=True, exist_ok=True)
                (metadata_dir / "inference.pdiparams.metadata").write_text(
                    "rev-ocr\netag\n0\n",
                    encoding="utf-8",
                )
                (model_dir / "inference.pdiparams").write_bytes(data)
                return str(model_dir)

            with patch(
                "services.model_download_service.get_hf_token",
                return_value=None,
            ), patch("services.model_download_service.HfApi") as mock_api_cls, patch(
                "services.model_download_service.snapshot_download",
                side_effect=_mock_snapshot_download,
            ):
                mock_api_cls.return_value.model_info.return_value = info
                result = ensure_hf_repo_local_dir_verified(
                    "PaddlePaddle/PP-OCRv5_server_det",
                    model_dir,
                )

        self.assertEqual(result, str(model_dir))
        self.assertTrue(bool(captured["kwargs"]["force_download"]))

    def test_repository_not_found_still_falls_back_to_model_name(self) -> None:
        statuses: list[str] = []

        with patch("services.model_download_service._find_cached_snapshot_path", return_value=None), patch(
            "services.model_download_service._fetch_verification_manifest",
            side_effect=RepositoryNotFoundError("missing"),
        ):
            result = ensure_model_cached("tiny", on_status=statuses.append)

        self.assertEqual(result, "tiny")
        self.assertTrue(any("not found" in status.lower() for status in statuses))


if __name__ == "__main__":
    unittest.main()
