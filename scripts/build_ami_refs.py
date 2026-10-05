#!/usr/bin/env python3
"""Build reference transcripts for AMI meetings from the public AMI manual word annotations.

Words from all speakers are merged by start time (a standard approximation for overlapping speech).
Punctuation, vocal sounds (laughter, coughs), gaps and disfluency markers are dropped; spoken fillers
(uh, um, mm...) are kept so ASR output can be scored with or without them.

    python scripts/build_ami_refs.py --annotations <dir containing words/*.xml> --meetings ES2004a IS1009a TS3003a
Annotations: https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip (CC BY 4.0)
Output: eval/ami/<meeting>.ref.txt (git-ignored).
"""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def meeting_words(words_dir: Path, meeting: str) -> list[tuple[float, str]]:
    words: list[tuple[float, str]] = []
    for path in sorted(words_dir.glob(f"{meeting}.*.words.xml")):
        for element in ET.parse(path).getroot():
            if element.tag.rsplit("}", 1)[-1] != "w" or element.get("punc") == "true" or not element.text:
                continue
            words.append((float(element.get("starttime", "0")), element.text.strip()))
    return sorted(words)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--annotations", type=Path, required=True, help="Directory containing words/*.xml")
    parser.add_argument("--meetings", nargs="+", required=True)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "eval" / "ami")
    args = parser.parse_args()

    from services.eval_metrics import normalize_text

    args.out.mkdir(parents=True, exist_ok=True)
    for meeting in args.meetings:
        words = meeting_words(args.annotations / "words", meeting)
        text = normalize_text(" ".join(word for _, word in words))
        (args.out / f"{meeting}.ref.txt").write_text(text + "\n", encoding="utf-8")
        print(f"{meeting}: {len(text.split())} words")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
