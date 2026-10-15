from services.term_correction import apply_terms, parse_terms


def _words(*texts: str) -> list[dict]:
    return [{"text": t, "start": float(i), "end": float(i) + 0.5} for i, t in enumerate(texts)]


def test_parse_terms_filters_short_and_duplicates() -> None:
    assert parse_terms("Peterson, peterson\nAI, Kubernetes ") == ["Peterson", "Kubernetes"]
    assert parse_terms(None) == []


def test_near_miss_replaced_keeping_punctuation_and_timing() -> None:
    out = apply_terms(_words("Ask", "Petersen,", "please."), ["Peterson"])
    assert [w["text"] for w in out] == ["Ask", "Peterson,", "please."]
    assert out[1]["start"] == 1.0 and out[1]["end"] == 1.5


def test_split_word_merged() -> None:
    out = apply_terms(_words("call", "Peter", "son", "now"), ["Peterson"])
    assert [w["text"] for w in out] == ["call", "Peterson", "now"]
    assert out[1]["start"] == 1.0 and out[1]["end"] == 2.5


def test_multiword_term() -> None:
    out = apply_terms(_words("the", "Kuber", "netes", "cluster"), ["Kubernetes"])
    assert [w["text"] for w in out] == ["the", "Kubernetes", "cluster"]
    out = apply_terms(_words("at", "Acme", "Corp", "today"), ["ACME Corp."])
    assert [w["text"] for w in out] == ["at", "ACME Corp.", "today"]


def test_unrelated_words_untouched() -> None:
    words = _words("the", "person", "spoke", "later")
    assert apply_terms(words, ["Peterson"]) == words
    assert apply_terms(words, []) == words
