from geometry_llm.text import answer_is_correct, find_entity_spans, normalize_answer, parse_aliases, token_positions_for_span


def test_answer_normalization():
    assert normalize_answer("  Paris!!! ") == "paris"
    assert answer_is_correct("NEW YORK.", ["New York", "NYC"])


def test_alias_parsing():
    assert parse_aliases('["UK", "United Kingdom"]') == ["UK", "United Kingdom"]
    assert parse_aliases("('1918',)") == ["1918"]
    assert parse_aliases({"values": ["a", "b"]}) == ["a", "b"]


def test_boundary_aware_spans():
    assert find_entity_spans("York and New York", "York") == [(0, 4), (13, 17)]


class FakeTokenizer:
    def __call__(self, text, **kwargs):
        # Character tokenizer makes expected offsets transparent.
        return {"input_ids": list(range(len(text))), "offset_mapping": [(i, i + 1) for i in range(len(text))]}


def test_span_detection():
    assert token_positions_for_span(FakeTokenizer(), "go to Rome now", "Rome") == [[6, 7, 8, 9]]


class LeadingSpaceTokenizer:
    def __call__(self, text, **kwargs):
        return {"input_ids": [0, 1, 2], "offset_mapping": [(0, 2), (2, 7), (7, 11)]}


def test_span_detection_allows_tokenizer_leading_space_only():
    tokenizer = LeadingSpaceTokenizer()
    assert token_positions_for_span(tokenizer, "A Rome trip", "Rome") == [[1]]
    assert token_positions_for_span(tokenizer, "A Rome's", "Rome") == []
