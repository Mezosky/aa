from __future__ import annotations

import ast
import re
import string
from collections.abc import Iterable


def normalize_answer(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"\s+", " ", value)
    return value.strip(string.whitespace + string.punctuation)


def parse_aliases(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        raw = value.strip()
        if raw.startswith(("[", "(", "{")):
            try:
                return parse_aliases(ast.literal_eval(raw))
            except (ValueError, SyntaxError):
                pass
        return [value]
    if isinstance(value, dict):
        for key in ("value", "values", "aliases", "minimal_aliases"):
            if key in value:
                return parse_aliases(value[key])
        return [str(v) for v in value.values()]
    if isinstance(value, Iterable):
        result = []
        for item in value:
            result.extend(parse_aliases(item))
        return result
    return [str(value)]


def answer_is_correct(prediction: str, aliases: list[str]) -> bool:
    pred = normalize_answer(prediction)
    return any(pred == normalize_answer(alias) for alias in aliases)


def find_entity_spans(text: str, entity: str) -> list[tuple[int, int]]:
    """Case-insensitive, boundary-aware character spans."""
    if not entity.strip():
        return []
    pattern = re.compile(r"(?<!\w)" + re.escape(entity.strip()) + r"(?!\w)", re.I)
    return [(m.start(), m.end()) for m in pattern.finditer(text)]


def token_positions_for_span(tokenizer, text: str, entity: str) -> list[list[int]]:
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = encoded["offset_mapping"]
    matches = []
    for start, end in find_entity_spans(text, entity):
        positions = [
            i for i, (left, right) in enumerate(offsets)
            if right > start and left < end
        ]
        if positions:
            token_start = min(offsets[i][0] for i in positions)
            token_end = max(offsets[i][1] for i in positions)
            # Byte-level BPE tokenizers commonly attach preceding whitespace to
            # the first token (for example, ``" Robert"``).  That is still a
            # valid entity span, but never accept a token that also contains
            # neighbouring non-whitespace text or punctuation.
            left_extra = text[token_start:start]
            right_extra = text[end:token_end]
            if token_start <= start and token_end >= end \
                    and (not left_extra or left_extra.isspace()) \
                    and (not right_extra or right_extra.isspace()):
                matches.append(positions)
    return matches
