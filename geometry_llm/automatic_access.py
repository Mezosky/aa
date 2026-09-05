"""Generated-entity lookup within one response, without a supplied hop template.

An alias is resolved only at the first newline. Both branches discard the old
KV cache and replay the ORIGINAL token IDs, changing only that entity span's
embeddings in the on branch. Thus a multi-token entity is never injected too
late into a stale cache. Each occurrence has its own sqrt(span length) divisor.
"""
from dataclasses import dataclass
import re

import torch

from .text import normalize_answer, token_positions_for_span


INSTRUCTION = (
    "Answer the question by first naming one intermediate entity needed to answer it, "
    "then give the final answer. Use exactly two lines and no explanation:\n"
    "Intermediate: <entity>\nAnswer: <final answer>"
)
ASSISTANT_PREFIX = "Intermediate:"
MAX_ENTITY_TOKENS = 32
MAX_FINAL_TOKENS = 32


@dataclass(frozen=True)
class Span:
    positions: tuple[int, ...]
    row: int


def prepare_prompt(tokenizer, question, source, table):
    content = INSTRUCTION + "\n\nQuestion: " + question
    if getattr(tokenizer, "chat_template", None):
        text = tokenizer.apply_chat_template([{"role": "user", "content": content}],
            tokenize=False, add_generation_prompt=True) + ASSISTANT_PREFIX
    else:
        text = content + "\n\n" + ASSISTANT_PREFIX
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    matches = token_positions_for_span(tokenizer, text, source)
    if len(matches) != 1:
        raise ValueError(f"Non-unique source in automatic-access prompt: {source!r}")
    key = f"entity:{source}"
    if key not in table.key_to_index:
        raise ValueError(f"Missing trained source row: {source!r}")
    return list(ids), [Span(tuple(matches[0]), table.key_to_index[key])]


def resolve_generated(tokenizer, prefix_ids, generated_ids, lookup, table):
    """Match the whole completed field, never a substring or current gold name.

    Offsets are accepted only if fast-tokenizer re-encoding preserves ALL actual
    token IDs. A conservative mismatch receives no lookup, never re-tokenization.
    This avoids silently changing BPE boundaries during cache replay.
    """
    generated = tokenizer.decode(generated_ids, skip_special_tokens=False,
                                  clean_up_tokenization_spaces=False)
    field = generated.split("\n", 1)[0].strip()
    result = dict(intermediate=field, resolved_entity=None, resolution="unknown_or_ambiguous_alias",
                  generated_positions=[], active_nonzero=False)
    if "\n" not in generated:
        return result | {"resolution": "no_completed_field"}, None
    entity = lookup.get(normalize_answer(field))
    if entity is None:
        return result, None
    result["resolved_entity"] = entity
    row = table.key_to_index.get(f"entity:{entity}")
    if row is None:
        return result | {"resolution": "alias_without_row"}, None
    full_ids = list(prefix_ids) + list(generated_ids)
    prefix = tokenizer.decode(prefix_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    full = tokenizer.decode(full_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    encoded = tokenizer(full, add_special_tokens=False, return_offsets_mapping=True)
    if encoded["input_ids"] != full_ids or not full.startswith(prefix) or full[len(prefix):] != generated:
        return result | {"resolution": "non_roundtrip_tokens"}, None
    line = generated.split("\n", 1)[0]
    left = len(prefix) + len(line) - len(line.lstrip())
    right = len(prefix) + len(line.rstrip())
    positions = tuple(i for i, (a,b) in enumerate(encoded["offset_mapping"])
                      if b > left and a < right)
    if not positions or min(positions) < len(prefix_ids):
        return result | {"resolution": "invalid_generated_span"}, None
    # A merged token may include whitespace/newline, but never an answer label.
    start = encoded["offset_mapping"][positions[0]][0]
    end = encoded["offset_mapping"][positions[-1]][1]
    if full[start:left].strip() or full[right:end].strip():
        return result | {"resolution": "mixed_entity_and_other_text_token"}, None
    nonzero = bool(table.delta[row].detach().ne(0).any())
    return result | dict(resolution="resolved_nonzero" if nonzero else "resolved_zero_row",
                         generated_positions=list(positions), active_nonzero=nonzero), Span(positions,row)


def parse_answer(generated):
    """Strict second-line answer; no gold-aware cleanup or substring scoring."""
    lines = generated.splitlines()
    if len(lines) < 2:
        return ""
    match = re.fullmatch(r"\s*Answer:\s*(.*?)\s*", lines[1], re.IGNORECASE)
    return match.group(1) if match else ""


def build_batch(model, table, sequences, spans, pad_id):
    width = max(map(len, sequences))
    device = model.get_input_embeddings().weight.device
    ids = torch.tensor([[pad_id]*(width-len(s))+s for s in sequences], device=device)
    mask = torch.tensor([[0]*(width-len(s))+[1]*len(s) for s in sequences], device=device)
    embeds = model.get_input_embeddings()(ids).clone()
    for b, (seq, occurrences) in enumerate(zip(sequences, spans)):
        used = set()
        for span in occurrences:
            assert span.positions and not used.intersection(span.positions)
            assert min(span.positions) >= 0 and max(span.positions) < len(seq)
            used.update(span.positions)
            positions = [width-len(seq)+p for p in span.positions]
            divisor = len(positions)**.5 if table.mode == "entity_span" else 1.
            update = table.delta[span.row].to(embeds.dtype) / divisor
            embeds[b, positions] += table.alpha * update
    return ids, embeds, mask


def eos_ids(model, tokenizer):
    values = {tokenizer.eos_token_id}
    for cfg in (model.config, model.generation_config):
        ids = getattr(cfg, "eos_token_id", None)
        values.update(ids if isinstance(ids, list) else [ids])
    return values - {None}


@torch.inference_mode()
def decode(model, tokenizer, table, sequences, spans, max_tokens, stop_at_newline=False):
    """Fresh prefill followed by cached decoding, preserving exact sampled IDs."""
    _, embeds, attention = build_batch(model, table, sequences, spans, tokenizer.pad_token_id)
    outputs = model(inputs_embeds=embeds, attention_mask=attention,
        position_ids=(attention.cumsum(-1)-1).clamp_min(0), use_cache=True, return_dict=True)
    past = outputs.past_key_values
    token = outputs.logits[:, -1].argmax(-1)
    generated = [[] for _ in sequences]
    reasons = ["token_limit"]*len(sequences)
    finished = torch.zeros(len(sequences), dtype=torch.bool, device=attention.device)
    stops = eos_ids(model, tokenizer)
    for step in range(max_tokens):
        for b, value in enumerate(token.tolist()):
            if finished[b]:
                continue
            if value in stops:
                finished[b] = True; reasons[b] = "eos"
                continue
            generated[b].append(value)
            if stop_at_newline and "\n" in tokenizer.decode(generated[b],
                    skip_special_tokens=False, clean_up_tokenization_spaces=False):
                finished[b] = True; reasons[b] = "newline"
        if bool(finished.all()) or step+1 == max_tokens:
            break
        next_token = torch.where(finished, tokenizer.pad_token_id, token)
        attention = torch.cat((attention, (~finished).long()[:, None]), dim=1)
        outputs = model(input_ids=next_token[:, None], attention_mask=attention,
            position_ids=(attention.sum(-1)-1)[:, None], past_key_values=past,
            use_cache=True, return_dict=True)
        past = outputs.past_key_values
        token = outputs.logits[:, -1].argmax(-1)
    # Deliberately return NO cache: the caller must replay the entity embeddings.
    return generated, reasons


@torch.inference_mode()
def paired_generate(model, tokenizer, table, questions, sources, lookup):
    prepared = [prepare_prompt(tokenizer,q,s,table) for q,s in zip(questions,sources)]
    prefixes, source_spans = map(list, zip(*prepared))
    intermediate, reasons = decode(model,tokenizer,table,prefixes,source_spans,
                                    MAX_ENTITY_TOKENS,stop_at_newline=True)
    results, on_spans, joined = [], [], []
    for prefix, generated, spans, reason in zip(prefixes,intermediate,source_spans,reasons):
        info, new_span = resolve_generated(tokenizer,prefix,generated,lookup,table)
        results.append(info | dict(boundary=reason, prefix_ids=prefix, intermediate_ids=generated))
        on_spans.append(spans + ([new_span] if new_span is not None else []))
        joined.append(prefix+generated)
    valid = [j for j,r in enumerate(reasons) if r == "newline"]
    for condition, spans in [("off",source_spans),("on",on_spans)]:
        final, stopped = decode(model,tokenizer,table,[joined[j] for j in valid],
            [spans[j] for j in valid],MAX_FINAL_TOKENS) if valid else ([],[])
        for j, tokens, stop in zip(valid,final,stopped):
            text = tokenizer.decode(intermediate[j]+tokens, skip_special_tokens=True,
                                     clean_up_tokenization_spaces=False)
            results[j][f"continuation_ids_{condition}"] = tokens
            results[j][f"response_{condition}"] = ASSISTANT_PREFIX+text
            results[j][f"answer_{condition}"] = parse_answer(text)
            results[j][f"final_stop_{condition}"] = stop
        for j in set(range(len(results)))-set(valid):
            results[j][f"continuation_ids_{condition}"] = []
            results[j][f"response_{condition}"] = ASSISTANT_PREFIX+tokenizer.decode(intermediate[j])
            results[j][f"answer_{condition}"] = ""
            results[j][f"final_stop_{condition}"] = "no_completed_intermediate"
    for result in results:
        if not result["active_nonzero"]:
            assert result["continuation_ids_on"] == result["continuation_ids_off"], "Inactive lookup changed output"
    return results


@torch.inference_mode()
def audit_cache(model, tokenizer, table, sequence, spans, steps=4):
    """Compare replay+cached logits to a full-prefix, no-cache reference each step."""
    _, embeds, mask = build_batch(model,table,[sequence],[spans],tokenizer.pad_token_id)
    out = model(inputs_embeds=embeds,attention_mask=mask,position_ids=mask.cumsum(-1)-1,
                use_cache=True,return_dict=True)
    past = out.past_key_values
    differences, matches = [], []
    for _ in range(steps):
        _, full, mask = build_batch(model,table,[sequence],[spans],tokenizer.pad_token_id)
        ref = model(inputs_embeds=full,attention_mask=mask,position_ids=mask.cumsum(-1)-1,
                    use_cache=False,return_dict=True).logits[:,-1]
        logits = out.logits[:,-1]
        differences.append(float((ref.float()-logits.float()).abs().max()))
        matches.append(bool(ref.argmax(-1).eq(logits.argmax(-1)).all()))
        token = logits.argmax(-1)
        sequence = sequence + [token.item()]
        out = model(input_ids=token[:,None],attention_mask=torch.ones(1,len(sequence),device=mask.device,dtype=mask.dtype),
                    position_ids=torch.tensor([[len(sequence)-1]],device=mask.device),
                    past_key_values=past,use_cache=True,return_dict=True)
        past = out.past_key_values
    return {"max_abs_logit_differences":differences,"greedy_matches":matches,"steps":steps}
