from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from .text import token_positions_for_span


DEFAULT_ANSWER_ONLY_INSTRUCTION = (
    "Complete the factual statement below. Reply with only the missing entity or value. "
    "Do not repeat the statement and do not add an explanation."
)


def configure_tokenizer(tokenizer, cfg: dict):
    tokenizer._geometry_answer_only_instruction = cfg.get("prompt", {}).get(
        "answer_only_instruction", DEFAULT_ANSWER_ONLY_INSTRUCTION
    )
    return tokenizer


def load_model_and_tokenizer(cfg: dict):
    name = cfg["model"]["name"]
    tokenizer = AutoTokenizer.from_pretrained(name, use_fast=True)
    configure_tokenizer(tokenizer, cfg)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype_name = cfg["model"].get("dtype", "bfloat16")
    dtype = getattr(torch, dtype_name)
    if dtype == torch.bfloat16 and not torch.cuda.is_available():
        dtype = torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=dtype, device_map=cfg["model"].get("device_map", "auto"),
        trust_remote_code=cfg["model"].get("trust_remote_code", False),
    )
    model.config.use_cache = False
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, tokenizer


def chat_prompt(tokenizer, prompt: str) -> str:
    instruction = getattr(tokenizer, "_geometry_answer_only_instruction", DEFAULT_ANSWER_ONLY_INSTRUCTION)
    content = f"{instruction}\n\nStatement: {prompt}"
    messages = [{"role": "user", "content": content}]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return content.rstrip() + "\nAnswer:"


@dataclass
class EncodedExample:
    input_ids: list[int]
    labels: list[int]
    delta_indices: list[int]
    prompt_length: int


class ResidualTable(nn.Module):
    """Compact residual rows; -1 in a position means leave that token untouched."""

    def __init__(self, keys: list[str], hidden_size: int, alpha: float = 1.0,
                 mode: str = "single_token"):
        super().__init__()
        self.keys = list(keys)
        self.key_to_index = {key: i for i, key in enumerate(keys)}
        self.mode = mode
        self.alpha = alpha
        self.delta = nn.Parameter(torch.zeros(len(keys), hidden_size))

    def forward(self, base: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        if self.delta.shape[0] == 0 or self.alpha == 0:
            return base
        valid = indices.ge(0)
        safe = indices.clamp_min(0)
        update = self.delta[safe].to(base.dtype)
        if self.mode == "entity_span":
            counts = (valid.sum(dim=1, keepdim=True).clamp_min(1).sqrt()
                      .to(base.dtype).unsqueeze(-1))
            update = update / counts
        return base + self.alpha * update * valid.unsqueeze(-1)

    def anchor_loss(self, base_norms: torch.Tensor, eps: float = 1e-8):
        return (self.delta.float().pow(2).sum(-1) / (base_norms.float().pow(2) + eps)).mean()


def discover_residual_keys(chains, tokenizer, mode: str) -> tuple[list[str], dict[str, str]]:
    """Return stable row keys plus entity -> representative key for analysis."""
    keys, entity_key = set(), {}
    for chain in chains:
        for prompt, entity in ((chain.prompt_1, chain.e1), (chain.prompt_2, chain.e2),
                               (chain.prompt_12, chain.e1)):
            formatted = chat_prompt(tokenizer, prompt)
            matches = token_positions_for_span(tokenizer, formatted, entity)
            if len(matches) != 1:
                continue
            if mode == "single_token":
                ids = tokenizer(formatted, add_special_tokens=False)["input_ids"]
                if len(matches[0]) != 1:
                    continue
                key = f"token:{ids[matches[0][0]]}"
            else:
                key = f"entity:{entity}"
            keys.add(key)
            entity_key.setdefault(entity, key)
        # Final entities are targets only: retain zero rows for geometry, but the
        # masking logic below guarantees these rows never touch teacher-forced answers.
        if mode == "entity_span":
            key = f"entity:{chain.e3}"
            keys.add(key)
            entity_key.setdefault(chain.e3, key)
        else:
            answer_ids = tokenizer(chain.e3, add_special_tokens=False)["input_ids"]
            if len(answer_ids) == 1:
                key = f"token:{answer_ids[0]}"
                keys.add(key)
                entity_key.setdefault(chain.e3, key)
    return sorted(keys), entity_key


def encode_example(tokenizer, prompt: str, entity: str | None, answer: str | None,
                   table: ResidualTable) -> EncodedExample:
    prefix = chat_prompt(tokenizer, prompt)
    prefix_enc = tokenizer(prefix, add_special_tokens=False, return_offsets_mapping=True)
    positions = [] if entity is None else token_positions_for_span(tokenizer, prefix, entity)
    if entity is not None and len(positions) != 1:
        raise ValueError(f"Expected exactly one occurrence of {entity!r} in formatted prompt")
    prompt_ids = list(prefix_enc["input_ids"])
    if answer is None:
        full_ids = prompt_ids
    else:
        # Tokenize jointly so BPE boundary behavior is identical to normal LM use.
        full_ids = tokenizer(prefix + answer, add_special_tokens=False)["input_ids"]
        if full_ids[:len(prompt_ids)] != prompt_ids:
            full_ids = prompt_ids + tokenizer(answer, add_special_tokens=False)["input_ids"]
        if tokenizer.eos_token_id is not None:
            full_ids = full_ids + [tokenizer.eos_token_id]
    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
    delta_indices = [-1] * len(full_ids)
    if entity is None:
        pass
    elif table.mode == "single_token":
        for pos in positions[0]:
            key = f"token:{prompt_ids[pos]}"
            if key in table.key_to_index:
                delta_indices[pos] = table.key_to_index[key]
    else:
        key = f"entity:{entity}"
        if key in table.key_to_index:
            for pos in positions[0]:
                delta_indices[pos] = table.key_to_index[key]
    return EncodedExample(full_ids, labels, delta_indices, len(prompt_ids))


def pad_batch(items: list[EncodedExample], pad_id: int, device) -> dict[str, torch.Tensor]:
    maximum = max(len(x.input_ids) for x in items)
    ids, labels, deltas, masks = [], [], [], []
    for item in items:
        padding = maximum - len(item.input_ids)
        ids.append(item.input_ids + [pad_id] * padding)
        labels.append(item.labels + [-100] * padding)
        deltas.append(item.delta_indices + [-1] * padding)
        masks.append([1] * len(item.input_ids) + [0] * padding)
    return {
        "input_ids": torch.tensor(ids, device=device),
        "labels": torch.tensor(labels, device=device),
        "delta_indices": torch.tensor(deltas, device=device),
        "attention_mask": torch.tensor(masks, device=device),
    }


def residual_forward(model, table: ResidualTable, batch: dict, output_hidden_states=False):
    base = model.get_input_embeddings()(batch["input_ids"])
    adapted = table(base, batch["delta_indices"])
    return model(
        inputs_embeds=adapted, attention_mask=batch["attention_mask"],
        labels=batch.get("labels"), use_cache=False,
        output_hidden_states=output_hidden_states, return_dict=True,
    )


@torch.no_grad()
def greedy_generate(model, tokenizer, table: ResidualTable, prompt: str, entity: str,
                    max_new_tokens: int, alpha: float | None = None) -> str:
    return greedy_generate_batch(model, tokenizer, table, [prompt], [entity],
                                 max_new_tokens, alpha)[0]


@torch.no_grad()
def greedy_generate_batch(model, tokenizer, table: ResidualTable, prompts: list[str],
                          entities: list[str | None], max_new_tokens: int,
                          alpha: float | None = None) -> list[str]:
    """Batched greedy decoding; residuals apply only to initial prompt spans."""
    if len(prompts) != len(entities):
        raise ValueError("prompts and entities must have equal length")
    if not prompts:
        return []
    encoded = [encode_example(tokenizer, p, e, None, table) for p, e in zip(prompts, entities)]
    maximum = max(len(item.input_ids) for item in encoded)
    device = model.get_input_embeddings().weight.device
    ids, delta_ids, masks = [], [], []
    for item in encoded:
        padding = maximum - len(item.input_ids)
        ids.append([tokenizer.pad_token_id] * padding + item.input_ids)
        delta_ids.append([-1] * padding + item.delta_indices)
        masks.append([0] * padding + [1] * len(item.input_ids))
    ids = torch.tensor(ids, device=device)
    delta_ids = torch.tensor(delta_ids, device=device)
    attention = torch.tensor(masks, device=device)
    position_ids = (attention.long().cumsum(-1) - 1).clamp_min(0)
    old_alpha = table.alpha
    if alpha is not None:
        table.alpha = alpha
    try:
        base = model.get_input_embeddings()(ids)
        outputs = model(inputs_embeds=table(base, delta_ids), attention_mask=attention,
                        position_ids=position_ids, use_cache=True, return_dict=True)
        past = outputs.past_key_values
        token = outputs.logits[:, -1].argmax(-1)
        generated = [[] for _ in prompts]
        finished = torch.zeros(len(prompts), dtype=torch.bool, device=device)
        for step in range(max_new_tokens):
            for row, value in enumerate(token.tolist()):
                if not finished[row] and value != tokenizer.eos_token_id:
                    generated[row].append(value)
            finished |= token.eq(tokenizer.eos_token_id)
            if finished.all() or step + 1 == max_new_tokens:
                break
            next_token = torch.where(finished, torch.full_like(token, tokenizer.pad_token_id), token)
            next_mask = (~finished).long()[:, None]
            attention = torch.cat((attention, next_mask), dim=1)
            next_position = (attention.long().sum(-1) - 1)[:, None]
            outputs = model(input_ids=next_token[:, None], attention_mask=attention,
                            position_ids=next_position, past_key_values=past,
                            use_cache=True, return_dict=True)
            past = outputs.past_key_values
            token = outputs.logits[:, -1].argmax(-1)
        decoded = [tokenizer.decode(row, skip_special_tokens=True).strip() for row in generated]
        return [value.splitlines()[0].strip() if value.splitlines() else "" for value in decoded]
    finally:
        table.alpha = old_alpha


@torch.no_grad()
def _greedy_generate_uncached(model, tokenizer, table: ResidualTable, prompt: str, entity: str,
                              max_new_tokens: int, alpha: float | None = None) -> str:
    """Reference implementation retained for decoder equivalence tests."""
    encoded = encode_example(tokenizer, prompt, entity, None, table)
    device = model.get_input_embeddings().weight.device
    ids = torch.tensor([encoded.input_ids], device=device)
    delta_ids = torch.tensor([encoded.delta_indices], device=device)
    old_alpha = table.alpha
    if alpha is not None:
        table.alpha = alpha
    try:
        generated = []
        for _ in range(max_new_tokens):
            batch = {
                "input_ids": ids, "delta_indices": delta_ids,
                "attention_mask": torch.ones_like(ids),
            }
            logits = residual_forward(model, table, batch).logits[:, -1]
            token = logits.argmax(-1)
            if token.item() == tokenizer.eos_token_id:
                break
            generated.append(token.item())
            ids = torch.cat((ids, token[:, None]), dim=1)
            delta_ids = torch.cat((delta_ids, torch.full_like(token[:, None], -1)), dim=1)
        return tokenizer.decode(generated, skip_special_tokens=True).splitlines()[0].strip()
    finally:
        table.alpha = old_alpha


class FrozenParameterGuard:
    def __init__(self, model):
        self.parameters = list(model.parameters())
        self.versions = [p._version for p in self.parameters]
        assert all(not p.requires_grad for p in self.parameters)

    def assert_unchanged(self):
        assert all(p.grad is None for p in self.parameters), "Frozen model received gradients"
        assert [p._version for p in self.parameters] == self.versions, "Frozen model parameter changed"


def base_row_norms(model, table: ResidualTable) -> torch.Tensor:
    embedding = model.get_input_embeddings().weight.detach()
    values = []
    average = embedding.float().norm(dim=-1).mean()
    for key in table.keys:
        values.append(embedding[int(key.split(":", 1)[1])].float().norm() if key.startswith("token:") else average)
    return torch.stack(values).to(table.delta.device)


def save_residual(path: Path, table: ResidualTable, metadata: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": table.state_dict(), "keys": table.keys, "mode": table.mode,
                "alpha": table.alpha, "metadata": metadata}, path)
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2, default=str))


def load_residual(path: Path, hidden_size: int, device) -> tuple[ResidualTable, dict]:
    item = torch.load(path, map_location=device, weights_only=False)
    table = ResidualTable(item["keys"], hidden_size, item["alpha"], item["mode"]).to(device)
    table.load_state_dict(item["state_dict"])
    return table, item.get("metadata", {})
