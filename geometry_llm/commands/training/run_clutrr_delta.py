#!/usr/bin/env python
"""Train two role residuals on short CLUTRR paths and test longer paths."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from geometry_llm.config import load_config
from tqdm import tqdm

from geometry_llm.modeling import FrozenParameterGuard, ResidualTable, load_model_and_tokenizer
from geometry_llm.text import answer_is_correct, token_positions_for_span


def read_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def format_prompt(tokenizer, row, labels):
    options = ", ".join(labels)
    content = (
        f"Story: {row['story']}\n"
        f"Question: What is {row['query_object']}'s relationship to {row['query_subject']}?\n"
        f"Choose exactly one label: {options}. Reply with only the label."
    )
    messages = [{"role": "user", "content": content}]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return content + "\nAnswer:"


def encode(tokenizer, row, labels, answer=None):
    prefix = format_prompt(tokenizer, row, labels)
    prompt_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(prefix + (answer or ""), add_special_tokens=False)["input_ids"]
    if full_ids[:len(prompt_ids)] != prompt_ids:
        full_ids = prompt_ids + tokenizer(answer or "", add_special_tokens=False)["input_ids"]
    if answer is not None and tokenizer.eos_token_id is not None:
        full_ids.append(tokenizer.eos_token_id)
    labels_out = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
    roles = [-1] * len(full_ids)
    for role_index, name in enumerate((row["query_subject"], row["query_object"])):
        matches = token_positions_for_span(tokenizer, prefix, name)
        if not matches:
            raise ValueError(f"Missing query name {name!r} in formatted CLUTRR prompt")
        for position in matches[-1]:
            roles[position] = role_index
    return {"input_ids": full_ids, "labels": labels_out, "roles": roles,
            "prompt_length": len(prompt_ids)}


def pad(items, pad_id, device, left=False):
    maximum = max(len(item["input_ids"]) for item in items)
    result = {"input_ids": [], "labels": [], "roles": [], "attention_mask": []}
    for item in items:
        amount = maximum - len(item["input_ids"])
        if left:
            result["input_ids"].append([pad_id] * amount + item["input_ids"])
            result["labels"].append([-100] * amount + item["labels"])
            result["roles"].append([-1] * amount + item["roles"])
            result["attention_mask"].append([0] * amount + [1] * len(item["input_ids"]))
        else:
            result["input_ids"].append(item["input_ids"] + [pad_id] * amount)
            result["labels"].append(item["labels"] + [-100] * amount)
            result["roles"].append(item["roles"] + [-1] * amount)
            result["attention_mask"].append([1] * len(item["input_ids"]) + [0] * amount)
    return {name: torch.tensor(value, device=device) for name, value in result.items()}


def add_roles(base, roles, table):
    valid = roles.ge(0)
    safe = roles.clamp_min(0)
    update = table.delta[safe].to(base.dtype)
    for role in range(len(table.keys)):
        selected = roles.eq(role)
        counts = selected.sum(1, keepdim=True).clamp_min(1).sqrt().to(base.dtype).unsqueeze(-1)
        update = torch.where(selected.unsqueeze(-1), update / counts, update)
    return base + update * valid.unsqueeze(-1)


def forward(model, table, batch):
    base = model.get_input_embeddings()(batch["input_ids"])
    return model(inputs_embeds=add_roles(base, batch["roles"], table),
                 attention_mask=batch["attention_mask"], labels=batch["labels"],
                 use_cache=False, return_dict=True)


@torch.no_grad()
def generate_batch(model, tokenizer, table, rows, labels, max_new_tokens, enabled):
    items = [encode(tokenizer, row, labels) for row in rows]
    device = model.get_input_embeddings().weight.device
    batch = pad(items, tokenizer.pad_token_id, device, left=True)
    ids, roles, attention = batch["input_ids"], batch["roles"], batch["attention_mask"]
    base = model.get_input_embeddings()(ids)
    embedded = add_roles(base, roles, table) if enabled else base
    position_ids = (attention.long().cumsum(-1) - 1).clamp_min(0)
    output = model(inputs_embeds=embedded, attention_mask=attention,
                   position_ids=position_ids, use_cache=True, return_dict=True)
    past, token = output.past_key_values, output.logits[:, -1].argmax(-1)
    generated = [[] for _ in rows]
    finished = torch.zeros(len(rows), dtype=torch.bool, device=device)
    for step in range(max_new_tokens):
        for index, value in enumerate(token.tolist()):
            if not finished[index] and value != tokenizer.eos_token_id:
                generated[index].append(value)
        finished |= token.eq(tokenizer.eos_token_id)
        if finished.all() or step + 1 == max_new_tokens:
            break
        next_token = torch.where(finished, torch.full_like(token, tokenizer.pad_token_id), token)
        attention = torch.cat((attention, (~finished).long()[:, None]), dim=1)
        next_position = (attention.long().sum(-1) - 1)[:, None]
        output = model(input_ids=next_token[:, None], attention_mask=attention,
                       position_ids=next_position, past_key_values=past,
                       use_cache=True, return_dict=True)
        past, token = output.past_key_values, output.logits[:, -1].argmax(-1)
    return [tokenizer.decode(values, skip_special_tokens=True).splitlines()[0].strip()
            if values else "" for values in generated]


@torch.no_grad()
def evaluate(model, tokenizer, table, rows, labels, cfg, enabled, name):
    output = []
    batch_size = cfg["model"]["evaluation_batch_size"]
    for start in tqdm(range(0, len(rows), batch_size), desc=f"CLUTRR {name}"):
        chunk = rows[start:start + batch_size]
        predictions = generate_batch(model, tokenizer, table, chunk, labels,
                                     cfg["model"]["max_answer_tokens"], enabled)
        for row, prediction in zip(chunk, predictions):
            output.append(row | {"prediction": prediction,
                                 "correct": answer_is_correct(prediction, [row["answer_relation"]])})
    return output


def save_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config_clutrr.yaml")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    out = Path(cfg["output_dir"])
    train = read_jsonl(out / "short_train.jsonl")
    validation = read_jsonl(out / "long_validation.jsonl")
    test = read_jsonl(out / "long_test.jsonl")
    labels = sorted({row["answer_relation"] for row in train})
    random.seed(cfg["training"]["seed"]); np.random.seed(cfg["training"]["seed"])
    torch.manual_seed(cfg["training"]["seed"])
    model, tokenizer = load_model_and_tokenizer(cfg)
    device = model.get_input_embeddings().weight.device
    table = ResidualTable(["role:query_subject", "role:query_object"],
                          model.get_input_embeddings().embedding_dim,
                          mode="entity_span").to(device)
    checkpoint = out / "role_delta.pt"
    if checkpoint.exists() and not args.overwrite:
        table.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    else:
        optimizer = torch.optim.AdamW([table.delta], lr=cfg["training"]["learning_rate"], weight_decay=0)
        guard = FrozenParameterGuard(model)
        batch_size = cfg["training"]["batch_size"]
        accumulation = cfg["training"]["gradient_accumulation_steps"]
        history, step = [], 0
        for epoch in range(cfg["training"]["epochs"]):
            random.Random(cfg["training"]["seed"] + epoch).shuffle(train)
            optimizer.zero_grad(set_to_none=True)
            running = []
            for start in tqdm(range(0, len(train), batch_size), desc=f"CLUTRR train epoch {epoch + 1}"):
                items = [encode(tokenizer, row, labels, row["answer_relation"])
                         for row in train[start:start + batch_size]]
                batch = pad(items, tokenizer.pad_token_id, device)
                loss = forward(model, table, batch).loss / accumulation
                loss.backward(); running.append(float(loss.detach()) * accumulation)
                if ((start // batch_size) + 1) % accumulation == 0 or start + batch_size >= len(train):
                    torch.nn.utils.clip_grad_norm_([table.delta], cfg["training"]["gradient_clip"])
                    optimizer.step(); optimizer.zero_grad(set_to_none=True); step += 1
                    guard.assert_unchanged()
            history.append({"epoch": epoch + 1, "optimizer_steps": step,
                            "mean_train_loss": float(np.mean(running)),
                            "residual_norm": float(table.delta.detach().float().norm())})
        torch.save(table.state_dict(), checkpoint)
        (out / "training_history.json").write_text(json.dumps(history, indent=2))
    report = {"labels": labels, "train_examples": len(train),
              "validation_examples": len(validation), "test_examples": len(test), "conditions": {}}
    prediction_cache = {}
    for enabled, condition in ((False, "original"), (True, "role_delta")):
        for split_name, rows in (("long_validation", validation), ("long_test", test)):
            path = out / "predictions" / f"{condition}_{split_name}.jsonl"
            if path.exists() and not args.overwrite:
                predictions = read_jsonl(path)
            else:
                predictions = evaluate(model, tokenizer, table, rows, labels, cfg, enabled,
                                       f"{condition} {split_name}")
                save_rows(path, predictions)
            key = f"{condition}_{split_name}"
            prediction_cache[key] = predictions
            report["conditions"][key] = {
                "n": len(predictions),
                "accuracy": float(np.mean([row["correct"] for row in predictions])),
            }
    report["paired_bootstrap"] = {}
    rng = np.random.default_rng(123)
    for split_name in ("long_validation", "long_test"):
        original = prediction_cache[f"original_{split_name}"]
        adapted = prediction_cache[f"role_delta_{split_name}"]
        differences = np.asarray([float(new["correct"]) - float(old["correct"])
                                  for old, new in zip(original, adapted)])
        draws = np.mean(rng.choice(differences, (10000, len(differences)), replace=True), axis=1)
        report["paired_bootstrap"][split_name] = {
            "accuracy_difference": float(differences.mean()),
            "ci95": [float(np.quantile(draws, .025)), float(np.quantile(draws, .975))],
            "gained": int(np.sum(differences == 1)), "lost": int(np.sum(differences == -1)),
        }
    report["residual_norm"] = float(table.delta.detach().float().norm())
    (out / "results.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
