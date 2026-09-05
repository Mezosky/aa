"""Coverage-guaranteed factual training and condition-consistent controls."""
from collections import Counter
import random

import torch
from torch.nn import functional as F


def unique_weighted_facts(chains, scope="both"):
    """Visit each distinct (subject, relation, answer) once per epoch.

    Deduplication is across chain occurrences and roles. Each retained fact's
    weight sums its contributions to equally weighted hop-wise, answer-balanced
    objectives. We normalize globally, never within a minibatch.
    """
    facts = {}
    allowed = (1, 2) if scope == "both" else (1,) if scope == "source" else (2,)
    for c in chains:
        for hop, entity, relation, answer, prompt in (
            (1, c.e1, c.r1_type, c.e2, c.prompt_1),
            (2, c.e2, c.r2_type, c.e3, c.prompt_2),
        ):
            if hop not in allowed:
                continue
            key = (entity, relation, answer)
            if key not in facts:
                facts[key] = dict(entity=entity, relation=relation, answer=answer,
                                  prompt=prompt, roles=set())
            facts[key]["roles"].add(hop)
    rows = [facts[k] for k in sorted(facts)]
    counts = {hop: Counter(r["answer"] for r in rows if hop in r["roles"]) for hop in allowed}
    for row in rows:
        weight = sum(1 / (len(allowed) * len(counts[h]) * counts[h][row["answer"]])
                     for h in row["roles"])
        row["weight"] = len(rows) * weight
        row["roles"] = sorted(row["roles"])
    assert abs(sum(r["weight"] for r in rows) - len(rows)) < 1e-6
    return rows


def coverage_epoch(rows, seed):
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    return order


def weighted_answer_loss(model, table, batch, weights):
    """Mean per-fact answer CE (including EOS), with fixed global weights.

    Apply the language-model head only where a teacher-forced target exists;
    this is the same shifted-token objective without prompt-position logits.
    """
    base = model.get_input_embeddings()(batch["input_ids"])
    hidden = model.model(inputs_embeds=table(base, batch["delta_indices"]),
                         attention_mask=batch["attention_mask"], use_cache=False,
                         return_dict=True).last_hidden_state
    labels = batch["labels"][:, 1:]
    valid = labels != -100
    logits = model.lm_head(hidden[:, :-1][valid]).float()
    token_losses = F.cross_entropy(logits, labels[valid], reduction="none")
    example_index = torch.arange(len(labels), device=labels.device)[:, None].expand_as(labels)[valid]
    sums = torch.zeros(len(labels), device=labels.device).scatter_add(0, example_index, token_losses)
    means = sums / valid.sum(1).clamp_min(1)
    return (means * torch.as_tensor(weights, device=means.device)).mean()


def single_edit_targets(raw, scope, bridge_edits):
    """Consequences under the shared source-only / bridge-only edit set.

    For bridge-only edits, follow the original first hop, then look for an
    installed bridge edit at that original subject and relation. Source-only
    consequences use the original answer for the second relation at the NEW
    bridge, not the original whole-chain answer.
    """
    if scope == "both":
        return raw["new_single_hops"][0]["answer"], raw["new_answer"]
    if scope == "source":
        return raw["new_single_hops"][0]["answer"], raw["requested_rewrite"][1]["target_true"]["str"]
    original = raw["orig"]["triples"]
    key = (original[1][0], original[1][1])
    return raw["single_hops"][0]["answer"], bridge_edits.get(key, raw["answer"])
