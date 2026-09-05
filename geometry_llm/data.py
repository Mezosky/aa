from __future__ import annotations

import json
import random
import re
from ast import literal_eval
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from datasets import load_dataset

from .text import parse_aliases, token_positions_for_span


def nested_get(row: dict, path: str):
    if path in row:
        return row[path]
    value: Any = row
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


@dataclass
class Chain:
    chain_id: str
    e1: str
    e2: str
    e3: str
    e2_aliases: list[str]
    e3_aliases: list[str]
    prompt_1: str
    prompt_2: str
    prompt_12: str
    fact_comp_type: str
    rough_fact_comp_type: str
    e1_id: str = ""
    e2_id: str = ""
    e3_id: str = ""
    e1_type: str = "unknown"
    e2_type: str = "unknown"
    e3_type: str = "unknown"
    e1_rough_type: str = "unknown"
    e2_rough_type: str = "unknown"
    e3_rough_type: str = "unknown"
    r1_type: str = "unknown"
    r2_type: str = "unknown"
    r1_rough_type: str = "unknown"
    r2_rough_type: str = "unknown"
    r1_template: str = ""
    r2_template: str = ""
    r1_template_id: int = -1
    r2_template_id: int = -1


def _value(row, name: str) -> str:
    item = nested_get(row, name)
    if isinstance(item, dict):
        item = item.get("value", item.get("name"))
    if item is None:
        raise KeyError(f"SOCRATES row is missing {name!r}; keys={list(row)}")
    return str(item)


def row_to_chain(row: dict, index: int) -> Chain:
    e2 = _value(row, "e2.value")
    e3 = _value(row, "e3.value")
    e2_aliases = nested_get(row, "e2.minimal_aliases")
    e3_aliases = nested_get(row, "e3.minimal_aliases")
    if e2_aliases is None:
        e2_aliases = nested_get(row, "e2.aliases")
    if e3_aliases is None:
        e3_aliases = nested_get(row, "e3.aliases")
    return Chain(
        chain_id=str(row.get("uid", row.get("id", index))),
        e1=_value(row, "e1.value"), e2=e2, e3=e3,
        e2_aliases=list(dict.fromkeys([e2] + parse_aliases(e2_aliases))),
        e3_aliases=list(dict.fromkeys([e3] + parse_aliases(e3_aliases))),
        prompt_1=_value(row, "r1(e1).prompt"),
        prompt_2=_value(row, "r2(e2).prompt"),
        prompt_12=_value(row, "r2(r1(e1)).prompt"),
        fact_comp_type=str(row.get("fact_comp_type", "unknown")),
        rough_fact_comp_type=str(row.get("rough_fact_comp_type", "unknown")),
        e1_id=str(row.get("e1.wikidata_qid", "")),
        e2_id=str(row.get("e2.wikidata_qid", "")),
        e3_id=str(row.get("e3.wikidata_qid", "")),
        e1_type=str(row.get("e1.category", "unknown")),
        e2_type=str(row.get("e2.category", "unknown")),
        e3_type=str(row.get("e3.category", "unknown")),
        e1_rough_type=str(row.get("e1.rough_category", "unknown")),
        e2_rough_type=str(row.get("e2.rough_category", "unknown")),
        e3_rough_type=str(row.get("e3.rough_category", "unknown")),
        r1_type=str(row.get("r1.category", "unknown")),
        r2_type=str(row.get("r2.category", "unknown")),
        r1_rough_type=str(row.get("r1.rough_category", "unknown")),
        r2_rough_type=str(row.get("r2.rough_category", "unknown")),
        r1_template=str(row.get("r1.template", "")),
        r2_template=str(row.get("r2.template", "")),
        r1_template_id=int(row.get("r1.template_id", -1)),
        r2_template_id=int(row.get("r2.template_id", -1)),
    )


def mquake_cf_row_to_chain(row: dict, index: int) -> Chain | None:
    """Normalize a two-hop MQuAKE-CF counterfactual case.

    MQuAKE also contains longer cases. They are deliberately excluded here
    because the current intervention trains exactly two constituent hops.
    """
    hops = row.get("new_single_hops") or []
    rewrites = row.get("requested_rewrite") or []
    if len(hops) != 2 or len(rewrites) < 2:
        return None
    triples = nested_get(row, "orig.new_triples_labeled") or []
    e1 = str(rewrites[0].get("subject") or (triples[0][0] if triples else ""))
    e2 = str(hops[0]["answer"])
    e3 = str(row.get("new_answer") or hops[1]["answer"])
    questions = row.get("questions") or []
    if not e1 or not questions:
        return None
    r1 = str(rewrites[0].get("relation_id", "unknown"))
    r2 = str(rewrites[1].get("relation_id", "unknown"))
    return Chain(
        chain_id=str(row.get("case_id", index)),
        e1=e1, e2=e2, e3=e3,
        e2_aliases=list(dict.fromkeys([e2] + parse_aliases(hops[0].get("answer_alias")))),
        e3_aliases=list(dict.fromkeys([e3] + parse_aliases(row.get("new_answer_alias")))),
        prompt_1=str(hops[0]["cloze"]),
        prompt_2=str(hops[1]["cloze"]),
        prompt_12=str(questions[0]),
        fact_comp_type=f"{r1} then {r2}",
        rough_fact_comp_type="counterfactual two hop",
        e1_id=str(triples[0][0] if triples else ""),
        e2_id=str(rewrites[0].get("target_new", {}).get("id", "")),
        e3_id=str(rewrites[1].get("target_new", {}).get("id", "")),
        e1_type="counterfactual source", e2_type="counterfactual bridge",
        e3_type="counterfactual answer", r1_type=r1, r2_type=r2,
        r1_template=str(rewrites[0].get("prompt", "")),
        r2_template=str(rewrites[1].get("prompt", "")),
    )


def _dataset_rows(cfg: dict):
    data = cfg["data"]
    kwargs = {}
    if data.get("dataset_config"):
        kwargs["name"] = data["dataset_config"]
    if data.get("data_files"):
        kwargs["data_files"] = data["data_files"]
    return load_dataset(data["dataset"], split=data["split"], **kwargs)


def load_chains(cfg: dict) -> list[Chain]:
    ds = _dataset_rows(cfg)
    adapter = cfg["data"].get("adapter", "socrates")
    converter = {"socrates": row_to_chain, "twohopfact": row_to_chain,
                 "mquake_cf": mquake_cf_row_to_chain}.get(adapter)
    if converter is None:
        raise ValueError(f"Unsupported chain adapter: {adapter!r}")
    chains = [chain for i, row in enumerate(ds)
              if (chain := converter(dict(row), i)) is not None]
    chosen = cfg["data"].get("composition_type", "auto")
    if chosen == "auto":
        selection_file = cfg["data"].get("selection_file")
        if selection_file and Path(selection_file).exists():
            chosen = json.loads(Path(selection_file).read_text())["selected_fact_comp_type"]
        else:
            chosen = select_composition(chains, cfg["data"].get("min_examples", 40))
        cfg["data"]["resolved_composition_type"] = chosen
    if chosen != "all":
        chains = [c for c in chains if c.fact_comp_type == chosen]
    maximum = cfg["data"].get("max_examples")
    return chains[:maximum] if maximum else chains


def select_composition(chains: list[Chain], minimum: int = 40) -> str:
    groups: dict[str, list[Chain]] = defaultdict(list)
    for chain in chains:
        groups[chain.fact_comp_type].append(chain)
    eligible = [(name, rows) for name, rows in groups.items() if len(rows) >= minimum]
    if not eligible:
        eligible = list(groups.items())
    if not eligible:
        raise ValueError("Dataset contains no chains")
    # Prefer sample size and entity diversity, penalizing near-constant mappings.
    def score(item):
        _, rows = item
        diversity = min(len({r.e2 for r in rows}), len({r.e3 for r in rows}))
        concentration = max(
            max(Counter(r.e2 for r in rows).values()),
            max(Counter(r.e3 for r in rows).values()),
        ) / len(rows)
        return len(rows) * (1 - concentration) + 2 * diversity
    return max(eligible, key=score)[0]


def filter_token_mode(chains: list[Chain], tokenizer, mode: str):
    # Validate the exact string sent to the tokenizer. Llama's chat template
    # contains a date such as "26 Jul 2024", which can make a short entity like
    # "Jul" ambiguous even when it occurs once in the raw factual prompt.
    from .modeling import chat_prompt

    kept, failures = [], []
    for c in chains:
        fields = [(c.prompt_1, c.e1), (c.prompt_2, c.e2), (c.prompt_12, c.e1)]
        formatted = [(chat_prompt(tokenizer, prompt), ent) for prompt, ent in fields]
        matches = [token_positions_for_span(tokenizer, prompt, ent) for prompt, ent in formatted]
        valid = all(len(m) == 1 and len(m[0]) >= 1 for m in matches)
        if mode == "single_token":
            valid = valid and all(len(m[0]) == 1 for m in matches)
            valid = valid and all(len(tokenizer(e, add_special_tokens=False)["input_ids"]) == 1
                                  for e in (c.e1, c.e2, c.e3))
        if valid:
            kept.append(c)
        else:
            failures.append({"chain_id": c.chain_id, "reason": "missing/ambiguous span or token count"})
    return kept, failures


def prompt_template(prompt: str, entity: str) -> str:
    return re.sub(re.escape(entity), "<ENTITY>", prompt, flags=re.I)


def grouped_split(chains: list[Chain], fraction: float, seed: int):
    groups = defaultdict(list)
    for c in chains:
        key = ((c.r1_type, c.r1_template_id), (c.r2_type, c.r2_template_id))
        if c.r1_template_id < 0 or c.r2_template_id < 0:
            key = (prompt_template(c.prompt_1, c.e1), prompt_template(c.prompt_2, c.e2))
        groups[key].append(c)
    keys = list(groups)
    random.Random(seed).shuffle(keys)
    target = max(1, round(len(chains) * fraction))
    validation, training = [], []
    for key in keys:
        (validation if len(validation) < target else training).extend(groups[key])
    if not training:  # Degenerate datasets with one template.
        cut = max(1, len(validation) - target)
        training, validation = validation[:cut], validation[cut:]
    return training, validation


def save_chains(chains: list[Chain], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for chain in chains:
            handle.write(json.dumps(asdict(chain), ensure_ascii=False) + "\n")


def load_saved_chains(path: Path) -> list[Chain]:
    with path.open(encoding="utf-8") as handle:
        return [Chain(**json.loads(line)) for line in handle]


@dataclass
class ClutrrExample:
    example_id: str
    story: str
    query_subject: str
    query_object: str
    answer_relation: str
    approximate_hops: int


def clutrr_row_to_example(row: dict, index: int, label_names: list[str]) -> ClutrrExample:
    pair = literal_eval(str(row["sentence2"]))
    if not isinstance(pair, tuple) or len(pair) != 2:
        raise ValueError(f"Invalid CLUTRR query pair: {row['sentence2']!r}")
    entities = list(dict.fromkeys(re.findall(r"\[([^\]]+)\]", str(row["sentence1"]))))
    # The tasksource export omits CLUTRR's path-length column. For its chain-only
    # stories, the number of distinct entities minus one recovers path length.
    approximate_hops = max(1, len(entities) - 1)
    return ClutrrExample(
        example_id=str(row.get("id", index)), story=str(row["sentence1"]),
        query_subject=str(pair[0]), query_object=str(pair[1]),
        answer_relation=label_names[int(row["labels"])],
        approximate_hops=approximate_hops,
    )


def load_clutrr_examples(dataset: str, split: str) -> list[ClutrrExample]:
    ds = load_dataset(dataset, split=split)
    label_names = ds.features["labels"].names
    return [clutrr_row_to_example(dict(row), i, label_names) for i, row in enumerate(ds)]
