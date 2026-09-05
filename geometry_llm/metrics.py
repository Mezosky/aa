from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def bootstrap_ci(values, samples: int = 2000, seed: int = 123):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    means = [rng.choice(values, len(values), replace=True).mean() for _ in range(samples)]
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def effective_rank(matrix) -> float:
    singular = np.linalg.svd(np.asarray(matrix), compute_uv=False)
    probabilities = singular / singular.sum() if singular.sum() else singular
    entropy = -(probabilities * np.log(probabilities + 1e-12)).sum()
    return float(np.exp(entropy))


def centered_cka(x, y) -> float:
    x, y = np.asarray(x), np.asarray(y)
    x = x - x.mean(0, keepdims=True)
    y = y - y.mean(0, keepdims=True)
    xy = np.linalg.norm(x.T @ y, "fro") ** 2
    denom = np.linalg.norm(x.T @ x, "fro") * np.linalg.norm(y.T @ y, "fro")
    return float(xy / denom) if denom else float("nan")


def cosine_rows(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return np.sum(a * b, axis=-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-12)


def related_auc(related_scores, unrelated_scores) -> float:
    y = [1] * len(related_scores) + [0] * len(unrelated_scores)
    return float(roc_auc_score(y, list(related_scores) + list(unrelated_scores)))


def summarize_seed_metrics(rows: list[dict], fields: list[str], bootstrap_samples=2000):
    output = {}
    for field in fields:
        vals = [float(r[field]) for r in rows if field in r and r[field] is not None]
        if not vals:
            output[field] = {"mean": None, "std": None, "ci95": [None, None], "n_seeds": 0}
            continue
        output[field] = {
            "mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "ci95": bootstrap_ci(vals, bootstrap_samples), "n_seeds": len(vals),
        }
    return output


def grouped_bootstrap_metrics(rows: list[dict], group_field: str = "bridge_answer_group",
                              samples: int = 2000, seed: int = 123) -> dict:
    """Cluster bootstrap behavioral metrics without treating repeated facts as IID."""
    if not rows:
        return {}
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(str(row[group_field]), []).append(row)
    keys = sorted(groups)
    rng = np.random.default_rng(seed)

    def calculate(sampled: list[dict]) -> dict[str, float]:
        first = np.asarray([r["correct_1"] for r in sampled], dtype=float)
        second = np.asarray([r["correct_2"] for r in sampled], dtype=float)
        direct = np.asarray([r["correct_12"] for r in sampled], dtype=float)
        joint = (first * second).astype(bool)
        return {
            "A_1a": float(first.mean()),
            "A_1b": float(second.mean()),
            "A_2": float(direct.mean()),
            "A_explicit": float(joint.mean()),
            "A_1_independent": float(first.mean() * second.mean()),
            "C": float(direct[joint].mean()) if joint.any() else float("nan"),
        }

    point = calculate(rows)
    draws = {name: [] for name in point}
    for _ in range(samples):
        chosen = rng.choice(keys, len(keys), replace=True)
        sampled = [row for key in chosen for row in groups[key]]
        values = calculate(sampled)
        for name, value in values.items():
            if np.isfinite(value):
                draws[name].append(value)
    return {
        "n_rows": len(rows),
        "n_groups": len(keys),
        "group_field": group_field,
        "metrics": {
            name: {
                "estimate": value,
                "ci95": ([float(np.quantile(draws[name], 0.025)),
                           float(np.quantile(draws[name], 0.975))]
                          if draws[name] else [None, None]),
                "valid_bootstrap_draws": len(draws[name]),
            }
            for name, value in point.items()
        },
    }


def grouped_paired_difference(reference: list[dict], adapted: list[dict],
                              group_field: str = "bridge_answer_group",
                              samples: int = 2000, seed: int = 123) -> dict:
    """Cluster-bootstrap paired accuracy changes on chain-aligned predictions."""
    reference_by_id = {str(row["chain_id"]): row for row in reference}
    pairs = [(reference_by_id[str(row["chain_id"])], row) for row in adapted
             if str(row["chain_id"]) in reference_by_id]
    if not pairs:
        return {}
    groups: dict[str, list[tuple[dict, dict]]] = {}
    for pair in pairs:
        groups.setdefault(str(pair[1][group_field]), []).append(pair)
    keys = sorted(groups)
    fields = {"A_1a": "correct_1", "A_1b": "correct_2", "A_2": "correct_12"}

    def differences(sampled):
        return {
            name: float(np.mean([float(new[field]) - float(old[field])
                                 for old, new in sampled]))
            for name, field in fields.items()
        }

    point = differences(pairs)
    draws = {name: [] for name in fields}
    rng = np.random.default_rng(seed)
    for _ in range(samples):
        chosen = rng.choice(keys, len(keys), replace=True)
        values = differences([pair for key in chosen for pair in groups[key]])
        for name, value in values.items():
            draws[name].append(value)
    return {
        "n_pairs": len(pairs),
        "n_groups": len(keys),
        "group_field": group_field,
        "differences": {
            name: {
                "estimate": value,
                "ci95": [float(np.quantile(draws[name], 0.025)),
                         float(np.quantile(draws[name], 0.975))],
            }
            for name, value in point.items()
        },
    }
