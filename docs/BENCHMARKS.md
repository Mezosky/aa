# Dataset expansion

The benchmark suite separates four different scientific questions. Results should
be reported per dataset, never pooled.

| Dataset | Role | Protocol | Current status |
|---|---|---|---|
| SOCRATES | Natural-fact validation and geometric experiment | Train on two local facts and test their unseen composition | Complete: 101 chains from the selected composition, two models, three seeds, and controls |
| TwoHopFact | Exploratory scale validation outside the active paper | 44,940 formatted-span-valid chains from 45,595 records; 655 logged exclusions | Artifacts retained; not part of the access diagnosis |
| MQuAKE-CF | Primary counterfactual access test | Train on counterfactual constituent facts; compare direct answering, generated-bridge access, and explicit facts | Complete: Llama and Qwen, three residual seeds, dataset-specific one-hop selection, oracle, paired access ablation, consistency audit, and LoRA pilots |
| CLUTRR | Secondary controlled length generalization | Train on paths of at most three hops and test paths of at least four hops | Complete one-seed role-residual pilot: 8,401 train, 914 validation, and 887 test records |

## Interpretation rules

TwoHopFact is secondary because its scale is useful but shortcut control is weaker
than SOCRATES. Report composition-wise results so dominant templates cannot hide
failures.

MQuAKE-CF tests a stronger claim than ordinary factual access. Success requires the
counterfactual local edits to alter the answer to an unseen consequence question.
The current adapter intentionally retains only two-hop cases because the main
trainer exposes exactly two constituent facts.

CLUTRR cannot use an entity-indexed table without changing the research question.
People are synthetic and change between stories. The implemented extension learns
two shared residuals at the query-subject and query-object spans and holds out
longer relation paths. This is a soft-prompt-like role intervention, not evidence
that the SOCRATES entity table generalizes to unseen entities.

## Reproduction

```bash
python -m geometry_llm evaluate_baseline --config configs/config_twohopfact_llama.yaml
python -m geometry_llm train_delta --config configs/config_twohopfact_llama.yaml --condition correct_delta
python -m geometry_llm evaluate_baseline --config configs/config_mquake_cf_llama.yaml
python -m geometry_llm train_delta --config configs/config_mquake_cf_llama.yaml --condition correct_delta
python -m geometry_llm evaluate_selected --config configs/config_mquake_cf_llama.yaml --condition correct_delta
python -m geometry_llm evaluate_oracle --config configs/config_mquake_cf_llama.yaml
python -m geometry_llm analyze_relations --config configs/config_mquake_cf_llama.yaml --condition correct_delta --seed 13
python -m geometry_llm evaluate_baseline --config configs/config_mquake_cf_qwen.yaml
python -m geometry_llm evaluate_oracle --config configs/config_mquake_cf_qwen.yaml
python -m geometry_llm train_delta --config configs/config_mquake_cf_qwen.yaml --condition correct_delta
python -m geometry_llm evaluate_selected --config configs/config_mquake_cf_qwen.yaml --condition correct_delta
python -m geometry_llm prepare_clutrr --config configs/config_clutrr.yaml
python -m geometry_llm run_clutrr_delta --config configs/config_clutrr.yaml
```

Run shuffled-label and norm-matched random controls exactly as in the SOCRATES
protocol before treating either validation as evidence.
