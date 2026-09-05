# Embedding residual geometry experiment

## Project organization

Shared implementation lives in `geometry_llm/`; command modules are grouped under
`geometry_llm/commands/` by data, training, evaluation, analysis, audits, and reports.
YAML settings live in `configs/`, and protocols and result notes live in `docs/`.
The local `paper/` and `outputs/` folders are excluded by `.gitignore` and remain
available on disk. See [the code layout guide](docs/CODE_LAYOUT.md).

Run workflows from the project root with their existing names and options:

```bash
python -m geometry_llm --help
python -m geometry_llm train_delta --config configs/config_llama.yaml --condition correct_delta
```

Old `python NAME.py ...` commands now use `python -m geometry_llm NAME ...`.
Bare config filenames remain supported; all examples below use the new layout.

## Research scope

This repository studies access to factual edits installed through entity-indexed input residuals. Direct questions activate the source row; a generated-bridge pipeline explicitly accesses the second row. The experiments diagnose this interface using three seeds for Llama and Qwen, MQuAKE consistency audits, and conditional comparisons on a common set of examples. See [RESULTS.md](docs/RESULTS.md) for findings and limitations.

The final additional control performs **automatic lookup during intermediate
generation**, without a supplied hop template or new training. It uses all six
coverage-trained MQuAKE residual tables and rebuilds the KV cache with the generated
entity's edited embeddings. Observed gains are small; correct lookup alone does
not ensure the edited consequence is used. See [AUTOMATIC_ACCESS.md](docs/AUTOMATIC_ACCESS.md)
for the protocol, completed results, audits, and reproduction commands.

The default comparison model is `Qwen/Qwen2.5-7B-Instruct`. Its default SOCRATES composition is university → country → national anthem because a baseline probe found usable one-hop knowledge. A paired, fully specified Llama 3.1 8B configuration is included in `configs/config_llama.yaml`:

```bash
python -m geometry_llm evaluate_baseline --config configs/config_llama.yaml
python -m geometry_llm train_delta --config configs/config_llama.yaml --condition correct_delta
```

The Llama configuration uses exactly the same 101 chains, prompts, entity-span protocol, grid, and three seeds as the Qwen comparison. This paired design isolates the model change; use `data.composition_type=auto` instead if the goal is to select a separate optimal composition for each model.

## Setup

Python 3.10–3.12 and a CUDA GPU with enough memory for a frozen 7B/8B model are recommended. Llama requires accepting its Hugging Face license and logging in.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Every command accepts repeatable dotted overrides such as `--set data.max_examples=20`. Outputs live under `outputs/<config-hash>/`, so configurations do not overwrite each other. Generations, checkpoints, and hidden states are cached there.

## Safe, short path first

Inspect the dataset/tokenizer, then run the approximately 20-chain safety test before any grid search:

```bash
python -m geometry_llm inspect_dataset --set data.max_examples=20
python -m geometry_llm smoke_test --set data.max_examples=20
```

The smoke test verifies that only residual rows receive gradients, frozen parameter version counters do not change, residual and label masks are disjoint, prompt labels are `-100`, no composed prompt enters training, KV cache is disabled, and input/output weights remain unchanged.

## Experiment

Build the full typed graph and run the required baseline audit over every composition. Two shards can run on separate GPUs:

```bash
python -m geometry_llm build_graph
CUDA_VISIBLE_DEVICES=0 python -m geometry_llm evaluate_baseline --all-compositions --num-shards 2 --shard-index 0
CUDA_VISIBLE_DEVICES=1 python -m geometry_llm evaluate_baseline --all-compositions --num-shards 2 --shard-index 1
python -m geometry_llm select_composition 'outputs/<hash>/predictions/original_all_shard-*-of-2.jsonl'
```

The selector uses only one-hop baseline knowledge and data/graph diversity; two-hop accuracy is excluded. Graph nodes use Wikidata IDs where available, and each chain is represented as `e1 -[r1]-> e2 -[r2]-> e3`. The graph is used for audit, splitting, and matched controls, never as a training objective.

Baseline on the automatically selected composition:

```bash
python -m geometry_llm evaluate_baseline
```

The complete per-composition baseline requested in the protocol is deliberately explicit because it is expensive:

```bash
python -m geometry_llm evaluate_baseline --all-compositions
```

Run correct and within-relation shuffled controls:

```bash
python -m geometry_llm train_delta --condition correct_delta
python -m geometry_llm train_delta --condition shuffled_delta \
  --control-from outputs/<hash>/training/correct_delta_selected.json
```

`--control-from` fixes the shuffled run to the correct condition's selected learning rate, anchor, and epoch count, so the control differs only in its within-relation answer assignment.

The default grid is 3 seeds × 3 learning rates × 4 anchors. For a small trial:

```bash
python -m geometry_llm train_delta --condition correct_delta \
  --set data.max_examples=20 --set training.seeds='[13]' \
  --set training.learning_rates='[0.0003]' \
  --set training.anchor_coefficients='[0.001]' --set training.epochs=1
```

Hyperparameters and epoch count are selected using only one-hop validation loss. The code then retrains from zero on all selected chains and writes `final.pt`; this prevents held-out entity-specific rows from remaining at zero during two-hop evaluation. Never select a checkpoint using two-hop output. Evaluate all controls with the `checkpoint` path in `training/*_selected.json`:

```bash
python -m geometry_llm evaluate --condition correct_delta --checkpoint outputs/<hash>/checkpoints/correct_delta/seed-13/lr-<selected>_anchor-<selected>/final.pt --seed 13
python -m geometry_llm evaluate --condition shuffled_delta --checkpoint outputs/<hash>/checkpoints/shuffled_delta/seed-13/lr-<selected>_anchor-<selected>/final.pt --seed 13
python -m geometry_llm evaluate --condition random_delta --checkpoint outputs/<hash>/checkpoints/correct_delta/seed-13/lr-<selected>_anchor-<selected>/final.pt --seed 13
```

`geometry_llm/commands/evaluation/evaluate.py` writes individual-chain JSONL, complete and baseline-knowledge-conditioned metrics, and mean/SD/bootstrap intervals as seeds accumulate. Greedy decoding and maximum length are shared by every condition.

## Geometry, propagation, and interventions

```bash
python -m geometry_llm analyze_geometry --checkpoint <best.pt>
python -m geometry_llm analyze_embedding_structure --config configs/config_llama.yaml
python -m geometry_llm analyze_geometric_memory
python -m geometry_llm analyze_layers --checkpoint <best.pt> --max-examples 100
python -m geometry_llm run_interventions --checkpoint <best.pt> --seed 13
python -m geometry_llm run_residual_swaps --checkpoint <best.pt> --seed 13
python -m geometry_llm analyze_behavior --config configs/config_llama.yaml --seed 13
python -m geometry_llm plot_results
```

`geometry_llm/commands/reports/plot_results.py` collects the available artifacts into a consistent publication-style figure
suite under `outputs/<hash>/plots/`, exporting both high-resolution PNG and vector PDF. It can be
rerun while an experiment is in progress: missing geometry, layer, or intervention outputs are
skipped and added automatically after those stages finish.

Geometry is reported separately for `delta` and `E0 + delta`: cosine and norm-controlled distance distributions, related/unrelated AUC, effective rank, singular spectrum, related-neighbour retrieval, and relation-vector consistency. PCA and UMAP are supporting plots only. Unrelated controls match endpoint graph role/entity type and approximately match target frequency while excluding observed edges.

`geometry_llm/commands/analysis/analyze_embedding_structure.py` is the direct embedding-deformation analysis and does not assume that the residual should reconstruct a global graph. Across seeds and controls it measures the effective entity-span change `Delta/sqrt(span length)`: radial versus tangential motion, angular rotation, cone aperture, residual dimensionality, overlap with the original entity subspace, CKA, pairwise and geodesic-distance preservation, nearest-neighbor turnover, and local organization by supervised target. It also produces PCA and UMAP motion maps as supporting visualizations.

`geometry_llm/commands/analysis/analyze_geometric_memory.py` adds the stricter paper-inspired tests: role-conditioned direct and
unseen two-hop retrieval using both cosine and inner products, graph-component spectral alignment,
row-permutation nulls, and source-to-answer cosine heatmaps. It also reports an important
identifiability constraint of the main protocol: final-answer entities never receive residual
gradients, because answer tokens are intentionally not residualized during teacher forcing.

Layer analysis compares residual-off and residual-on hidden states at the entity and final prompt token. It reports relative change, representation preservation CKA, and bridge/answer alignment CKA. Layerwise alignment or decodability does not identify the location of reasoning.

`geometry_llm/commands/reports/make_paper_figures.py` also computes title-free cross-layer CKA heatmaps from the cached layer representations. These compare every frozen layer with every adapted layer at both the entity span and final prompt token.

## Additional benchmarks

The dataset expansion is staged so completed SOCRATES results are not mixed with planned validation runs. See `docs/BENCHMARKS.md` for the exact scientific role and current status of each dataset.

```bash
# Full 45,595-chain secondary validation
python -m geometry_llm evaluate_baseline --config configs/config_twohopfact_llama.yaml
python -m geometry_llm train_delta --config configs/config_twohopfact_llama.yaml --condition correct_delta

# Two-hop counterfactual cases from MQuAKE-CF
python -m geometry_llm evaluate_baseline --config configs/config_mquake_cf_llama.yaml
python -m geometry_llm train_delta --config configs/config_mquake_cf_llama.yaml --condition correct_delta
python -m geometry_llm evaluate_baseline --config configs/config_mquake_cf_qwen.yaml
python -m geometry_llm train_delta --config configs/config_mquake_cf_qwen.yaml --condition correct_delta

# Prepare and run the short-to-long CLUTRR role-residual experiment
python -m geometry_llm prepare_clutrr --config configs/config_clutrr.yaml
python -m geometry_llm run_clutrr_delta --config configs/config_clutrr.yaml
```

TwoHopFact and MQuAKE-CF use the entity-indexed two-hop residual trainer. CLUTRR uses two shared query-role residuals because its synthetic people change across stories. This makes CLUTRR a soft-prompt-like length-generalization experiment, not a direct extension of the entity-specific factual table.

For a more identifiable geometric test, `geometry_llm/commands/training/run_geometric_calibration.py` constructs a connected, bounded-degree tree from the SOCRATES entity graph. Every retained node occurs as an input and receives bidirectional local-edge supervision, while all distance-two pairs remain unseen. This removes the zero-row ambiguity of final-answer-only nodes in the natural-language experiment:

```bash
CUDA_VISIBLE_DEVICES=0 python -m geometry_llm run_geometric_calibration --config configs/config.yaml
CUDA_VISIBLE_DEVICES=1 python -m geometry_llm run_geometric_calibration --config configs/config_llama.yaml
python -m geometry_llm compare_geometric_calibrations
```

The calibration reports local and unseen two-hop generation, direct-edge and distance-two AUC, graph-distance Spearman correlation, Laplacian low-frequency CKA and permutation enrichment, effective rank, and the same measurements for the propagated hidden-state change at layers 0, 4, …, 32. Its publication-style PNG/PDF figures are written below `outputs/geometric_calibration/`.

After both natural-language runs and all controls are complete, create the paired model figure with:

```bash
python -m geometry_llm compare_models
python -m geometry_llm compare_embedding_structures
```

Interventions cover scaling, zeroing, row permutation, targeted donor swaps, and projection of leading relation-displacement directions. Conditional estimates vary across seeds and populations. A failed donor swap shows non-portability under that intervention, not absence of factual information.

## Access and conditional audit

Coverage-guaranteed MQuAKE fits are the primary experiments; source-only and
bridge-only controls use condition-consistent targets. See
[FINAL_CONFIRMATION.md](docs/FINAL_CONFIRMATION.md) for the protocol and exact
reproduction commands. Run `make_confirmation_report` before `make_access_report`
so the access plot uses the primary coverage estimates. The original reference
fits remain available separately.

```bash
python -m geometry_llm audit_mquake --source /tmp/MQuAKE-CF-3k-v2.json --root outputs/265b7bb1723b
python -m geometry_llm audit_composition --root outputs/bb53acb4d683
python -m geometry_llm audit_composition --root outputs/c243073e2862
python -m geometry_llm audit_composition --root outputs/265b7bb1723b
python -m geometry_llm audit_composition --root outputs/86cc2e353f23
python -m geometry_llm evaluate_access --config configs/config_mquake_cf_llama.yaml --root outputs/265b7bb1723b
python -m geometry_llm evaluate_access --config configs/config_mquake_cf_qwen.yaml --root outputs/86cc2e353f23
python -m geometry_llm run_lora_baseline --config configs/config_mquake_cf_llama.yaml --root outputs/265b7bb1723b
python -m geometry_llm run_lora_baseline --config configs/config_mquake_cf_qwen.yaml --root outputs/86cc2e353f23
python -m geometry_llm run_lora_baseline --config configs/config_mquake_cf_qwen.yaml --root outputs/86cc2e353f23 --learning-rate 0.0003
python -m geometry_llm run_lora_baseline --config configs/config_mquake_cf_qwen.yaml --root outputs/86cc2e353f23 --learning-rate 0.0003 --epochs 6
python -m geometry_llm calibrate_lora --config configs/config_mquake_cf_qwen.yaml --root outputs/86cc2e353f23 --checkpoint outputs/86cc2e353f23/lora/seed-13-lr-0.0003-epochs-6/selected.pt --scales 0.8125 0.84375 0.875
python -m geometry_llm run_lora_baseline --config configs/config_mquake_cf_llama.yaml --root outputs/265b7bb1723b --joint-residual
python -m geometry_llm run_lora_baseline --config configs/config_mquake_cf_qwen.yaml --root outputs/86cc2e353f23 --joint-residual --learning-rate 0.0003
python -m geometry_llm make_access_report
```

The legacy `correct_explicit` / `A_explicit` fields mean independent constituent coverage, not an executed pipeline. Pipeline predictions are saved separately as `access_seed-*.jsonl`. The alias resolver never substitutes the gold bridge. The second relation template is supplied, which removes autonomous decomposition from this control.

LoRA selection matches constituent efficacy on installed facts and uses no composed outcomes. This is distinct from held-out generalization. Final evaluation uses the same triplet batching as residual evaluation; bfloat16 greedy outputs can differ slightly from the paired-prompt batches used for efficacy selection, so comparisons use the final evaluation counts.

`--joint-residual` trains the entity table and LoRA together from zero effective updates. It retains both full component budgets, so it is **not** a parameter-matched control. The residual optimizer group reuses its locally selected rate and anchor; the LoRA group uses the supplied rate. Checkpoints at epochs 1, 2, 4, and 8 are compared using constituent efficacy only. Separate `joint_lora/` outputs include the selected checkpoint and paired residual-off and LoRA-off evaluations without retraining. Frozen backbone weights are never optimized.

Replicated residual/control intervals include optimization and bridge/answer-group variation; single-fit LoRA and joint intervals include group variation only.

## Matched final-state geometry

The geometry analysis compares all 533 MQuAKE prompts in both models under frozen, residual, LoRA, joint, random-row, permuted-row, and joint-component-removal conditions. This uses the existing selected adapters, without additional LLM training. Extraction records checkpoint and dataset hashes and does not include answer tokens.

```bash
CUDA_VISIBLE_DEVICES=0 python -m geometry_llm extract_final_geometry --config configs/config_mquake_cf_llama.yaml --root outputs/265b7bb1723b
CUDA_VISIBLE_DEVICES=1 python -m geometry_llm extract_final_geometry --config configs/config_mquake_cf_qwen.yaml --root outputs/86cc2e353f23
for run_root in outputs/265b7bb1723b outputs/86cc2e353f23; do
  OPENBLAS_NUM_THREADS=2 python -m geometry_llm analyze_final_geometry --root "$run_root"
  OPENBLAS_NUM_THREADS=2 python -m geometry_llm measure_geometry_angles_norms --root "$run_root"
  OPENBLAS_NUM_THREADS=2 python -m geometry_llm audit_active_geometry --root "$run_root"
done
python -m geometry_llm make_final_geometry_report
```

Each run's `final_geometry/seed-13/` contains representation caches, provenance, covariance spectra, state/displacement CKA, conditional label-permutation diagnostics, shared UMAP coordinates, and group-held-out behavioral probes. `angles_norms.json` and `angles_norms_raw.npz` contain absolute state/update norms, paired norm ratios, rotation angles, cone-related measurements in `analysis.json`, radial/tangential components, and between-interface displacement angles. Zero-displacement directions are undefined, not treated as zero-degree observations. `active_subset.json` audits sampling coverage and repeats key checks on the 478 common active-source queries.

Probe training holds out gold answer identities, averages three fixed fold assignments, and fits geometric references and transformations within training folds. Its control includes relation, prompt length, and frozen geometry. This is holdout for the post-hoc classifier, not for the already fitted LLM adapter. Bootstrap intervals condition on fixed out-of-fold predictions. Neither UMAP islands nor covariance concentration alone establishes a factual manifold or a reasoning mechanism.

The inverse-frequency replacement sampler can leave eligible facts unvisited. In the reference residual fits, 55 source entities were never drawn and remain zero. All 533 chains remain in behavioral reports; the active-source geometry check is explicitly separate. No training data were silently removed or retrained during this analysis.

## Token modes and composition selection

`single_token` is the preferred controlled experiment, but the default Qwen university composition has no usable single-token subset and therefore uses the specified `entity_span` fallback:

```bash
python -m geometry_llm inspect_dataset --set data.token_mode=single_token --set data.composition_type=auto
```

This is an **entity-span adapter**: one entity row is distributed over every token in the validated span with `1/sqrt(span length)` scaling. Choose a composition manually with `--set data.composition_type='<type>'`; otherwise the selector rewards sample size and bridge/answer diversity and penalizes dominant mappings.

## Tests

```bash
pytest -q
python -m compileall -q geometry_llm tests
```

Unit tests cover answer normalization, alias parsing, span matching, entity-span scaling, and separation of residual masks from answer labels.
