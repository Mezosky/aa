# Final targeted confirmation

This iteration adds new results under each MQuAKE root's `confirmation_v1/` directory. Original predictions and adapters are not overwritten.

## Reproduce

```bash
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m geometry_llm run_final_confirmation --config configs/config_mquake_cf_llama.yaml --root outputs/265b7bb1723b
CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m geometry_llm run_final_confirmation --config configs/config_mquake_cf_qwen.yaml --root outputs/86cc2e353f23
.venv/bin/python -m geometry_llm audit_final_confirmation --root outputs/265b7bb1723b
.venv/bin/python -m geometry_llm audit_final_confirmation --root outputs/86cc2e353f23
.venv/bin/python -m geometry_llm make_confirmation_report
.venv/bin/python -m geometry_llm make_access_report
tectonic -o paper paper/main.tex
.venv/bin/python -m pytest -q
```

The model and tokenizer files must already be cached for offline execution. The original source JSON is `/tmp/MQuAKE-CF-3k-v2.json`; `--source` accepts another copy. Its checksum and the selected-chain checksum are recorded in the experiment manifest.

## Protocol

- The same 533 audited counterfactual cases supply 892 unique facts and 839 eligible input-entity rows.
- Each epoch visits each unique fact exactly once, shuffling its order. Loss weights balance answers within each trained hop. Token cross-entropy is averaged within each fact, including EOS; weights are normalized globally, not within batches.
- Hyperparameters and epoch counts are fixed to the previous dataset-specific one-hop selection. These are confirmation settings, not a new search. The new protocol also changes duplicate weighting and per-fact normalization, so differences are not attributed solely to coverage.
- Both-edits adapters use seeds 13, 37, and 71 in both models. Source-only and bridge-only adapters use seed 13 and the corresponding fixed epoch count. Their optimizer-step budgets differ because they contain different numbers of facts.
- Eight cases touching entities shared between source and bridge roles are excluded from the single-edit comparison, leaving 525 matched cases. Training still uses each condition's full fact pool.
- Single-edit consequence targets are recomputed from the edits actually present. Source-only uses the unedited second fact at the new bridge. Bridge-only follows the original bridge and applies any matching bridge edit in the shared table. This avoids scoring incompatible factual worlds against the same two-edit target.
- Unseen wording uses supplied one-hop questions and the third composed question. The second composed variant sometimes asks for two answers and is not selected. Span checks exclude case 1062 in both models, leaving 532 cases. No paraphrase outcome chooses a checkpoint.
- Access on/off uses the same generated first-stage bridge, second-stage prompt, and alias resolution, with no gold-bridge fallback.

## Scope

Coverage-guaranteed results are primary for MQuAKE in the abstract, Figure 1, and Table 1. Figure 2 shows the matched protocol comparison and the single-edit direct changes as grouped points with error bars. The standalone/joint LoRA table retains the original reference fits; their selection was against that reference residual, not the new confirmation. The reference and coverage protocols are never pooled.

Standalone and joint LoRA remain exploratory single-fit comparisons. The appendix reports nonzero residual-row coordinates plus LoRA factors as active parameters, alongside allocated counts including reserved rows. This is not a claim of matched active capacity or matched parameter efficiency.

The main manuscript retains four pages and the wrapfigure/wraptable style. The shortened geometry appendix preserves quantitative tables and key plots; `paper/final_geometry_extended.tex` retains the earlier extended source. Broad locality and autonomous decomposition remain outside the empirical claims.
