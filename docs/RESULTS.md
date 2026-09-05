# Audited access results

The current evidence supports an access bottleneck in entity-indexed input adaptation. Independent local probes activate different rows; direct MQuAKE questions activate only the source row. An internal bridge representation does not perform an external lookup.

These results supersede the earlier claim that SOCRATES conditional composition is unchanged. Primary confirmation statistics are in `outputs/confirmation_report/summary.json`; the reference audit is in `outputs/access_report/summary.json`, with source predictions preserved.

## Primary MQuAKE-CF coverage confirmation

The final paper uses coverage-guaranteed fits as its primary MQuAKE evidence. All 892 unique facts are visited each epoch, and all 839 eligible entity rows receive updates, for three fits per model. Rates and epoch counts remain fixed to the earlier dataset-specific one-hop selection. No composed or paraphrase outcome selects a checkpoint.

| Model | Hop 1 | Hop 2 | Both facts | Direct | Pipeline | Stage 2 off |
|---|---:|---:|---:|---:|---:|---:|
| Llama | 44.9% | 38.1% | 15.4% | 1.1% | 15.5% | 0.6% |
| Qwen | 32.9% | 25.1% | 8.2% | 1.4% | 9.2% | 1.3% |

The paired access effects are **14.9 points [8.8, 21.1]** and **7.9 [5.2, 10.8]**. They survive guaranteed coverage, but are smaller than in the reference protocol. Coverage, deduplication, and per-fact weighting change together; this does not isolate the effect of coverage alone.

Single-edit controls use 525 role-separated cases and condition-consistent consequence targets. Bridge-only training leaves all direct prediction strings unchanged in both models. This includes 227 cases where the bridge edits change the original consequence. Source-only training leaves second-fact predictions unchanged but increases direct accuracy by 9.5 points [5.6, 13.7] for Llama and 3.6 [0.9, 6.5] for Qwen. These controls have one fit each, not three.

On 532 matched unseen question forms, local gains persist but weaken: joint constituent coverage changes from 15.5% to 11.8% in Llama and 8.2% to 3.2% in Qwen. This supports partial wording transfer, not fully template-independent factual installation. Direct question-form accuracy remains near 1%.

Audited statistics are in `outputs/confirmation_report/summary.json`; reproduction and protocol details are in [FINAL_CONFIRMATION.md](FINAL_CONFIRMATION.md). The older results below remain a separately labelled reference, including the exploratory LoRA comparisons and geometry.

## Automatic intermediate access without a supplied hop template

One new evaluation uses the same coverage-trained tables, original composed
questions, and frozen models. A generic two-line response format lets the model
generate its own intermediate entity. At its newline, the global alias dictionary
resolves the entity; paired continuations replay identical original token IDs with
the generated-span residual on or off. There is no gold bridge, constituent
question, or supplied second-hop relation template, and no further training.

| Model | Correct bridge | Final, lookup off | Final, lookup on | Complete path, off | Complete path, on |
|---|---:|---:|---:|---:|---:|
| Llama | 22.5% | 0.8% | 2.2% | 0.0% | 1.4% |
| Qwen | 9.9% | 1.1% | 1.6% | 0.0% | 0.4% |

The paired final changes are +1.4 points [0.0, 3.2] and +0.5 [0.1, 1.1].
Llama's interval touches zero. On cases where the correct bridge is generated and
its nonzero residual actually activated, final accuracy is only 6.6% and 4.6%
(versus zero with lookup off). Output-field compliance on these subsets exceeds
97%; a relaxed field-position check leaves paired gains unchanged. Automatic
access recovers some complete paths but does not reliably transfer the edit into
the new continuation context. This limits any explanation based only on missing lookup.

All 533 cases per fit are retained. Complete-path numerators are 6, 12, 5 for
Llama and 4, 2, 1 for Qwen, versus zero in every paired off branch. They are
repeated evaluations of the same cases, not independent additional cases.
All inactive pairs have identical output token IDs. Tiny-model cache equivalence
tests and all 48 sampled full-model greedy decisions pass; BF16 numerical logit
differences are preserved in the audits. See [AUTOMATIC_ACCESS.md](AUTOMATIC_ACCESS.md)
and `outputs/automatic_access_report/summary.json` for full counts and uncertainty.

## Reference MQuAKE-CF, dataset-specific selection

Both models use seeds 13, 37, and 71, one-hop validation, and the same 533 cases.

| Model | Hop 1 | Hop 2 | Both facts | Direct | Generated-bridge pipeline | Stage 2 off | Both facts in context |
|---|---:|---:|---:|---:|---:|---:|---:|
| Llama | 37.3% | 51.6% | 21.4% | 1.6% | 21.4% | 0.8% | 67.0% |
| Qwen | 25.6% | 34.1% | 9.4% | 1.2% | 10.8% | 1.4% | 85.2% |

Values are means over residual seeds. The frozen explicit-facts oracle is deterministic: 357/533 for Llama and 454/533 for Qwen.

The paired effect of enabling the second-stage residual is 20.6 percentage points for Llama (95% group/seed bootstrap interval 15.6–25.8) and 9.4 for Qwen (7.0–12.2). Complete-path success, requiring a correct generated bridge as well, is 20.4% and 9.4%. Disabling the second row gives zero complete-path successes in all six runs.

The pipeline uses a supplied second-relation template. A global alias dictionary resolves generated bridges without consulting the gold bridge. Unknown or ambiguous strings receive no row. Thus the experiment diagnoses access under supplied decomposition; it does not demonstrate autonomous question decomposition.

## Corrected conditional comparison

| Model | Frozen C | Seed 13 | Seed 37 | Seed 71 | Mean adapted C |
|---|---:|---:|---:|---:|---:|
| SOCRATES Llama | 7/29 (24.1%) | 17/74 | 30/86 | 29/82 | 31.1% |
| SOCRATES Qwen | 0/10 | 7/52 | 6/63 | 5/63 | 10.3% |
| MQuAKE Llama | 0/0 (undefined) | 3/126 | 2/128 | 3/88 | 2.5% |
| MQuAKE Qwen | 0/0 (undefined) | 0/51 | 0/52 | 1/48 | 0.7% |

Condition-specific fractions use different populations. On chains where both frozen and adapted models recover both constituents:

| SOCRATES model | Seed 13 frozen → adapted | Seed 37 | Seed 71 |
|---|---|---|---|
| Llama | 5/24 → 4/24 | 7/26 → 6/26 | 7/25 → 7/25 |
| Qwen | 0/10 → 3/10 | 0/9 → 2/9 | 0/10 → 3/10 |

Every Llama paired interval includes zero. Qwen has positive point estimates on small common sets. Neither finding establishes that composition is unchanged. Per-seed cluster intervals and all scaling/projection counts are in each root's `analysis/conditional_audit.json`.

## Shared-supervision audit

MQuAKE-CF-3k-v2 has 3,000 cases, including 1,135 with two new hops. Requiring two requested edits gives 536 candidates; IDs 17, 699, and 1099 fail unique formatted-span checks, leaving 533 identical selected cases for both models.

One table is shared across these cases per model and seed. The 1,066 constituent records represent 892 unique facts and 174 repetitions. No duplicate case IDs, conflicting entity-ID/relation targets, conflicting normalized-name/relation targets, conflicting exact prompts, or subject-name identity collisions were found. All 1,066 structural rewrite checks pass. Repeated facts are retained.

This is a structural audit, not a manual semantic audit of every generated question. See `outputs/265b7bb1723b/analysis/dataset_audit.json`.

## Adaptation and interpretation

The LoRA comparison matches the allocated residual parameter budget approximately and selects using constituent efficacy, never composed accuracy. The Llama seed-13 comparison achieves similar one-hop efficacy and a higher direct score. Exact selected settings, Qwen tuning, and paired uncertainty are generated in `outputs/access_report/lora_comparison.json`. This single-seed comparison does not establish general superiority of an adaptation method.

The jointly trained residual + LoRA control uses both full budgets and is not parameter-matched. Both models select epoch 4 using constituent efficacy only. Llama uses 8,675,328 trainable parameters and Qwen 7,648,256. These are single-fit controls, so their shaded 95% intervals reflect bridge/answer-group uncertainty, not optimization variability.

| Joint control | First hop | Second hop | Direct | Conditional count |
|---|---:|---:|---:|---:|
| Llama | 46.0% | 60.4% | 18/533 (3.4%) | 7/159 |
| Qwen | 37.5% | 54.6% | 17/533 (3.2%) | 4/114 |

Both joint fits improve constituent accuracy over standalone LoRA, but their direct scores are lower in these runs. Paired joint-versus-LoRA direct intervals include zero; this is not evidence for a general ranking. The joint control reinforces the distinction between local fitting and consequence transfer. Paired component removals and exact settings are reported in the appendix and `outputs/access_report/joint_comparison.json`.

Failed donor swaps establish non-portability under base-entity mismatch: E_i + delta_j differs from E_j + delta_j. They do not establish absence of factual information. Geometry describes source-update propagation, not access to an uninjected bridge row. CLUTRR remains a separate shared-role intervention in the appendix.

## Final-state structure, angles, and norms

Matched MQuAKE final-prompt representations show different broad structures. Covariance-entropy effective rank changes from 217.6 to 209.8 (residual), 75.6 (LoRA), and 110.4 (joint) in Llama; Qwen changes from 155.5 to 131.5, 45.2, and 76.2. LoRA concentrates variance and narrows the average cone, but these finite-sample statistics do not establish a low-dimensional factual manifold.

LoRA rotates final states by 70.5° in Llama and 51.7° in Qwen while their mean paired norm ratios are only 1.025 and 1.072. Relative displacement norms are 1.168 and 0.907. Changes are therefore substantially directional, with negative radial and large tangential components, rather than simple rescaling. Update-to-base angles are distinct from state rotation angles; zero update directions are excluded and counted explicitly.

The joint adapter with residuals disabled retains CKA 0.970/0.973 against the full joint adapter despite losing most constituent-fact coverage. Coarse shape is therefore insufficient to explain factual access. Relation structure is strong across conditions; answer-specific excess over relation-matched permutations is small and also occurs in controls. Geometric behavioral prediction is model- and outcome-dependent after controlling for frozen geometry, with gold-answer identities held out from post-hoc probe training.

The sampling audit found 55 unvisited source rows in each reference residual fit. On the common 478 active-source queries, residual-versus-LoRA ranks remain 203.2 versus 75.9 (Llama) and 124.9 versus 45.6 (Qwen). Thus zero rows do not explain the broad spectral contrast. Detailed tables, grouped uncertainty, raw norms and angles, active-only probe checks, and UMAP fidelity diagnostics are in Appendix F and `outputs/final_geometry_report/`.
