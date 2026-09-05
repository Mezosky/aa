# Access and conditional-accuracy review

1. Conditional claims corrected throughout the active paper. Every seed, numerator/denominator, cluster interval, and scale/projection condition is reported. Llama SOCRATES averages 31.1%, not the selected seed's 23.0%.
2. Injection verified: every direct MQuAKE prompt has zero active bridge rows. An executed generated-bridge pipeline and paired second-stage residual-off control are complete for both models and all three seeds.
3. Common-set comparisons use the intersection where both frozen and adapted models recover both constituents. Empty MQuAKE intersections are undefined, not zero.
4. Shared MQuAKE supervision audited by IDs, normalized names, relations, and exact prompts. Selection is 3,000 source → 1,135 two-hop → 536 two-edit → 533 span-valid cases; 174 repeated facts are retained and no incompatible targets are found.
5. Dataset-specific one-hop validation and three residual seeds are complete for Llama and Qwen. LoRA budget and efficacy comparisons are recorded separately, with only local outcomes selecting settings. Joint residual + LoRA controls and both component-removal evaluations are also complete for both models. They retain both full parameter budgets and use a single fit per model, explicitly distinguished from parameter-matched and multi-replicate comparisons.
6. Donor swaps are described as portability tests under base-entity mismatch, not tests proving absence of factual information.
7. The four-page main paper focuses on access controls, the common-set comparison, and the contribution. Geometry and CLUTRR are in an indexed appendix. The pooling definition, actual anchor normalization, exact settings, and baseline provenance are explicit. Existing composability metrics are credited to prior work.

The paper retains compact wrapfigures, a colored wraptable, and a row of additional behavioral plots. Figures are regenerated from saved predictions by `geometry_llm/commands/reports/make_access_report.py`.

The broader earlier roadmap still contains uncompleted studies (additional SOCRATES compositions, paraphrases, locality, and randomized layerwise controls). They are not evidence for the active paper's conclusions.
