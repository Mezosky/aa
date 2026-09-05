# Automatic generated-entity access

One evaluation experiment, with no new training, on the six coverage-guaranteed
MQuAKE residual fits (Llama and Qwen, seeds 13, 37, 71; 533 cases each).

The original composed question is placed under a single generic instruction:

> Answer the question by first naming one intermediate entity needed to answer it,
> then give the final answer. Use exactly two lines and no explanation:
> Intermediate: <entity>
> Answer: <final answer>

The assistant is prefilled with `Intermediate:`. The source residual is active
in the question in both conditions. Greedy generation produces the intermediate
field, stopping at its first newline (32-token budget). The whole field is
resolved against the existing global, unambiguous alias dictionary. Neither a
gold bridge, a constituent question, nor a second-relation template is supplied.
Unknown/ambiguous names, reserved zero rows, invalid token spans and non-roundtrip
tokenizations receive no effective additional update. Every case stays in the
denominator. EOS or an exhausted entity budget without a newline is a format
failure, not an excluded case.

At the newline, both conditions discard their old KV cache and replay the same
original prefix and generated token IDs. The lookup-on replay adds the resolved
row to the generated entity's token embeddings; lookup-off does not. Each entity
occurrence is normalized independently by the square root of its token count.
No sampled tokens are replaced or re-tokenized. Continuation remains in the same
assistant response to the composed question, with 32 further tokens and greedy
decoding; no new question or answer cue is appended. Lookup is applied only to
the completed intermediate field, not arbitrary entities in the final answer.

The paired final answer is the exact alias-matched `Answer:` field on the second
line. Complete-path success requires both the generated intermediate and that
answer to be correct. Bridge generation is identical across the paired branches
by construction. A separately labelled formatting-sensitivity check permits a
single `Answer:` line later in the response; it never searches for gold strings.
This does not change the generation protocol or replace the strict main score.

Audits compare replayed cached logits against full-prefix uncached decoding on
tiny FP32 Llama/Qwen models and actual BF16 model continuations. No-cache and
cached BF16 reductions need not be numerically identical; both maximum logit
differences and greedy agreement are saved. Every zero/absent-update pair must
have identical final token IDs. Checkpoint and dataset hashes, exact token IDs,
resolved spans, full responses and all failures are retained in
`outputs/<model-root>/automatic_access_v1/`.

## Reproduction

```bash
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m geometry_llm evaluate_automatic_access --config configs/config_mquake_cf_llama.yaml --root outputs/265b7bb1723b
CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m geometry_llm evaluate_automatic_access --config configs/config_mquake_cf_qwen.yaml --root outputs/86cc2e353f23
.venv/bin/python -m geometry_llm make_automatic_access_report
.venv/bin/python -m pytest -q
tectonic -o paper paper/main.tex
```

## Completed results

Means over the three existing fits; all rates use 533 cases per fit.

| Model | Correct intermediate | Final, off | Final, on | Complete path, off | Complete path, on |
| --- | ---: | ---: | ---: | ---: | ---: |
| Llama | 22.5% | 0.8% | 2.2% | 0.0% | 1.4% |
| Qwen | 9.9% | 1.1% | 1.6% | 0.0% | 0.4% |

Paired final gains: Llama +1.4 percentage points [0.0, 3.2]; Qwen +0.5
[0.1, 1.1], using 5,000 joint bridge/answer-group and fit bootstrap draws.
Llama's interval touches zero. These are small observed gains, not a robust
solution to consequence propagation.

The correct bridge's nonzero row is activated on 21.7% and 9.5% of cases.
Within that common paired subset, final accuracy with lookup is only 6.6%
and 4.6%, versus zero without lookup. Answer-field compliance is 97.5% and
97.3% on this subset. Relaxing the field's line-position requirement does
not change the paired gains. Thus failures include both generating the
intermediate and using the edit in the new continuation context; missing
lookup alone is not a sufficient explanation.

Counts, numerical cache audits, diagnostics, confidence intervals and plots:
`outputs/automatic_access_report/summary.json`. Full responses and original
token IDs remain in the model-specific `automatic_access_v1` directories.
The paper reports the control in Figure 2 and Appendix I.
