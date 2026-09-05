# Code layout

The source is organized by role. The reorganization does not retrain models,
change experiment settings, or move existing paper and result artifacts.

```text
geometry_llm/
  modeling.py, data.py, metrics.py, ...  shared experiment implementation
  commands/
    data/        dataset preparation, inspection, and selection
    training/    residual, LoRA, joint-adapter, and calibration fits
    evaluation/  predictions, access controls, and interventions
    analysis/    behavioral and representation-geometry measurements
    audits/      consistency, coverage, and statistical checks
    reports/     figures and tables from saved results
configs/         original YAML settings, contents unchanged
docs/            protocols, results, benchmarks, and review notes
tests/           unit and project-layout regression tests
paper/           local manuscript and compiled PDF; Git-ignored
outputs/         local predictions, checkpoints, and figures; Git-ignored
```

## Run commands

Run from the project root after installing `requirements.txt` in your environment:

```bash
python -m geometry_llm --help
python -m geometry_llm train_delta --help
python -m geometry_llm evaluate_automatic_access --config configs/config_mquake_cf_llama.yaml --root outputs/265b7bb1723b
python -m pytest -q
```

Command names and options are unchanged. Replace old `python NAME.py ...`
invocations with `python -m geometry_llm NAME ...`; an optional `.py` suffix
is also accepted by the launcher. The documented workflows have been updated.
Original top-level script paths and imports have moved into the corresponding
`geometry_llm.commands.<group>` package; there are no duplicate legacy wrappers.

Bare config names such as `--config config_llama.yaml` still resolve to
`configs/config_llama.yaml`. Explicit paths take precedence, and config contents
and output hashes are unchanged. Data and output paths remain relative to the
working directory, so use the project root for the documented workflows.

The report commands still read existing `outputs/` artifacts and some write into
the local `paper/` folder. Ignoring those folders does not disable local reporting
or include their contents in a clone. Historical strings in cached outputs or
archived paper sources are preserved rather than rewritten.

## Version-control scope

`.gitignore` excludes `/paper/`, `/outputs/`, Python environments and caches, and
local credential files. It does not delete these files or untrack anything already
committed. The code repository is <https://github.com/Mezosky/aa>. Paper sources,
the standalone Overleaf upload ZIP, and generated results remain local and are
not included in this repository.
