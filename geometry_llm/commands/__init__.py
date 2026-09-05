"""Discoverable command registry; importing it does not load models or datasets."""
from __future__ import annotations

import argparse
import importlib
import sys


COMMAND_GROUPS = {
    "data": (
        "build_graph", "inspect_dataset", "prepare_clutrr", "select_composition",
    ),
    "training": (
        "calibrate_lora", "run_clutrr_delta", "run_final_confirmation",
        "run_geometric_calibration", "run_lora_baseline", "train_delta",
    ),
    "evaluation": (
        "evaluate", "evaluate_access", "evaluate_automatic_access", "evaluate_baseline",
        "evaluate_oracle", "evaluate_selected", "run_interventions", "run_residual_swaps",
    ),
    "analysis": (
        "analyze_behavior", "analyze_embedding_structure", "analyze_final_geometry",
        "analyze_geometric_memory", "analyze_geometry", "analyze_layers", "analyze_relations",
        "compare_embedding_structures", "compare_geometric_calibrations", "compare_models",
        "extract_final_geometry", "measure_geometry_angles_norms",
    ),
    "audits": (
        "audit_active_geometry", "audit_composition", "audit_final_confirmation",
        "audit_mquake", "smoke_test",
    ),
    "reports": (
        "make_access_report", "make_automatic_access_report", "make_confirmation_report",
        "make_final_geometry_report", "make_paper_figures", "plot_results", "prepare_overleaf",
    ),
}
COMMANDS = {
    name: f"geometry_llm.commands.{group}.{name}"
    for group, names in COMMAND_GROUPS.items() for name in names
}
# These legacy report generators use fixed artifact paths and have no parser.
# Intercept help before importing them so --help cannot regenerate artifacts.
NO_ARGUMENT_COMMANDS = frozenset({
    "make_access_report", "make_automatic_access_report",
    "make_final_geometry_report", "make_paper_figures",
})


def main(argv: list[str] | None = None) -> int:
    catalog = "Commands by purpose:\n" + "\n".join(
        f"  {group}:\n    " + "\n    ".join(names)
        for group, names in COMMAND_GROUPS.items()
    )
    parser = argparse.ArgumentParser(
        prog="python -m geometry_llm",
        description="Run the research workflows from the project root.",
        epilog=catalog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("command", nargs="?", help="Existing command name, with optional .py suffix")
    parser.add_argument("arguments", nargs=argparse.REMAINDER, help="Arguments passed unchanged to the command")
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    name = args.command.removesuffix(".py")
    if name not in COMMANDS:
        parser.error(f"unknown command {args.command!r}; use --help to list commands")
    if name in NO_ARGUMENT_COMMANDS:
        if args.arguments in (["--help"], ["-h"]):
            print(f"usage: python -m geometry_llm {name}\n\n"
                  "Regenerate reports from existing local artifacts using fixed paths.\n"
                  "This command takes no arguments. Run from the project root.\n"
                  f"Source: {COMMANDS[name].replace('.', '/')}.py")
            return 0
        if args.arguments:
            parser.error(f"{name} takes no arguments")
    previous_argv = sys.argv
    try:
        sys.argv = [f"python -m geometry_llm {name}", *args.arguments]
        result = importlib.import_module(COMMANDS[name]).main()
        return result if isinstance(result, int) else 0
    finally:
        sys.argv = previous_argv
