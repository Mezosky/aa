"""Regression checks for command dispatch, config compatibility and ignore rules."""
import importlib.util
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from geometry_llm import commands
from geometry_llm.config import config_id, load_config, resolve_config_path


ROOT = Path(__file__).resolve().parents[1]


def test_every_command_has_a_module_and_only_registered_commands_exist():
    paths = {p.stem for p in (ROOT / "geometry_llm/commands").glob("*/*.py") if p.name != "__init__.py"}
    assert paths == set(commands.COMMANDS)
    for module in commands.COMMANDS.values():
        assert importlib.util.find_spec(module) is not None


def test_launcher_forwards_arguments_and_restores_argv(monkeypatch):
    previous = sys.argv
    called = []
    def imported(name):
        called.append(name)
        return SimpleNamespace(main=lambda: called.append(list(sys.argv)))
    monkeypatch.setattr(commands.importlib, "import_module", imported)
    assert commands.main(["train_delta.py", "--config", "config_llama.yaml", "--set", "training.epochs=1"]) == 0
    assert called == [commands.COMMANDS["train_delta"],
        ["python -m geometry_llm train_delta", "--config", "config_llama.yaml", "--set", "training.epochs=1"]]
    assert sys.argv is previous


def test_report_help_and_unknown_commands_never_import(monkeypatch, capsys):
    def unexpected(_):
        raise AssertionError("Help must not load or execute a report")
    monkeypatch.setattr(commands.importlib, "import_module", unexpected)
    for name in commands.NO_ARGUMENT_COMMANDS:
        assert commands.main([name, "--help"]) == 0
    with pytest.raises(SystemExit) as error:
        commands.main(["does_not_exist"])
    assert error.value.code == 2
    assert "usage:" in capsys.readouterr().out


def test_module_help_works_without_loading_research_dependencies():
    result = subprocess.run([sys.executable, "-m", "geometry_llm", "--help"], cwd=ROOT,
        capture_output=True, text=True, check=True)
    assert all(name in result.stdout for name in commands.COMMANDS)


def test_legacy_configs_preserve_content_hashes(monkeypatch, tmp_path):
    for path in (ROOT / "configs").glob("*.yaml"):
        assert load_config(path.name) == load_config(path)
        assert config_id(load_config(path.name)) == config_id(load_config(path))
    monkeypatch.chdir(tmp_path)
    assert resolve_config_path("config_llama.yaml") == ROOT / "configs/config_llama.yaml"
    with pytest.raises(FileNotFoundError):
        load_config("missing/config_llama.yaml")


def test_local_explicit_config_takes_precedence(tmp_path, monkeypatch):
    path = tmp_path / "config.yaml"
    path.write_text("model: {name: test}\n")
    monkeypatch.chdir(tmp_path)
    assert load_config("config.yaml")["model"]["name"] == "test"


def test_gitignore_omits_artifacts_but_not_source(tmp_path):
    # Use a throwaway repository; do not initialize or change the user's Git state.
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / ".gitignore").write_bytes((ROOT / ".gitignore").read_bytes())
    ignored = ["paper/main.tex", "paper/main.pdf", "outputs/run/predictions.jsonl", ".venv/bin/python"]
    kept = ["geometry_llm/config.py", "configs/config.yaml", "docs/RESULTS.md", "README.md"]
    result = subprocess.run(["git", "-C", str(tmp_path), "check-ignore", "--no-index", "--stdin"],
        input="\n".join(ignored + kept) + "\n", capture_output=True, text=True, check=True)
    assert result.stdout.splitlines() == ignored
