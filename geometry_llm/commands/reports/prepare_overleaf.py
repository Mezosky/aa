"""Package the active paper and its local dependencies for an Overleaf upload."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo


DEPENDENCY = re.compile(
    r"\\(inputtablerows|input|include|includegraphics|bibliography|bibliographystyle|usepackage|RequirePackage)"
    r"(?:\[[^\]]*\])?\s*\{([^{}]+)\}"
)


def project_path(root: Path, relative: str, suffix: str = "") -> Path:
    """Reject external references, including escaping symlinks."""
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Paper dependency must be local: {relative}")
    if suffix and not path.suffix:
        path = path.with_suffix(suffix)
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError(f"Paper dependency escapes project: {relative}")
    return resolved


def collect_dependencies(paper: Path) -> list[Path]:
    """Follow the literal dependency forms used by this paper, not arbitrary TeX."""
    paper = paper.resolve()
    seen = set()
    def visit(path):
        if path in seen:
            return
        if not path.is_file():
            raise FileNotFoundError(f"Missing paper dependency: {path}")
        seen.add(path)
        if path.suffix not in {".tex", ".sty"}:
            return
        text = re.sub(r"(?<!\\)%[^\n]*", "", path.read_text(encoding="utf-8"))
        for command, value in DEPENDENCY.findall(text):
            suffix = {"input": ".tex", "inputtablerows": ".tex", "include": ".tex",
                "bibliography": ".bib", "bibliographystyle": ".bst",
                "usepackage": ".sty", "RequirePackage": ".sty"}.get(command, "")
            names = value.split(",") if command in {
                "bibliography", "usepackage", "RequirePackage"} else [value]
            for name in names:
                dependency = project_path(paper, name.strip(), suffix)
                # Distribution-provided packages/styles need no uploaded copy.
                if command in {"usepackage", "RequirePackage", "bibliographystyle"} and not dependency.exists():
                    continue
                visit(dependency)
    visit(paper / "main.tex")
    if (paper / "README.md").is_file():
        seen.add(project_path(paper, "README.md"))
    return sorted(seen, key=lambda path: path.relative_to(paper).as_posix())


def refresh_images(paper: Path, repository: Path) -> int:
    """Explicitly copy the mapped source PDFs; never regenerate or edit plots."""
    paper, repository = paper.resolve(), repository.resolve()
    mapping = json.loads((paper / "images/manifest.json").read_text())["images"]
    copies = []
    for target, source in mapping.items():
        if not target.startswith("images/") or not source.startswith("outputs/"):
            raise ValueError("Figure mapping must copy outputs/ files into images/")
        destination = project_path(paper, target)
        origin = project_path(repository, source)
        if not origin.is_file():
            raise FileNotFoundError(origin)
        copies.append((origin, destination))
    for origin, destination in copies:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(origin, destination)
    return len(copies)


def package_project(paper: Path, output: Path) -> list[str]:
    paper, output = paper.resolve(), output.resolve()
    files = collect_dependencies(paper)
    if output.suffix.lower() != ".zip" or output in files:
        raise ValueError("Export must be a .zip file, not a manuscript dependency")
    output.parent.mkdir(parents=True, exist_ok=True)
    names = []
    with ZipFile(output, "w", compression=ZIP_DEFLATED) as archive:
        for path in files:
            name = path.relative_to(paper).as_posix()
            # Stable metadata makes repeated exports byte-identical.
            info = ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, path.read_bytes())
            names.append(name)
    return names


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-dir", type=Path, default=Path("paper"))
    parser.add_argument("--output", type=Path, help="Defaults to PAPER_DIR/overleaf.zip")
    parser.add_argument("--refresh-images", action="store_true",
        help="Copy the mapped PDFs from local outputs before packaging")
    args = parser.parse_args()
    if args.refresh_images:
        count = refresh_images(args.paper_dir, Path(__file__).resolve().parents[3])
        print(f"Copied {count} existing figure PDFs into {args.paper_dir / 'images'}")
    destination = args.output or args.paper_dir / "overleaf.zip"
    names = package_project(args.paper_dir, destination)
    print(f"Packaged {len(names)} files in {destination}; main document: main.tex")


if __name__ == "__main__":
    main()
