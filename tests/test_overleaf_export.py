from pathlib import Path
from zipfile import ZipFile

import pytest

from geometry_llm.commands.reports.prepare_overleaf import collect_dependencies, package_project


def sample_project(root):
    (root / 'sections').mkdir()
    (root / 'images').mkdir()
    (root / 'generated').mkdir()
    (root / 'main.tex').write_text(
        '\\documentclass{article}\n\\usepackage{graphicx,localstyle}\n'
        '\\input{sections/intro}\n\\bibliographystyle{unsrtnat}\n\\bibliography{references}\n')
    (root / 'sections/intro.tex').write_text(
        '% \\input{not-a-real-dependency}\n\\inputtablerows{generated/rows.tex}\n'
        '\\includegraphics[width=\\linewidth]{images/plot.pdf}\n')
    (root / 'generated/rows.tex').write_text('A & B \\\\\n')
    (root / 'images/plot.pdf').write_bytes(b'%PDF-test-fixture')
    (root / 'references.bib').write_text('@article{key,title={Example}}\n')
    (root / 'localstyle.sty').write_text('\\RequirePackage{amsmath}\n')
    (root / 'main.pdf').write_bytes(b'compiled output must not be uploaded')
    (root / 'old_draft.tex').write_text('\\input{../outputs/missing}\n')
    (root / 'main.aux').write_text('stale build file')


def test_zip_contains_only_active_local_dependencies_and_is_reproducible(tmp_path):
    sample_project(tmp_path)
    output = tmp_path / 'overleaf.zip'
    names = package_project(tmp_path, output)
    assert set(names) == {'main.tex', 'sections/intro.tex', 'generated/rows.tex',
                          'images/plot.pdf', 'references.bib', 'localstyle.sty'}
    first = output.read_bytes()
    assert package_project(tmp_path, output) == names
    assert output.read_bytes() == first
    with ZipFile(output) as archive:
        assert archive.testzip() is None
        assert archive.read('images/plot.pdf') == (tmp_path / 'images/plot.pdf').read_bytes()


def test_external_and_missing_dependencies_fail_before_export(tmp_path):
    sample_project(tmp_path)
    intro = tmp_path / 'sections/intro.tex'
    intro.write_text('\\includegraphics{../outputs/plot.pdf}')
    with pytest.raises(ValueError, match='must be local'):
        package_project(tmp_path, tmp_path / 'overleaf.zip')
    assert not (tmp_path / 'overleaf.zip').exists()
    intro.write_text('\\input{missing}')
    with pytest.raises(FileNotFoundError, match='Missing paper dependency'):
        collect_dependencies(tmp_path)


def test_export_cannot_overwrite_source(tmp_path):
    sample_project(tmp_path)
    original = (tmp_path / 'main.tex').read_bytes()
    with pytest.raises(ValueError):
        package_project(tmp_path, tmp_path / 'main.tex')
    assert (tmp_path / 'main.tex').read_bytes() == original
