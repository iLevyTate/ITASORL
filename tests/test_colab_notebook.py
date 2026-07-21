"""Structural checks for the Colab runner notebook (`notebooks/colab_gpu.ipynb`).

The notebook is hand-edited JSON. These tests guard the invariants that keep
"Runtime -> Run all" safe and that enforce the make-a-copy-first workflow. They
never execute a cell (Colab, a GPU, and Google Drive are all unavailable in CI),
so they only inspect the notebook's source.
"""
import json
from pathlib import Path

NB_PATH = Path(__file__).resolve().parent.parent / "notebooks" / "colab_gpu.ipynb"


def _load():
    return json.loads(NB_PATH.read_text())


def _src(cell):
    # nbformat allows source as a single string or a list of lines; this
    # notebook mixes both, so normalise before matching.
    s = cell["source"]
    return s if isinstance(s, str) else "".join(s)


def _code_cells(nb):
    return [c for c in nb["cells"] if c["cell_type"] == "code"]


def test_notebook_is_valid_nbformat4():
    nb = _load()
    assert nb["nbformat"] == 4
    assert nb["cells"], "notebook has no cells"
    for cell in nb["cells"]:
        assert cell["cell_type"] in {"markdown", "code"}


def test_every_code_cell_compiles():
    # Colab form magics (`# @title` / `# @param`) are plain comments, so each
    # code cell is valid standalone Python. Catches an edit that corrupts a cell.
    nb = _load()
    for i, cell in enumerate(_code_cells(nb)):
        compile(_src(cell), f"<cell {i}>", "exec")


def test_copy_guard_is_the_first_code_cell():
    # The make-a-copy guard must run before any heavy work so "Run all" stops at
    # it - i.e. before the config form clones the repo or `run_e2e.py` launches.
    nb = _load()
    code = _code_cells(nb)
    guard = _src(code[0])

    assert "I_MADE_A_COPY" in guard
    assert "Save a copy in Drive" in guard
    assert "colab_gpu.ipynb" in guard  # the canonical read-only name it checks
    assert "raise" in guard            # it can actually halt Run all

    rest = "\n".join(_src(c) for c in code[1:])
    assert "run_e2e.py" in rest, "the experiment-run cell moved or vanished"
    assert "run_e2e.py" not in guard, "the guard must precede the run cell"


def test_copy_guard_degrades_gracefully():
    # If Colab internals change so the notebook name can't be read, the guard
    # must warn rather than block - it only raises when it *positively* sees the
    # pristine GitHub filename, and it is skipped entirely off-Colab.
    nb = _load()
    guard = _src(_code_cells(nb)[0])
    assert "_in_colab" in guard
    assert "Continuing anyway" in guard


def test_intro_documents_make_a_copy_first():
    nb = _load()
    markdown = "\n".join(_src(c) for c in nb["cells"] if c["cell_type"] == "markdown")
    assert "Save a copy in Drive" in markdown
