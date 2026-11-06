"""The README feature matrix and ``xrspatial.__all__`` describe the same
surface (issue #2930). A row without an export, or an export without a
row, fails here so the two cannot drift apart.
"""
import re
from pathlib import Path

import xrspatial
import xrspatial.mcda

README = Path(xrspatial.__file__).resolve().parents[1] / "README.md"

# Exported names that are not spatial operations and so have no matrix row.
EXPORTS_WITHOUT_ROW = {
    "LANDFORM_CLASSES",  # constant backing the Landforms row
    "mcda",  # subpackage; its functions are checked against xrspatial.mcda
}

_ROW = re.compile(r"^\|\s*\[([^\]]+)\]\(([^)]+)\)\s*(?:`([^`]+)`)?")


def _row_name(label, path, override):
    if override:
        return override
    m = re.search(r"\(([^()]+)\)\s*$", label)
    if m:
        return m.group(1).lower()
    return re.sub(r"[\s/-]+", "_", label.strip()).lower()


def _matrix_rows():
    """Yield (section, function name, link path) for each function row."""
    text = README.read_text(encoding="utf-8")
    start = text.index("#### Supported Spatial Functions")
    section = None
    for line in text[start:].splitlines():
        if line.startswith("### "):
            section = line.strip("# *")
            continue
        m = _ROW.match(line)
        if m and m.group(2).startswith("xrspatial/"):
            yield section, _row_name(*m.groups()), m.group(2)


def test_readme_matrix_matches_public_surface():
    top_rows, mcda_rows = {}, {}
    for section, name, path in _matrix_rows():
        if path.startswith("xrspatial/experimental/"):
            continue  # experimental: documented, deliberately not exported
        (mcda_rows if "MCDA" in section else top_rows)[name] = path

    exports = set(xrspatial.__all__) - EXPORTS_WITHOUT_ROW
    mcda_exports = set(xrspatial.mcda.__all__)

    missing_export = sorted(n for n in top_rows if n not in exports)
    missing_row = sorted(exports - set(top_rows))
    mcda_missing_export = sorted(n for n in mcda_rows if n not in mcda_exports)

    assert not missing_export, (
        "README rows with no matching name in xrspatial.__all__ "
        f"(add the export or a `name` override on the row): {missing_export}")
    assert not missing_row, (
        "xrspatial.__all__ names with no README matrix row "
        f"(add a row or list it in EXPORTS_WITHOUT_ROW): {missing_row}")
    assert not mcda_missing_export, (
        f"MCDA README rows not in xrspatial.mcda.__all__: {mcda_missing_export}")


def test_all_names_resolve():
    for name in xrspatial.__all__:
        assert hasattr(xrspatial, name), name
