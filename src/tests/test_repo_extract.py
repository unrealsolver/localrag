from pathlib import Path
import pytest

from ..repo_extract import is_file_excluded


@pytest.mark.parametrize(
    "path,is_filtered",
    [
        ["src/package-lock.json", True],
        ["package-lock.json", True],
        ["src/uv.lock", True],
        ["uv.lock", True],
        ["src/poetry.lock", True],
        ["src/package.json", False],
        ["src/pyproject.toml", False],
    ],
)
def test_is_file_excluded(path: str, is_filtered):
    assert is_file_excluded(Path(path)) is is_filtered
