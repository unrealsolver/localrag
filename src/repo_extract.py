from dataclasses import dataclass
import fnmatch
from pathlib import Path

from git import Repo


# Garbage that poisons RAG
EXCLUDE_PATTERNS = ["*.ipynb", "*.lock", "package-lock.json"]
EXT_TO_LANG = {
    "py": "python",
    "ts": "typescript",
    "tsx": "typescript",
    "js": "javascript",
    "jsx": "javascript",
    "java": "java",
    "go": "go",
    "rs": "rust",
    "cpp": "cpp",
    "cc": "cpp",
    "cxx": "cpp",
    "c": "c",
    "cs": "csharp",
    "kt": "kotlin",
    "kts": "kotlin",
    "json": "json",
    "toml": "toml",
    "yaml": "yaml",
    "yml": "yaml",
    "md": "markdown",
}

textchars = bytearray({7, 8, 9, 10, 12, 13, 27} | set(range(0x20, 0x100)) - {0x7F})


@dataclass
class IndexableFile:
    root: Path
    "file root"

    rel: Path
    "relative path"

    @property
    def abs(self) -> Path:
        "absolute path"
        return self.root / self.rel

    @property
    def ext(self) -> str:
        "file extension"
        return str(self.rel).split(".")[-1]

    def read(self):
        with open(self.abs, "rt") as fd:
            return fd.read()

    @property
    def lang(self):
        "file language"
        return EXT_TO_LANG.get(self.ext, None)


def is_binary_string(chunk: bytes):
    return bool(chunk.translate(None, textchars))


def ensure_repo_clean_or_warn(repo_path: Path) -> None:
    """Just confirm it's a git repo and optionally warn about uncommitted changes."""
    repo = Repo(repo_path)
    if repo.is_dirty():
        print("[WARN] Repo has uncommitted changes; they will still be indexed.")


def is_file_excluded(file_path: Path):
    return any(fnmatch.fnmatch(file_path.name, pattern) for pattern in EXCLUDE_PATTERNS)


def is_file_binary(file_path: Path):
    assert file_path.is_absolute()
    with open(file_path, "rb") as fd:
        return is_binary_string(fd.read(1024))


def list_files_for_index(repo_root: Path) -> list[IndexableFile]:
    repo = Repo(repo_root)

    # -c  = cached (tracked)
    # -o  = others (untracked, but not ignored)
    # --exclude-standard = respect .gitignore, .git/info/exclude, global ignores
    git_files = repo.git.ls_files("-co", "--exclude-standard").splitlines()
    files = [IndexableFile(rel=d, root=repo_root) for d in git_files]

    files = filter(lambda d: not is_file_excluded(d.rel), files)
    files = filter(lambda d: not is_file_binary(d.abs), files)
    files = list(files)

    print(f"[INFO] Collected {len(files)} files for indexing.")
    return files
