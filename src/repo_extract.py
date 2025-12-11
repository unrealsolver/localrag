import fnmatch
from pathlib import Path

from git import Repo


# Garbage that poisons RAG
EXCLUDE_PATTERNS = ["*.lock", "*/package-lock.json"]

textchars = bytearray({7, 8, 9, 10, 12, 13, 27} | set(range(0x20, 0x100)) - {0x7F})


def is_binary_string(chunk: bytes):
    return bool(chunk.translate(None, textchars))


def ensure_repo_clean_or_warn(repo_path: Path) -> None:
    """Just confirm it's a git repo and optionally warn about uncommitted changes."""
    repo = Repo(repo_path)
    if repo.is_dirty():
        print("[WARN] Repo has uncommitted changes; they will still be indexed.")


def is_file_excluded(file_path: Path):
    return any(
        fnmatch.fnmatch(file_path.as_posix(), pattern) for pattern in EXCLUDE_PATTERNS
    )


def is_file_binary(file_path: Path):
    with open(file_path, "rb") as fd:
        return is_binary_string(fd.read(1024))


def list_files_for_index(repo_root: Path) -> list[Path]:
    repo = Repo(repo_root)

    # -c  = cached (tracked)
    # -o  = others (untracked, but not ignored)
    # --exclude-standard = respect .gitignore, .git/info/exclude, global ignores
    files_rel = [
        Path(d) for d in repo.git.ls_files("-co", "--exclude-standard").splitlines()
    ]

    files_rel = filter(lambda d: not is_file_excluded(d), files_rel)

    files_abs = [repo_root / p for p in files_rel]

    files_abs = filter(lambda d: not is_file_binary(d), files_abs)
    files_abs = list(files_abs)
    print(f"[INFO] Collected {len(files_abs)} files for indexing.")
    return files_abs
