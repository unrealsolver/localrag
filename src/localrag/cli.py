from __future__ import annotations

import argparse
from pathlib import Path


class CLIArgs(argparse.Namespace):
    path: Path
    llm: str
    em: str
    reindex: bool
    debug: bool


def existing_dir(path_str: str) -> Path:
    """Argparse type that ensures the path is an existing directory."""
    p = Path(path_str)
    if not p.is_dir():
        raise argparse.ArgumentTypeError(f"{path_str!r} is not an existing directory")
    return p


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Example CLI with path + optional LLM/EM overrides."
    )

    # Positional argument: path (directory)
    parser.add_argument(
        "path",
        type=existing_dir,
        help="Path to the directory to process.",
    )

    # Optional: --llm (LLM model name override)
    parser.add_argument(
        "--llm",
        metavar="LLM_MODEL",
        help="Override the default language model name.",
        default="llama3.2:1b",
    )

    # Optional: --em (embedding model name override)
    parser.add_argument(
        "--em",
        metavar="EMBED_MODEL",
        help="Override the default embedding model name.",
        default="BAAI/bge-small-en-v1.5",
    )

    parser.add_argument(
        "-r",
        "--reindex",
        help="Force reindex repo",
        action="store_true",
    )

    parser.add_argument(
        "-d",
        "--debug",
        help="Debug mode",
        action="store_true",
    )

    return parser


def parse_args() -> CLIArgs:
    parser = build_parser()
    return parser.parse_args(namespace=CLIArgs())
