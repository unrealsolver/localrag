import os
from pathlib import Path

from git import Repo

from llama_index.core import (
    PromptTemplate,
    VectorStoreIndex,
    StorageContext,
    Settings,
    SimpleDirectoryReader,
    load_index_from_storage,
)
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from cli import parse_args

DEBUG = False

PERSIST_DIR = Path("../.llamaindex_storage")  # local metadata

QDRANT_URL = "http://localhost:6334"
QDRANT_COLLECTION_NAME = "git_repo_index"

# Ignore typical junk (you can customize)
DEFAULT_EXCLUDE_DIRS = {
    ".git",
    ".idea",
    ".vscode",
    "__pycache__",
    "node_modules",
    "dist",
    "build",
    "experiemnts",
    ".local_kafka",
    ".venv",
    ".local_storage",
}

DEFAULT_EXTENSIONS = {
    ".py",
    ".ts",
    ".tsx",
    ".js",
    ".jsx",
    ".java",
    ".kt",
    ".kts",
    ".go",
    ".rs",
    ".c",
    ".cpp",
    ".h",
    ".hpp",
    ".md",
    ".txt",
    ".yaml",
    ".yml",
    ".toml",
    "Makefile",
}

qa_prompt_tmpl = PromptTemplate(
    """You are a strict documentation assistant.
You can ONLY answer using the information in the provided context.
If the answer is not clearly in the context, say exactly: "I don't know."

Question: {query_str}

Context:
{context_str}

Answer briefly and do NOT write code unless explicitly asked."""
)

Settings.llm = Ollama(
    model="llama3.2:1b",  # name as exposed by Ollama
    base_url="http://localhost:11434",  # default Ollama endpoint
    request_timeout=300.0,  # seconds
)

Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# ---------- HELPERS ----------


def ensure_repo_clean_or_warn(repo_path: Path) -> None:
    """Just confirm it's a git repo and optionally warn about uncommitted changes."""
    repo = Repo(repo_path)
    if repo.is_dirty():
        print("[WARN] Repo has uncommitted changes; they will still be indexed.")


def list_files_for_index(root: Path) -> list[Path]:
    """Collect files to index based on extension + exclusions."""
    files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        # filter out excluded directories
        dirnames[:] = [d for d in dirnames if d not in DEFAULT_EXCLUDE_DIRS]

        for fn in filenames:
            p = Path(dirpath) / fn
            if p.suffix.lower() in DEFAULT_EXTENSIONS:
                files.append(p)

    print(f"[INFO] Collected {len(files)} files for indexing.")
    return files


# ---------- MAIN INDEXING LOGIC ----------
def get_qdrant_vector_store() -> QdrantVectorStore:
    client = QdrantClient(url=QDRANT_URL, prefer_grpc=True)
    return QdrantVectorStore(
        client=client,
        collection_name=QDRANT_COLLECTION_NAME,
    )


def read_documents(repo_path):
    # 1. (Optional) sanity check repo
    ensure_repo_clean_or_warn(repo_path)

    # 2. Collect files to index
    files = list_files_for_index(repo_path)

    # 3. Use LlamaIndex file reader
    #    You can pass file paths directly; SimpleDirectoryReader will just load them.
    return SimpleDirectoryReader(input_files=[str(f) for f in files]).load_data()


def build_index(documents) -> VectorStoreIndex:
    vector_store = get_qdrant_vector_store()
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store,
    )
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True,
    )
    # save metadata (docstore, index config, etc.)
    index.storage_context.persist(persist_dir=str(PERSIST_DIR))
    return index


def load_existing_index() -> VectorStoreIndex:
    vector_store = get_qdrant_vector_store()
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store,
        persist_dir=str(PERSIST_DIR),
    )
    index = load_index_from_storage(storage_context)
    return index


def build_or_load_index(documents=None) -> VectorStoreIndex:
    if PERSIST_DIR.exists():
        # assumes Qdrant collection still exists too
        print("[INFO] Loading existing index...")
        return load_existing_index()
    else:
        if documents is None:
            raise ValueError("Need documents on first run to build index.")
        print("[INFO] Building index for the first time...")
        return build_index(documents)


def interactive_query(index: VectorStoreIndex) -> None:
    print("\n[INFO] Enter interactive query mode. Type 'exit' to quit.\n")
    query_engine = index.as_query_engine(
        similarity_top_k=10,
        llm=Settings.llm,
        text_qa_template=qa_prompt_tmpl,
        streaming=True,
    )

    while True:
        q = input("Query> ").strip()
        if not q or q.lower() in {"exit", "quit"}:
            break
        resp = query_engine.query(q)

        if DEBUG:
            for i, n in enumerate(resp.source_nodes, 1):
                print(f"\n--- Source {i} | score={n.score} ---")
                print(n.metadata.get("file_path", "no path"))
                print(n.text[:600])

        print("\n----- ANSWER -----")
        for token in resp.response_gen:
            print(token, end="", flush=True)
        print("------------------\n")


if __name__ == "__main__":
    args = parse_args()

    repo_path = args.path.resolve()
    if not repo_path.exists():
        raise SystemExit(f"Repo path does not exist: {repo_path}")

    documents = read_documents(repo_path)
    index = build_or_load_index(documents)
    interactive_query(index)
