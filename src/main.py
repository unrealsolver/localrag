import logging
import shutil
from pathlib import Path


from llama_index.core import (
    PromptTemplate,
    VectorStoreIndex,
    StorageContext,
    Settings,
    load_index_from_storage,
)
from llama_index.core.schema import BaseNode
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from cli import CLIArgs, parse_args
from repo_extract import ensure_repo_clean_or_warn, is_git_repo, list_files_for_index
from chunking import chunk
from util import format_context_node


logger = logging.getLogger(__name__)

args: CLIArgs

PERSIST_DIR = Path("../.llamaindex_storage")  # local metadata

QDRANT_URL = "http://localhost:6334"
QDRANT_COLLECTION_NAME = "git_repo_index"

qa_prompt_tmpl = PromptTemplate(
    """You are a strict documentation assistant.
You should STRONGLY answer using the information in the provided context.
If the answer is not clearly in the context, say exactly: "I don't know."

Question: {query_str}

Context:
{context_str}

Answer briefly and do NOT write code unless explicitly asked."""
)


Settings.llm = Ollama(
    model="llama3.2",  # name as exposed by Ollama
    base_url="http://localhost:11434",  # default Ollama endpoint
    request_timeout=300.0,  # seconds
)

Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")


_qdrant_client = None


def get_qdrant_client():
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(url=QDRANT_URL, prefer_grpc=True)
    return _qdrant_client


_qdrant_vector_store = None


def get_qdrant_vector_store():
    global _qdrant_vector_store
    if _qdrant_vector_store is None:
        client = get_qdrant_client()
        _qdrant_vector_store = QdrantVectorStore(
            client=client,
            collection_name=QDRANT_COLLECTION_NAME,
        )
    return _qdrant_vector_store


# ---------- MAIN INDEXING LOGIC ----------


def build_index(nodes: list[BaseNode]) -> VectorStoreIndex:
    vector_store = get_qdrant_vector_store()
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store,
    )
    index = VectorStoreIndex(
        nodes,
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


def build_or_load_index(repo_path: Path) -> VectorStoreIndex:
    if PERSIST_DIR.exists():
        # assumes Qdrant collection still exists too
        print("[INFO] Loading existing index...")
        return load_existing_index()
    else:
        files = list_files_for_index(repo_path)
        nodes = chunk(files)
        if len(nodes) == 0:
            raise ValueError("Need documents on first run to build index.")
        print("[INFO] Building index for the first time...")
        return build_index(nodes)


def teardown():
    # TODO
    ...


def interactive_query(index: VectorStoreIndex) -> None:
    print("\n[INFO] Enter interactive query mode. Type 'exit' to quit.\n")
    query_engine = index.as_query_engine(
        similarity_top_k=10,
        llm=Settings.llm,
        text_qa_template=qa_prompt_tmpl,
        streaming=True,
    )

    while True:
        try:
            q = input("Query> ").strip()
            if not q or q.lower() in {"exit", "quit"}:
                break
            resp = query_engine.query(q)

            if args.debug:
                for i, n in enumerate(resp.source_nodes, 1):
                    format_context_node(n, i)

            print("\n----- ANSWER -----")
            for token in resp.response_gen:
                print(token, end="", flush=True)
            print("\n------------------\n")
        except (EOFError, KeyboardInterrupt):
            teardown()
            exit(0)


if __name__ == "__main__":
    args = parse_args()

    repo_path = args.path.resolve()
    if not repo_path.exists():
        raise SystemExit(f"Repo path does not exist: {repo_path}")

    if args.reindex:
        # Wipe state
        if PERSIST_DIR.exists():
            shutil.rmtree(PERSIST_DIR)

        q_client = get_qdrant_client()
        q_client.delete_collection(QDRANT_COLLECTION_NAME)

    if not is_git_repo(repo_path):
        logger.error("Provided path contain no git repository")
        exit(1)

    ensure_repo_clean_or_warn(repo_path)
    index = build_or_load_index(repo_path)
    interactive_query(index)
