import logging
from llama_index.core import Document
from llama_index.core.node_parser import CodeSplitter, SentenceSplitter
from llama_index.core.schema import BaseNode, MetadataMode
from repo_extract import IndexableFile


logger = logging.getLogger(__name__)


def assign_line_ranges(doc: Document, nodes: list[BaseNode]):
    text = doc.text
    cursor = 0  # important for overlapping chunks / duplicate substrings

    for n in nodes:
        chunk = n.get_content(metadata_mode=MetadataMode.NONE)
        start = text.find(chunk, cursor)
        if start < 0:
            # fallback: try from beginning (can be wrong if chunk repeats)
            start = text.find(chunk)
        if start >= 0:
            end = start + len(chunk)

            start = text.count("\n", 0, start) + 1
            end = text.count("\n", 0, end) + 1

            n.metadata["line_range"] = [start, end]

            cursor = (
                start + 1
            )  # mirrors LlamaIndex’s own offset tracking :contentReference[oaicite:1]{index=1}
        else:
            logger.error("Chunk could not be found in the document")

    return nodes


def chunk(files: list[IndexableFile]) -> list[BaseNode]:
    nodes = []
    for idx, file in enumerate(files):
        print(file.rel, ":", idx, "/", len(files))

        doc = Document(
            text=file.read(),
            extra_info={"path": file.rel, "lang": file.lang or "generic"},
        )

        if (lang := file.lang) is not None:
            print("code", lang)
            cs = CodeSplitter(
                language=lang, chunk_lines=80, chunk_lines_overlap=20, max_chars=1500
            )
            new_nodes = cs.get_nodes_from_documents([doc])
            for n in new_nodes:
                n.metadata
        else:
            print("text")
            sp = SentenceSplitter(chunk_overlap=20)
            new_nodes = sp.get_nodes_from_documents([doc])
        assign_line_ranges(doc, new_nodes)
        print("added", len(new_nodes))
        nodes += new_nodes
    return nodes
