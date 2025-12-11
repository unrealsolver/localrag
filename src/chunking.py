from llama_index.core import Document
from llama_index.core.node_parser import CodeSplitter, SentenceSplitter
from llama_index.core.schema import BaseNode
from repo_extract import IndexableFile


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
        else:
            print("text")
            sp = SentenceSplitter(chunk_overlap=20)
            new_nodes = sp.get_nodes_from_documents([doc])
        print("added", len(new_nodes))
        nodes += new_nodes
    return nodes
