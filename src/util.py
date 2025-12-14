from typing import cast
from llama_index.core.schema import NodeWithScore
from colored import Fore, Style


N_LINES = 3
"number of lines of a chunk that would be printed (both ends of chunk)"


def format_line_no(line_num: int, idx: int, offset: int):
    return f"{Fore.cyan}{str(line_num + idx).rjust(offset)}|{Style.reset}"


def format_context_node(node: NodeWithScore, index: int):
    print(
        f"\n--- {Fore.yellow}Source{Style.reset} {index} | {Fore.yellow}score{Style.reset}={node.score} ---"
    )
    print("meta =", node.metadata)
    lines = node.text.split("\n")

    if "line_range" in node.metadata:
        line_range = cast(tuple[int, int], node.metadata["line_range"])
    else:
        line_range = (1, len(lines))

    line_num_ofst = len(str(max(line_range)))

    if len(lines) <= N_LINES * 2:
        print(node.text)
    else:
        for idx, line in enumerate(lines[:3]):
            print(f"{format_line_no(line_range[0], idx, line_num_ofst)}{line}")
        padding = " " * (line_num_ofst - 1)
        print(
            f"{Fore.cyan}…{padding}|{Style.reset} ~~~~~~ skipping {len(lines) - N_LINES * 2} lines ~~~~~~"
        )
        for idx, line in enumerate(lines[-3:]):
            print(
                f"{format_line_no(line_range[1] - N_LINES + 1, idx, line_num_ofst)}{line}"
            )
