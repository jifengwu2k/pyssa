"""Print cyclomatic complexity for every IR region in a Python source file.

This tool compiles a Python file with pyssa, walks the resulting nested Region tree,
and prints JSON describing the cyclomatic complexity of each region.

Complexity is computed from the explicit region CFG only:

    M = E - N + 2P

where:
- N is the number of basic blocks in the region
- E is the number of explicit CFG edges between basic blocks
- P is the number of weakly connected components in that CFG

Because pyssa regions already carry explicit branches/jumps and nested child regions,
this stays structural and local to each region.
"""

import argparse
import json
import sys
from typing import Dict, Sequence, Set, Tuple

from pyssa import Branch, Escape, ForIter, Jump, Region, compile_file, new_compiler_state


def block_successors(block) -> Tuple[object, ...]:
    if not block.instructions:
        return ()

    terminator = block.instructions[-1]
    if isinstance(terminator, (Jump, Escape)):
        return (terminator.target,)
    if isinstance(terminator, Branch):
        return (terminator.true_label, terminator.false_label)
    if isinstance(terminator, ForIter):
        return (terminator.body_label, terminator.exit_label)
    return ()


def cyclomatic_complexity(region: Region) -> int:
    labels = [block.label for block in region.basic_blocks]
    label_set = set(labels)
    adjacency: Dict[object, Set[object]] = {label: set() for label in labels}
    edge_count = 0

    for block in region.basic_blocks:
        for succ in block_successors(block):
            edge_count += 1
            if succ in label_set:
                adjacency[block.label].add(succ)
                adjacency[succ].add(block.label)

    component_count = 0
    seen = set()
    for label in labels:
        if label in seen:
            continue
        component_count += 1
        stack = [label]
        seen.add(label)
        while stack:
            current = stack.pop()
            for neighbor in adjacency[current]:
                if neighbor in seen:
                    continue
                seen.add(neighbor)
                stack.append(neighbor)

    block_count = len(labels)
    return edge_count - block_count + 2 * component_count


def region_to_json(region: Region) -> Dict[str, object]:
    return {
        "name": region.name,
        "cyclomatic_complexity": cyclomatic_complexity(region),
        "child_regions": [region_to_json(child) for child in region.child_regions],
    }


def main(argv: Sequence[str] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("python_file")
    args = parser.parse_args(argv)

    module_region = compile_file(new_compiler_state(), args.python_file)
    json.dump(region_to_json(module_region), sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
