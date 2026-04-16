"""Print cyclomatic complexity per nested Python code object using bytecode.

This is a bytecode-based comparison tool next to the IR-based
`cyclomatic_complexity.py` tool. It emits a recursive JSON tree with the shape:

    {
      "name": str,
      "cyclomatic_complexity": int,
      "child_regions": [...]
    }
"""

import argparse
import dis
import json
import sys
import types
from typing import Dict, List, Optional, Sequence, Set, Tuple


UNCONDITIONAL_JUMPS = {
    "JUMP",
    "JUMP_BACKWARD",
    "JUMP_BACKWARD_NO_INTERRUPT",
    "JUMP_FORWARD",
    "JUMP_NO_INTERRUPT",
}

CONDITIONAL_JUMPS = {
    "FOR_ITER",
    "POP_JUMP_IF_FALSE",
    "POP_JUMP_IF_NONE",
    "POP_JUMP_IF_NOT_NONE",
    "POP_JUMP_IF_TRUE",
    "SEND",
}

TERMINATORS = {
    "RAISE_VARARGS",
    "RERAISE",
    "RETURN_CONST",
    "RETURN_VALUE",
}


class CodeNode:
    def __init__(self, name: str, cyclomatic_complexity: int, child_regions: List["CodeNode"]) -> None:
        self.name = name
        self.cyclomatic_complexity = cyclomatic_complexity
        self.child_regions = child_regions

    def to_json(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "cyclomatic_complexity": self.cyclomatic_complexity,
            "child_regions": [child.to_json() for child in self.child_regions],
        }


def instruction_successors(instructions: Sequence[dis.Instruction], index: int) -> List[int]:
    instr = instructions[index]
    next_offset = instructions[index + 1].offset if index + 1 < len(instructions) else None

    if instr.opname in UNCONDITIONAL_JUMPS:
        return [instr.argval]

    if instr.opname in CONDITIONAL_JUMPS:
        successors = []
        if next_offset is not None:
            successors.append(next_offset)
        successors.append(instr.argval)
        return successors

    if instr.opname in TERMINATORS:
        return []

    if next_offset is None:
        return []
    return [next_offset]


def block_ranges(instructions: Sequence[dis.Instruction], leaders: Set[int]) -> Tuple[List[List[dis.Instruction]], Dict[int, int]]:
    blocks: List[List[dis.Instruction]] = []
    block_by_offset: Dict[int, int] = {}
    current: List[dis.Instruction] = []

    for instr in instructions:
        if current and instr.offset in leaders:
            blocks.append(current)
            current = []
        current.append(instr)
        block_by_offset[instr.offset] = len(blocks)

    if current:
        blocks.append(current)

    return blocks, block_by_offset


def code_complexity(code: types.CodeType) -> int:
    instructions = list(dis.get_instructions(code))
    if not instructions:
        return 1

    leaders = {instructions[0].offset}
    for index, instr in enumerate(instructions):
        successors = instruction_successors(instructions, index)
        for target in successors:
            leaders.add(target)

        next_offset = instructions[index + 1].offset if index + 1 < len(instructions) else None
        if next_offset is not None and (len(successors) != 1 or successors[0] != next_offset):
            leaders.add(next_offset)

    exception_entries = list(dis.Bytecode(code).exception_entries)
    instruction_offsets = {instr.offset for instr in instructions}
    for entry in exception_entries:
        if entry.start in instruction_offsets:
            leaders.add(entry.start)
        if entry.end in instruction_offsets:
            leaders.add(entry.end)
        if entry.target in instruction_offsets:
            leaders.add(entry.target)

    blocks, block_by_offset = block_ranges(instructions, leaders)
    if not blocks:
        return 1

    edges: Set[Tuple[int, int]] = set()
    for block_index, block in enumerate(blocks):
        last_index = next(i for i, instr in enumerate(instructions) if instr.offset == block[-1].offset)
        for target in instruction_successors(instructions, last_index):
            if target in block_by_offset:
                edges.add((block_index, block_by_offset[target]))

    for entry in exception_entries:
        if entry.target not in block_by_offset:
            continue
        target_block = block_by_offset[entry.target]
        for block_index, block in enumerate(blocks):
            covered = False
            for instr in block:
                if entry.start <= instr.offset < entry.end:
                    covered = True
                    break
            if covered:
                edges.add((block_index, target_block))

    adjacency: Dict[int, Set[int]] = {index: set() for index in range(len(blocks))}
    for src, dst in edges:
        adjacency[src].add(dst)
        adjacency[dst].add(src)

    components = 0
    seen = set()
    for block_index in range(len(blocks)):
        if block_index in seen:
            continue
        components += 1
        stack = [block_index]
        seen.add(block_index)
        while stack:
            current = stack.pop()
            for neighbor in adjacency[current]:
                if neighbor in seen:
                    continue
                seen.add(neighbor)
                stack.append(neighbor)

    return len(edges) - len(blocks) + 2 * components


def child_code_objects(code: types.CodeType) -> List[types.CodeType]:
    return [const for const in code.co_consts if isinstance(const, types.CodeType)]



def build_tree(code: types.CodeType) -> CodeNode:
    return CodeNode(
        name=code.co_name,
        cyclomatic_complexity=code_complexity(code),
        child_regions=[build_tree(child) for child in child_code_objects(code)],
    )



def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("python_file")
    args = parser.parse_args(argv)

    with open(args.python_file, "r") as f:
        source = f.read()

    code = compile(source, args.python_file, "exec")
    json.dump(build_tree(code).to_json(), sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
