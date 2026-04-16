# AGENTS.md

### Core IR shape

- The top level is a `Region`, not a separate module container.
- Regions are **nested** through `child_regions`.
- Region references are local and structural; do not reintroduce global region registries during lowering.
- Each region owns an explicit CFG:
  - labeled basic blocks
  - explicit branches/jumps
  - explicit child regions

### Lowering philosophy

- Lower from Python AST.
- Do not preserve bytecode quirks just for fidelity.
- Prefer explicit lowering functions with explicit parameters over large class-based implicit mutable state.
- When adding compiler state, keep it minimal and obvious.

### Simplicity over cleverness

- Prefer fewer IR instructions.
- Remove specialized instructions when ordinary instructions already express the behavior.
- Prefer ordinary object operations over interpreter intrinsics where possible.
- Prefer Python truthiness directly in `Branch` and similar control flow.
- Avoid effect-only instructions unless they are clearly necessary for executable semantics.

### Data model conventions

- Use `attrs`, not `dataclasses`.
- Use type annotations.
- Use `COWList` for IR sequences and path-like data.
- Prefer structured data over stringly-typed encodings.
- Do not add metadata unless it is actively useful.

### Runtime/interpreter principles

- The interpreter is semantic, not optimized.
- It should execute the IR directly and simply.
- Do not add runtime-only escape hatches unless there is a strong reason.
- If a runtime helper can be expressed as normal IR, prefer the normal IR.
