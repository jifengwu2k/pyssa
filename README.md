# pyssa

## Dependency notes

This code expects:

- Python 3.6+
- `attrs`
- `cowlist`

## Quick start

```bash
python compile_to_ir.py ir_test_program.py
python compare_exec_with_ir.py ir_test_program.py
python cyclomatic_complexity.py ir_test_program.py
```

The `cyclomatic_complexity.py` script is a small example of a source-analysis tool made possible by pyssa's explicit nested `Region` CFG IR: it compiles Python to IR, walks each region, and reports cyclomatic complexity structurally from the region CFG.

## Case study: cyclomatic complexity three ways

As a small experiment, this repository includes three versions of the same tool:

- `cyclomatic_complexity.py`: compute complexity from pyssa IR regions
- `cyclomatic_complexity_ast.py`: compute complexity directly from Python AST nodes
- `cyclomatic_complexity_bytecode.py`: compute complexity from CPython bytecode CFG recovery

You can run them on the same input:

```bash
python cyclomatic_complexity.py ir_test_program.py
python cyclomatic_complexity_ast.py ir_test_program.py
python cyclomatic_complexity_bytecode.py ir_test_program.py
```

All three emit the same recursive JSON shape:

```json
{
  "name": "<module>",
  "cyclomatic_complexity": 2,
  "child_regions": []
}
```

### Result

For this mini-experiment, the pyssa-based version is the smallest and simplest implementation in terms of LOC.

As a second metric beyond LOC, we can also measure the complexity of the implementations themselves. Since pyssa reports complexity per region, it is useful to summarize the recursive region tree for each implementation with several descriptive statistics rather than just a single total. The table below reports the sum, min, max, average, and population standard deviation of per-region cyclomatic complexity for each implementation file.

| Tool | Basis | LOC | Sum | Min | Max | Avg | Stddev |
|---|---|---:|---:|---:|---:|---:|---:|
| `cyclomatic_complexity.py` | pyssa `Region` CFG | 94 | 19 | 1 | 13 | 3.80 | 4.67 |
| `cyclomatic_complexity_ast.py` | Python AST | 238 | 71 | 1 | 10 | 1.97 | 1.72 |
| `cyclomatic_complexity_bytecode.py` | CPython bytecode | 202 | 59 | 1 | 32 | 5.36 | 8.51 |

Viewed this way, the pyssa-based implementation is still clearly the smallest by LOC, while the bytecode-based implementation shows the highest peak and widest spread of local complexity. The AST-based implementation distributes complexity more evenly, but across many more lines of code and more implementation machinery.

### Why the pyssa version is smaller

With pyssa, the hard parts are already explicit in the IR:

- nested scopes are already nested as `child_regions`
- control flow is already split into basic blocks
- branches, jumps, and loop iteration edges are already explicit
- the tool can compute complexity directly from the region CFG

The AST version has to reconstruct complexity from syntax-level decision points and scope structure. The bytecode version has to recover a CFG from instructions, jump targets, fallthrough, and exception tables.

### Caveat

The three tools are intentionally a comparison, not a proof that all definitions coincide exactly. In practice they can disagree:

- AST complexity depends on which syntax forms count as decision points
- bytecode complexity depends on how CFG recovery treats compiler-generated structure
- pyssa complexity reflects the explicit IR CFG produced by the frontend

That difference is part of the point of the experiment: pyssa gives a direct, structured, implementation-friendly CFG representation for building analysis tools.

## Current AST lowering coverage

### ASDL builtin leaf types

These are not lowered as standalone nodes, but they are supported as payloads wherever the containing node is supported:

| ASDL builtin | Status | Notes |
|---|---|---|
| `identifier` | Supported | names, attributes, aliases, arguments, etc. |
| `int` | Supported | flags, levels, simple markers, conversions, etc. |
| `string` | Supported | import names, type comments when ignored, string constants, etc. |
| `constant` | Supported | lowered through `Constant(...)` and pattern singleton/value payloads when applicable |

### `mod`

| Node | Status | Notes |
|---|---|---|
| `Module` | Supported | main entry path |
| `Interactive` | Not yet | parser mode not wired in |
| `Expression` | Not yet | expression-only compilation mode not exposed |
| `FunctionType` | Not yet | type-comment function syntax not lowered |

### `stmt`

| Node | Status | Notes |
|---|---|---|
| `FunctionDef` | Supported | decorators, defaults, kw-defaults; not all annotation/type-param features |
| `AsyncFunctionDef` | Supported | same caveats as `FunctionDef` |
| `ClassDef` | Supported | bases, keywords, decorators |
| `Return` | Supported | |
| `Delete` | Supported | names, attributes, subscripts, and tuple/list delete targets |
| `Assign` | Supported | includes tuple/list unpacking |
| `TypeAlias` | Not yet | |
| `AugAssign` | Supported | names, attributes, subscripts |
| `AnnAssign` | Partial | value-bearing forms supported; annotation semantics mostly ignored |
| `For` | Supported | includes `else` |
| `AsyncFor` | Supported | includes `else`; still less battle-tested than sync `for` |
| `While` | Supported | includes `else` |
| `If` | Supported | |
| `With` | Supported | |
| `AsyncWith` | Supported | |
| `Match` | Partial | value/singleton/or/as, sequence, mapping, and class patterns; guards supported, but not all pattern forms |
| `Raise` | Supported | explicit exception, bare `raise`, and `raise ... from ...` |
| `Try` | Supported | `except` / `else` / `finally` |
| `TryStar` | Partial | `except*` lowering is implemented, but exception-group semantics are still incomplete |
| `Assert` | Not yet | |
| `Import` | Supported | |
| `ImportFrom` | Supported | includes `import *` |
| `Global` | Supported | via scope analysis |
| `Nonlocal` | Supported | via scope analysis |
| `Expr` | Supported | |
| `Pass` | Supported | no-op |
| `Break` | Supported | |
| `Continue` | Supported | |

### `expr`

| Node | Status | Notes |
|---|---|---|
| `BoolOp` | Supported | `and` / `or` short-circuiting |
| `NamedExpr` | Supported | walrus operator for name targets |
| `BinOp` | Supported | for the operator set currently implemented in the interpreter |
| `UnaryOp` | Supported | |
| `Lambda` | Supported | lowered as a nested expression-bodied function region |
| `IfExp` | Supported | |
| `Dict` | Supported | includes `**` unpacking |
| `Set` | Supported | |
| `ListComp` | Supported | |
| `SetComp` | Supported | |
| `DictComp` | Supported | |
| `GeneratorExp` | Supported | lowered as a synthetic nested generator region |
| `Await` | Supported | |
| `Yield` | Supported | |
| `YieldFrom` | Supported | |
| `Compare` | Supported | includes chained comparisons |
| `Call` | Supported | includes `*args` and `**kwargs` call forms |
| `FormattedValue` | Supported | f-string field formatting, including `!s` / `!r` / `!a` and format specs |
| `Interpolation` | Not yet | |
| `JoinedStr` | Supported | f-strings / joined strings |
| `TemplateStr` | Not yet | |
| `Constant` | Supported | |
| `Attribute` | Supported | load/store |
| `Subscript` | Supported | load/store |
| `Starred` | Supported | assignment unpacking plus unpacking in calls and sequence builders |
| `Name` | Supported | |
| `List` | Supported | |
| `Tuple` | Supported | |
| `Slice` | Supported | |

### Supporting AST records / enums

| Node | Status | Notes |
|---|---|---|
| `expr_context = Load/Store/Del` | Partial | handled for supported name/attr/subscript/list/tuple uses |
| `boolop = And/Or` | Supported | |
| `operator` | Partial | many arithmetic/bitwise ops supported; only those implemented in the interpreter execute |
| `unaryop` | Supported | `+`, `-`, `not`, `~` |
| `cmpop` | Supported | includes chained comparison lowering |
| `comprehension` | Supported | sync and async comprehensions, including generator expressions |
| `excepthandler = ExceptHandler` | Supported | normal `except` lowering |
| `arguments` | Supported | positional, posonly, kwonly, defaults, `*args`, and `**kwargs` |
| `arg` | Partial | binding supported; annotations/type comments mostly ignored |
| `keyword` | Supported | named keywords plus `**kwargs` unpacking |
| `alias` | Supported | imports |
| `withitem` | Supported | sync and async |
| `match_case` | Partial | supported for the currently lowered `match` subset |
| `pattern` variants | Partial | value/singleton/or/as, sequence, mapping, and class patterns are lowered; others are not yet |
| `type_ignore` | Ignored | parser carries them, lowering does not use them |
| `type_param` variants | Not yet | |

Still intentionally incomplete:

- full `match` / pattern matching coverage
- some corners of named-expression support beyond name targets
- some corners of class construction such as `class C(**kwargs)` forms
- the full Python surface area
