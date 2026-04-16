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
```

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
