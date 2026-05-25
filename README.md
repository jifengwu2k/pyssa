# pyssa

A Python IR with pluggable execution semantics.

## Motivation

`pyssa` is motivated by two use cases:

1. a Python-version-agnostic bytecode-like IR for analysis
2. custom interpreters that mostly behave like Python, but intentionally diverge in specific places

It lowers Python AST into an explicit control-flow IR with nested regions, basic blocks, and ordinary operations. The goal is a stable, executable semantic IR that is easier to analyze, easier to transform, and easier to reinterpret with custom execution rules.

## Installation

```bash
pip install -r requirements.txt
```

Dependencies:
- Python 3.6+
- `attrs`
- `cowlist`
- `typing_extensions`

## Quickstart

Subclass `BaseFrame` and override the `dispatch_*` methods you need.
Unhandled instructions will raise `NotImplementedError`.
`BaseFrame` provides block navigation, name resolution, and a dispatch loop.

**Name resolution methods:**

- `load_name(self, scope: Scope, name: str) -> Any`
- `store_name(self, scope: Scope, name: str, value: Any) -> None`
- `delete_name(self, scope: Scope, name: str) -> None`
- `has_name(self, name: str) -> Optional[Scope]`
- `load_builtin(self, name: str) -> Any`

```python
from pyssa.compiler import compile_source, new_compiler_state
from pyssa.ir import print_region_ir
from pyssa.interpreter import BaseFrame, make_frame, NextInstructionEvent, ReturnEvent


# 1 ── compile Python source into a pyssa IR region ────────────────────────

source = "result = x + y"
module_ir = compile_source(new_compiler_state(), source, path="<tiny-add>")

# Inspect the bytecode-like IR before execution.
print_region_ir(module_ir)
#   region <module> entry=L0
#     block L0:
#       t0 = LoadName(scope=<Scope.NAME: 'name'>, name='x')
#       t1 = LoadName(scope=<Scope.NAME: 'name'>, name='y')
#       t2 = BinaryOp(op='+', lhs=t0, rhs=t1)
#       StoreName(src=t2, scope=<Scope.NAME: 'name'>, name='result')
#       t3 = Const(value=None)
#       Return(value=t3)


# 2 ── define a custom interpreter frame ───────────────────────────────────

class TinyAddOnlyFrame(BaseFrame):
    """A custom interpreter that only understands addition."""

    def __init__(self, region_ir, globals, locals, cells,
                 block_label, instr_index, finished, return_value):
        super().__init__(region_ir, globals, locals, cells,
                         block_label=block_label, instr_index=instr_index,
                         finished=finished, return_value=return_value)
        self.temps = {}    # map temporaries to values

    def dispatch_const(self, instr):
        self.temps[instr.dst] = instr.value
        return NextInstructionEvent()

    def dispatch_load_name(self, instr):
        self.temps[instr.dst] = self.load_name(instr.scope, instr.name)
        return NextInstructionEvent()

    def dispatch_store_name(self, instr):
        self.store_name(instr.scope, instr.name, self.temps[instr.src])
        return NextInstructionEvent()

    def dispatch_binary_op(self, instr):
        if instr.op != "+":
            raise NotImplementedError(
                "TinyAddOnlyFrame only supports addition"
            )
        self.temps[instr.dst] = \
            self.temps[instr.lhs] + self.temps[instr.rhs]
        return NextInstructionEvent()

    def dispatch_return(self, instr):
        return ReturnEvent(self.temps[instr.value])


# 3 ── create a frame from the IR region and step through instructions ─────

add_frame = make_frame(
    TinyAddOnlyFrame,
    module_ir,
    globals={"x": 2, "y": 3},
)

while not add_frame.finished:
    instr = add_frame.get_current_instruction()
    print(f"  [{type(instr).__name__}]")

    event = add_frame.dispatch_current_instruction()
    if isinstance(event, ReturnEvent):
        break   # frame is done

print(add_frame.locals["result"])     # 5
```

## IR overview

The IR is defined in terms of a small set of core classes:

- **`Region`** — a named executable unit with its own CFG: labeled `BasicBlock`s, nested `child_regions`, and metadata (`locals`, `cells`, `freevars`, argument descriptors, exception handlers).
- **`BasicBlock`** — a straight-line sequence of `Instruction`s with a `BasicBlockLabel`.
- **`Instruction`** — two families: `ValueInstruction` (produces a result in `dst: TemporaryValue`) and `EffectInstruction` (side effects only).
- **Operands** — `TemporaryValue` (SSA-style temporaries), `BasicBlockLabel`, `RegionLabel`, `SyntheticLocal` (compiler-generated locals), and `UnpackedTemporaryValue` (splat markers).

## Case study: cyclomatic complexity three ways

Three versions of the same tool are included:

- `cyclomatic_complexity.py`: compute complexity from pyssa IR regions
- `cyclomatic_complexity_ast.py`: compute complexity directly from Python AST nodes
- `cyclomatic_complexity_bytecode.py`: compute complexity from CPython bytecode CFG recovery

Run them on the same input:

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

| Tool | Basis | LOC | Sum | Min | Max | Avg | Stddev |
|---|---|---:|---:|---:|---:|---:|---:|
| `cyclomatic_complexity.py` | pyssa `Region` CFG | 94 | 19 | 1 | 13 | 3.80 | 4.67 |
| `cyclomatic_complexity_ast.py` | Python AST | 238 | 71 | 1 | 10 | 1.97 | 1.72 |
| `cyclomatic_complexity_bytecode.py` | CPython bytecode | 202 | 59 | 1 | 32 | 5.36 | 8.51 |

The pyssa-based implementation is the smallest by LOC. With pyssa, the hard parts are already explicit in the IR:

- nested scopes are already nested as `child_regions`
- control flow is already split into basic blocks
- branches, jumps, and loop iteration edges are already explicit

## AST lowering coverage

### ASDL builtin leaf types

| ASDL builtin | Status | Notes |
|---|---|---|
| `identifier` | Supported | names, attributes, aliases, arguments |
| `int` | Supported | flags, levels, markers, conversions |
| `string` | Supported | import names, string constants |
| `constant` | Supported | lowered through `Constant(...)` and pattern payloads |

### `mod`

| Node | Status |
|---|---|
| `Module` | Supported |
| `Interactive` | Not yet |
| `Expression` | Not yet |
| `FunctionType` | Not yet |

### `stmt`

| Node | Status | Notes |
|---|---|---|
| `FunctionDef` | Supported | decorators, defaults, kw-defaults |
| `AsyncFunctionDef` | Supported | same caveats as `FunctionDef` |
| `ClassDef` | Supported | bases, keywords, decorators |
| `Return` | Supported | |
| `Delete` | Supported | names, attributes, subscripts, tuple/list targets |
| `Assign` | Supported | includes tuple/list unpacking |
| `TypeAlias` | Not yet | |
| `AugAssign` | Supported | names, attributes, subscripts |
| `AnnAssign` | Partial | value-bearing forms supported; annotations mostly ignored |
| `For` | Supported | includes `else` |
| `AsyncFor` | Supported | includes `else` |
| `While` | Supported | includes `else` |
| `If` | Supported | |
| `With` | Supported | |
| `AsyncWith` | Supported | |
| `Match` | Partial | value/singleton/or/as, sequence, mapping, class patterns; guards |
| `Raise` | Supported | explicit, bare `raise`, `raise ... from ...` |
| `Try` | Supported | `except` / `else` / `finally` |
| `TryStar` | Partial | `except*` lowering implemented; exception-group semantics incomplete |
| `Assert` | Supported | lowers to conditional `AssertionError` raise |
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
| `BinOp` | Supported | operator set implemented in interpreter |
| `UnaryOp` | Supported | |
| `Lambda` | Supported | lowered as nested expression-bodied function region |
| `IfExp` | Supported | |
| `Dict` | Supported | includes `**` unpacking |
| `Set` | Supported | |
| `ListComp` | Supported | |
| `SetComp` | Supported | |
| `DictComp` | Supported | |
| `GeneratorExp` | Supported | lowered as synthetic nested generator region |
| `Await` | Supported | |
| `Yield` | Supported | |
| `YieldFrom` | Supported | |
| `Compare` | Supported | includes chained comparisons |
| `Call` | Supported | includes `*args` and `**kwargs` |
| `FormattedValue` | Supported | f-string formatting, `!s` / `!r` / `!a` and format specs |
| `Interpolation` | Not yet | |
| `JoinedStr` | Supported | f-strings / joined strings |
| `TemplateStr` | Not yet | |
| `Constant` | Supported | |
| `Attribute` | Supported | load/store |
| `Subscript` | Supported | load/store |
| `Starred` | Supported | assignment unpacking and call/sequence splatting |
| `Name` | Supported | |
| `List` | Supported | |
| `Tuple` | Supported | |
| `Slice` | Supported | |

### Supporting AST records

| Node | Status | Notes |
|---|---|---|
| `expr_context = Load/Store/Del` | Partial | handled for supported name/attr/subscript uses |
| `boolop = And/Or` | Supported | |
| `operator` | Partial | arithmetic/bitwise ops; only interpreter-implemented ones execute |
| `unaryop` | Supported | `+`, `-`, `not`, `~` |
| `cmpop` | Supported | includes chained comparison lowering |
| `comprehension` | Supported | sync and async comprehensions |
| `excepthandler = ExceptHandler` | Supported | normal `except` lowering |
| `arguments` | Supported | positional, posonly, kwonly, defaults, `*args`, `**kwargs` |
| `arg` | Partial | binding supported; annotations/type comments mostly ignored |
| `keyword` | Supported | named keywords plus `**kwargs` unpacking |
| `alias` | Supported | imports |
| `withitem` | Supported | sync and async |
| `match_case` | Partial | supported for currently lowered `match` subset |
| `pattern` variants | Partial | value/singleton/or/as, sequence, mapping, class patterns |
| `type_ignore` | Ignored | |
| `type_param` variants | Not yet | |

