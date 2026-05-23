# pyssa

A Python AST-to-IR compiler and interpreter with pluggable execution semantics.

## Motivation

`pyssa` is motivated by two use cases:

1. a Python-version-agnostic bytecode-like IR for analysis
2. custom interpreters that mostly behave like Python, but intentionally diverge in specific places

The project lowers Python AST into an explicit control-flow IR with nested regions, basic blocks, and ordinary operations. The goal is not fidelity to CPython bytecode quirks. The goal is a stable, executable semantic IR that is easier to analyze, easier to transform, and easier to reinterpret with custom execution rules.

## Installation

```bash
pip install -r requirements.txt
```

Dependencies:
- Python 3.6+
- `attrs`
- `cowlist`
- `typing_extensions`

## Quickstart: compile, run, and step through instructions

The quickest end-to-end workflow is: compile source to IR, inspect the IR, run the module, then materialize a frame and step it instruction by instruction.

To run a module, wrap its IR in a `Function`, create a frame with `make_frame`, and dispatch instructions in a loop. For instruction-level stepping, use `get_current_instruction()` and `dispatch_current_instruction()` directly.

```python
import sys

from pyssa.compiler import compile_source, new_compiler_state
from pyssa.ir import print_region_ir
from pyssa.interpreter import Function, Interpreter

source = '''
def make_adder(n=0):
    def adder(x, y=1):
        return x + y + n
    return adder

f = make_adder(10)
result = f(5)
'''

# 1. Compile Python source to IR
module_ir = compile_source(new_compiler_state(), source, path='<example>')

# 2. Inspect the IR
print_region_ir(module_ir)

# 3. Run the module normally
module_fn = Function(region_ir=module_ir, globals_dict={})
interp = Interpreter(search_path=list(sys.path))
module_frame = interp.make_frame(module_fn, (), {})
while not module_frame.finished:
    module_event = module_frame.dispatch_current_instruction()
    if module_event is not None:
        break
print(module_frame.locals["result"])   # 16

# 4. Step through one call instruction by instruction
f = module_frame.locals["f"]
print(f.__defaults__)         # (1,)

call_frame = interp.make_frame(f, (100,), {})
while not call_frame.finished:
    instr = call_frame.get_current_instruction()
    print(f"L{call_frame.block_label.index}:{call_frame.instr_index}  {type(instr).__name__}")

    call_event = call_frame.dispatch_current_instruction()
    if call_event is not None:
        print(f"returned: {call_event.value}")  # 111
        break
```


## Building Custom Interpreters

Choose the frame type when creating a frame:

- `interp.make_frame(..., frame_class=MyFrame)`

A simple way to customize execution is to subclass the concrete `Frame`, intercept each instruction, and then delegate to the normal implementation. You can still execute the resulting frame step by step:

```python
import sys

from pyssa.compiler import compile_source, new_compiler_state
from pyssa.interpreter import Frame, Function, Interpreter

class TracingFrame(Frame):
    def dispatch_current_instruction(self):
        instr = self.get_current_instruction()
        if instr is not None:
            print(f"before  L{self.block_label.index}:{self.instr_index}  {type(instr).__name__}")
        event = super().dispatch_current_instruction()
        if event is not None:
            print(f"event   {type(event).__name__}({event.value!r})")
        return event

source = '''
def square(x):
    return x * x
'''

module_ir = compile_source(new_compiler_state(), source, path='<trace>')
module_fn = Function(region_ir=module_ir, globals_dict={})
interp = Interpreter(search_path=list(sys.path))
trace_frame = interp.make_frame(module_fn, (), {}, frame_class=TracingFrame)
while not trace_frame.finished:
    trace_event = trace_frame.dispatch_current_instruction()
    if trace_event is not None:
        break

# For manual stepping, request the same frame class explicitly.
step_frame = interp.make_frame(trace_frame.locals["square"], (7,), {}, frame_class=TracingFrame)
while not step_frame.finished:
    step_event = step_frame.dispatch_current_instruction()
    if step_event is not None:
        print(step_event.value)  # 49
        break
```

If you want to define instruction semantics from scratch, subclass `BaseFrame`. It provides:

- **Fields**: `interpreter`, `function`, `locals`, `globals`, `block_label`, `instr_index`, `cells`, `try_stack`, `finished`, `return_value`, `current_exception`
- **Name resolution**: `load_name`, `store_name`, `delete_name`, `has_name`, `load_builtin`
- **Exception handling**: `handle_exception`
- **Block navigation**: `get_block`, `fallthrough_label`, `get_current_instruction`
- **Dispatch loop**: `dispatch_current_instruction`
- **Instruction stubs**: all `dispatch_*` methods raise `NotImplementedError`

Supply additional state in your `__init__` and implement only the handlers you need. Using `make_frame` directly with a `Function` wrapping the module IR allows symbolically executing only portions of modules:

```python
from pyssa.compiler import compile_source, new_compiler_state
from pyssa.interpreter import BaseFrame, Function, Interpreter, ReturnEvent

class TinyAddOnlyFrame(BaseFrame):
    """Enough for a tiny add-only subset."""

    def __init__(self, interpreter, function, globals, locals, cells, block_label=None, instr_index=0):
        super().__init__(interpreter, function, locals, globals, block_label, instr_index, cells)
        self.temps = {}

    def dispatch_const(self, instr):
        self.temps[instr.dst] = instr.value

    def dispatch_load_name(self, instr):
        self.temps[instr.dst] = self.load_name(instr.scope, instr.name)

    def dispatch_store_name(self, instr):
        self.store_name(instr.scope, instr.name, self.temps[instr.src])

    def dispatch_binary_op(self, instr):
        if instr.op != "+":
            raise NotImplementedError("TinyAddOnlyFrame only supports addition")
        self.temps[instr.dst] = self.temps[instr.lhs] + self.temps[instr.rhs]

    def dispatch_return(self, instr):
        return ReturnEvent(self.temps[instr.value])

source = "result = x + y"
module_ir = compile_source(new_compiler_state(), source, path="<tiny-add>")
module_function = Function(region_ir=module_ir, globals_dict={"x": 2, "y": 3})
interp = Interpreter()
add_frame = interp.make_frame(module_function, (), {}, frame_class=TinyAddOnlyFrame)
while not add_frame.finished:
    add_event = add_frame.dispatch_current_instruction()
    if add_event is not None:
        break
print(add_frame.locals["result"])  # 5
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

