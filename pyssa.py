# Copyright (c) 2026 Jifeng Wu
# Licensed under the Apache-2.0 License. See LICENSE file in the project root for full license information.
import ast
import builtins
import collections.abc
import importlib
import importlib.machinery
import importlib.util
import inspect
import operator
import os
import symtable
import sys
import types
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import attrs
from cowlist import COWList

# ---------------------------------------------------------------------------
# Core IR data model
# ---------------------------------------------------------------------------


class Scope(str, Enum):
    # Variable addressing modes used by the frontend and interpreter.
    LOCAL = "local"
    GLOBAL = "global"
    NAME = "name"
    CELL = "cell"


# Operands.
@attrs.define(frozen=True)
class Operand:
    pass


@attrs.define(frozen=True)
class TemporaryValue(Operand):
    index: int


@attrs.define(frozen=True)
class BasicBlockLabel(Operand):
    index: int


@attrs.define(frozen=True)
class RegionLabel(Operand):
    index: int


@attrs.define(frozen=True)
class SyntheticLocal(Operand):
    index: int
    purpose: str = ""


@attrs.define(frozen=True)
class UnpackedTemporaryValue(Operand):
    value: TemporaryValue


# Optional source provenance attached to instructions.
@attrs.define(frozen=True)
class SourceSpan:
    lineno: Optional[int] = None
    end_lineno: Optional[int] = None
    col_offset: Optional[int] = None
    end_col_offset: Optional[int] = None


# Base instruction families.
@attrs.define(frozen=True, kw_only=True)
class Instruction:
    span: Optional[SourceSpan] = None


@attrs.define(frozen=True, kw_only=True)
class EffectInstruction(Instruction):
    pass


@attrs.define(frozen=True, kw_only=True)
class ValueInstruction(Instruction):
    dst: TemporaryValue


# Variable and constant operations.
@attrs.define(frozen=True)
class Const(ValueInstruction):
    value: Any


@attrs.define(frozen=True)
class LoadName(ValueInstruction):
    scope: Scope = Scope.LOCAL
    name: str = ""


@attrs.define(frozen=True)
class StoreName(EffectInstruction):
    src: TemporaryValue
    scope: Scope = Scope.LOCAL
    name: str = ""


@attrs.define(frozen=True)
class DeleteName(EffectInstruction):
    scope: Scope = Scope.LOCAL
    name: str = ""


# Pure-ish computation and object access.
@attrs.define(frozen=True)
class UnaryOp(ValueInstruction):
    op: str
    src: TemporaryValue


@attrs.define(frozen=True)
class BinaryOp(ValueInstruction):
    op: str
    lhs: TemporaryValue
    rhs: TemporaryValue


@attrs.define(frozen=True)
class CompareOp(ValueInstruction):
    cmp: str
    lhs: TemporaryValue
    rhs: TemporaryValue


@attrs.define(frozen=True)
class LoadAttr(ValueInstruction):
    obj: TemporaryValue
    attr_name: str


@attrs.define(frozen=True)
class StoreAttr(EffectInstruction):
    obj: TemporaryValue
    attr_name: str
    value: TemporaryValue


@attrs.define(frozen=True)
class DeleteAttr(EffectInstruction):
    obj: TemporaryValue
    attr_name: str


@attrs.define(frozen=True)
class LoadItem(ValueInstruction):
    obj: TemporaryValue
    key: TemporaryValue


@attrs.define(frozen=True)
class StoreItem(EffectInstruction):
    obj: TemporaryValue
    key: TemporaryValue
    value: TemporaryValue


@attrs.define(frozen=True)
class DeleteItem(EffectInstruction):
    obj: TemporaryValue
    key: TemporaryValue


# Aggregate builders and destructuring helpers.
@attrs.define(frozen=True)
class BuildTuple(ValueInstruction):
    items: Sequence[Operand] = attrs.field(factory=COWList)


@attrs.define(frozen=True)
class BuildList(ValueInstruction):
    items: Sequence[Operand] = attrs.field(factory=COWList)


@attrs.define(frozen=True)
class BuildSet(ValueInstruction):
    items: Sequence[Operand] = attrs.field(factory=COWList)


@attrs.define(frozen=True)
class BuildMap(ValueInstruction):
    items: Sequence[Tuple[Optional[TemporaryValue], TemporaryValue]] = attrs.field(factory=COWList)


@attrs.define(frozen=True)
class BuildSlice(ValueInstruction):
    start: TemporaryValue
    stop: TemporaryValue
    step: Optional[TemporaryValue] = None


@attrs.define(frozen=True)
class BuildString(ValueInstruction):
    parts: Sequence[TemporaryValue] = attrs.field(factory=COWList)


@attrs.define(frozen=True)
class FormatValue(ValueInstruction):
    value: TemporaryValue
    conversion: Optional[str] = None
    spec: Optional[TemporaryValue] = None


@attrs.define(frozen=True)
class Unpack(EffectInstruction):
    src: TemporaryValue
    dsts: Sequence[TemporaryValue] = attrs.field(factory=COWList)
    star_index: Optional[int] = None


# Calls, imports, and function/class creation.
@attrs.define(frozen=True)
class Call(ValueInstruction):
    callee: TemporaryValue
    args: Sequence[Operand] = attrs.field(factory=COWList)
    kwargs: Sequence[Tuple[Optional[str], TemporaryValue]] = attrs.field(factory=COWList)
    flags: int = 0


@attrs.define(frozen=True)
class ImportName(ValueInstruction):
    module: Optional[str]
    fromlist: Sequence[str] = attrs.field(factory=COWList)
    level: int = 0


@attrs.define(frozen=True)
class ImportFrom(ValueInstruction):
    module_obj: TemporaryValue
    name: str


@attrs.define(frozen=True)
class ImportStar(EffectInstruction):
    module_obj: TemporaryValue


@attrs.define(frozen=True)
class MakeFunction(ValueInstruction):
    code: RegionLabel
    defaults: Sequence[TemporaryValue] = attrs.field(factory=COWList)
    kwdefaults: Sequence[Tuple[str, TemporaryValue]] = attrs.field(factory=COWList)
    annotations: Sequence[Tuple[str, TemporaryValue]] = attrs.field(factory=COWList)
    closure: Sequence[TemporaryValue] = attrs.field(factory=COWList)
    flags: int = 0


@attrs.define(frozen=True)
class BuildClass(ValueInstruction):
    body_func: TemporaryValue
    name: TemporaryValue
    bases: Sequence[TemporaryValue] = attrs.field(factory=COWList)
    keywords: Sequence[Tuple[str, TemporaryValue]] = attrs.field(factory=COWList)


# Iteration, generators, and async machinery.
@attrs.define(frozen=True)
class GetIter(ValueInstruction):
    iterable: TemporaryValue


@attrs.define(frozen=True)
class ForIter(EffectInstruction):
    iter_obj: TemporaryValue
    value_dst: TemporaryValue
    body_label: BasicBlockLabel
    exit_label: BasicBlockLabel


@attrs.define(frozen=True)
class GetAIter(ValueInstruction):
    iterable: TemporaryValue


@attrs.define(frozen=True)
class GetANext(ValueInstruction):
    aiter: TemporaryValue


@attrs.define(frozen=True)
class GetAwaitable(ValueInstruction):
    value: TemporaryValue
    where: int = 0


@attrs.define(frozen=True)
class YieldValue(ValueInstruction):
    value: TemporaryValue


@attrs.define(frozen=True)
class YieldFrom(ValueInstruction):
    value: TemporaryValue


@attrs.define(frozen=True)
class AwaitValue(ValueInstruction):
    value: TemporaryValue


# Exception and control-flow operations.
@attrs.define(frozen=True)
class CurrentException(ValueInstruction):
    pass


@attrs.define(frozen=True)
class Raise(EffectInstruction):
    exc: TemporaryValue
    cause: Optional[TemporaryValue] = None


@attrs.define(frozen=True)
class Reraise(EffectInstruction):
    pass


@attrs.define(frozen=True)
class CheckExcMatch(ValueInstruction):
    exc: TemporaryValue
    typ: TemporaryValue


@attrs.define(frozen=True)
class CheckEGMatch(ValueInstruction):
    exc: TemporaryValue
    typ: TemporaryValue


@attrs.define(frozen=True)
class PushTry(EffectInstruction):
    except_label: Optional[BasicBlockLabel] = None
    finally_label: Optional[BasicBlockLabel] = None


@attrs.define(frozen=True)
class PopTry(EffectInstruction):
    pass


@attrs.define(frozen=True)
class ClearException(EffectInstruction):
    pass


@attrs.define(frozen=True)
class EndFinally(EffectInstruction):
    pass


@attrs.define(frozen=True)
class Escape(EffectInstruction):
    target: BasicBlockLabel


@attrs.define(frozen=True)
class Jump(EffectInstruction):
    target: BasicBlockLabel


@attrs.define(frozen=True)
class Branch(EffectInstruction):
    cond: TemporaryValue
    true_label: BasicBlockLabel
    false_label: BasicBlockLabel


@attrs.define(frozen=True)
class Return(EffectInstruction):
    value: TemporaryValue


# Pattern-matching helpers.
@attrs.define(frozen=True)
class MatchMapping(ValueInstruction):
    value: TemporaryValue


@attrs.define(frozen=True)
class MatchSequence(ValueInstruction):
    value: TemporaryValue


@attrs.define(frozen=True)
class MatchKeys(ValueInstruction):
    mapping: TemporaryValue
    keys: TemporaryValue


@attrs.define(frozen=True)
class MatchClass(ValueInstruction):
    value: TemporaryValue
    cls: TemporaryValue
    attr_names: Sequence[str] = attrs.field(factory=COWList)
    positional_count: int = 0


# CFG and region containers.
@attrs.define(frozen=True)
class BasicBlock:
    label: BasicBlockLabel
    instructions: Sequence[Instruction] = attrs.field(factory=COWList)


@attrs.define(frozen=True)
class ExceptionHandler:
    start_label: BasicBlockLabel
    end_label: BasicBlockLabel
    target_label: BasicBlockLabel
    stack_depth: Optional[int] = None
    push_lasti: bool = False
    pop_on_entry: int = 0


@attrs.define(frozen=True)
class Region:
    name: str
    entry_label: BasicBlockLabel
    label: Optional[RegionLabel] = None
    is_class: bool = False
    basic_blocks: Sequence[BasicBlock] = attrs.field(factory=COWList)
    child_regions: Sequence["Region"] = attrs.field(factory=COWList)
    locals: Sequence[str] = attrs.field(factory=COWList)
    cells: Sequence[str] = attrs.field(factory=COWList)
    freevars: Sequence[str] = attrs.field(factory=COWList)
    handlers: Sequence[ExceptionHandler] = attrs.field(factory=COWList)
    argcount: int = 0
    posonlyargcount: int = 0
    kwonlyargcount: int = 0
    vararg_name: Optional[str] = None
    kwarg_name: Optional[str] = None
    flags: int = 0

# ---------------------------------------------------------------------------
# IR printer
# ---------------------------------------------------------------------------


VALUE_FIELDS = set(["dst", "span"])
EFFECT_FIELDS = set(["span"])


def format_value(value):
    # Render small leaf values inline in instruction payloads.
    if isinstance(value, TemporaryValue):
        return "t%s" % (value.index,)
    if isinstance(value, BasicBlockLabel):
        return "L%s" % (value.index,)
    if isinstance(value, RegionLabel):
        return "R%s" % (value.index,)
    if isinstance(value, SyntheticLocal):
        suffix = "" if not value.purpose else ":%s" % (value.purpose,)
        return "s%s%s" % (value.index, suffix)
    if isinstance(value, Region):
        return "@%s" % (value.name,)
    if isinstance(value, UnpackedTemporaryValue):
        return "*%s" % (render_payload(value.value),)
    return repr(value)


def render_payload(value):
    # Render nested payload structures such as lists of args or block targets.
    if isinstance(value, TemporaryValue):
        return "t%s" % (value.index,)
    if isinstance(value, BasicBlockLabel):
        return "L%s" % (value.index,)
    if isinstance(value, RegionLabel):
        return "R%s" % (value.index,)
    if isinstance(value, SyntheticLocal):
        suffix = "" if not value.purpose else ":%s" % (value.purpose,)
        return "s%s%s" % (value.index, suffix)
    if isinstance(value, Region):
        return "@%s" % (value.name,)
    if isinstance(value, UnpackedTemporaryValue):
        return "*%s" % (render_payload(value.value),)
    if isinstance(value, COWList) or isinstance(value, (list, tuple)):
        return "[%s]" % ", ".join(render_payload(item) for item in value)
    return repr(value)


def render_instruction(instr, indent="    "):
    # Render one instruction using attrs field order and a small amount of IR-specific formatting.
    import attrs

    fields = attrs.fields(type(instr))
    names = [field.name for field in fields]
    payload_names = [name for name in names if name not in VALUE_FIELDS and name not in EFFECT_FIELDS]

    if hasattr(instr, "dst"):
        pieces = []
        for name in payload_names:
            pieces.append("%s=%s" % (name, render_payload(getattr(instr, name))))
        return "%st%s = %s(%s)" % (indent, instr.dst.index, type(instr).__name__, ", ".join(pieces))

    pieces = []
    for name in payload_names:
        pieces.append("%s=%s" % (name, render_payload(getattr(instr, name))))
    return "%s%s(%s)" % (indent, type(instr).__name__, ", ".join(pieces))


def print_instruction(instr, indent="    "):
    print(render_instruction(instr, indent=indent))


def print_region_ir(region_ir, indent=""):
    # Print one region followed by any nested child regions.
    label_prefix = "" if region_ir.label is None else "R%s " % (region_ir.label.index,)
    print("%sregion %s%s entry=L%s" % (indent, label_prefix, region_ir.name, region_ir.entry_label.index))
    for block in region_ir.basic_blocks:
        print("%s  block L%s:" % (indent, block.label.index))
        for instr in block.instructions:
            print_instruction(instr, indent=indent + "    ")
        if not block.instructions:
            print("%s    <empty>" % indent)
    for child_region in region_ir.child_regions:
        print()
        print_region_ir(child_region, indent=indent + "  ")

# ---------------------------------------------------------------------------
# AST -> IR compiler
# ---------------------------------------------------------------------------


# Error used when the frontend reaches syntax it still does not lower.
class UnsupportedFeature(NotImplementedError):
    def __init__(self, node: ast.AST, message: str) -> None:
        self.node = node
        self.message = message
        lineno = getattr(node, "lineno", None)
        col = getattr(node, "col_offset", None)
        location = ""
        if lineno is not None:
            location = " at line %s" % lineno
            if col is not None:
                location += ":%s" % col
        super().__init__("%s%s" % (message, location))


# Per-region lowering context.
@attrs.define
class RegionContext:
    # Per-region state shared by lowering helpers. The `code_obj` comes from Python's own
    # compiler and is used only for metadata/scope shape, not for bytecode lowering.
    name: str
    name_path: COWList
    is_class: bool
    node: ast.AST
    table: Any
    code_obj: types.CodeType
    builder: "BlockBuilder"
    child_tables: List[Any] = attrs.field(factory=list)
    child_codes: List[types.CodeType] = attrs.field(factory=list)
    next_child_table: int = 0
    next_child_code: int = 0
    next_child_region_label: int = 0


# CFG construction helper used while lowering a single region.
class BlockBuilder(object):
    # Helper for building CFG blocks incrementally. Lowering maintains an explicit basic-block
    # graph keyed by labels and later materializes the final ordered block sequence for the IR.
    def __init__(self) -> None:
        self.basic_blocks: List[BasicBlock] = []
        self.blocks_by_label: Dict[BasicBlockLabel, BasicBlock] = {}
        self.block_successors: Dict[BasicBlockLabel, COWList] = {}
        self.current_label: Optional[BasicBlockLabel] = None
        self.current_instructions: Optional[List[Any]] = None
        self.label_index: int = 0

    def new_label(self) -> BasicBlockLabel:
        label = BasicBlockLabel(self.label_index)
        self.label_index += 1
        return label

    def start(self) -> None:
        self.start_block(self.new_label())

    def start_block(self, label: BasicBlockLabel) -> None:
        if self.current_label is not None:
            self.finish_block()
        self.current_label = label
        self.current_instructions = []

    def is_open(self) -> bool:
        return self.current_label is not None

    def ensure_open(self) -> None:
        if not self.is_open():
            self.start_block(self.new_label())

    def emit(self, instr: Any) -> None:
        if not self.is_open():
            raise RuntimeError("attempted to emit into a closed block")
        self.current_instructions.append(instr)

    def terminate(self, instr: Any) -> None:
        self.emit(instr)
        self.finish_block()

    def finish_block(self) -> None:
        if self.current_label is None:
            return
        block = BasicBlock(label=self.current_label, instructions=COWList(self.current_instructions))
        self.basic_blocks.append(block)
        self.blocks_by_label[self.current_label] = block
        self.block_successors[self.current_label] = COWList(self.successors_for_block(block))
        self.current_label = None
        self.current_instructions = None

    def successors_for_block(self, block: BasicBlock) -> List[BasicBlockLabel]:
        if not block.instructions:
            return []
        instr = block.instructions[-1]
        if isinstance(instr, (Jump, Escape)):
            return [instr.target]
        if isinstance(instr, Branch):
            return [instr.true_label, instr.false_label]
        if isinstance(instr, ForIter):
            return [instr.body_label, instr.exit_label]
        return []

    def finish(self) -> COWList:
        self.finish_block()
        return COWList(self.basic_blocks)




# Kinds of nested executable regions that can appear under a parent region.
class ChildRegionType(str, Enum):
    FUNCTION = 'function'
    CLASS = 'class'


# Cross-region compiler state. This stays intentionally small and explicit.
@attrs.define
class CompilerState:
    temp_index: int = 0
    synthetic_local_index: int = 0
    loop_stack: List[Tuple[BasicBlockLabel, BasicBlockLabel]] = attrs.field(factory=list)
    region_nested_stacks: List[List[Region]] = attrs.field(factory=list)
    synthetic_region_name_stacks: List[Dict[str, int]] = attrs.field(factory=list)


def new_compiler_state() -> CompilerState:
    return CompilerState()


def fresh_temp(state: CompilerState) -> TemporaryValue:
    temp = TemporaryValue(state.temp_index)
    state.temp_index += 1
    return temp


def fresh_child_region_label(ctx: RegionContext) -> RegionLabel:
    label = RegionLabel(ctx.next_child_region_label)
    ctx.next_child_region_label += 1
    return label


def fresh_synthetic_local(state: CompilerState, purpose: str = "") -> SyntheticLocal:
    local = SyntheticLocal(index=state.synthetic_local_index, purpose=purpose)
    state.synthetic_local_index += 1
    return local

def compile_source(state: CompilerState, source: str, path: str = "<ast>") -> Region:
    """Compile one source string into the top-level Region."""
    tree = ast.parse(source, filename=path, mode='exec')
    root_table = symtable.symtable(source, path, 'exec')
    root_code = compile(source, path, 'exec')
    return compile_region_node(state, node=tree, table=root_table, code_obj=root_code, name='<module>', name_path=COWList(['<module>']), is_class=False, label=None)

def compile_file(state: CompilerState, path: str) -> Region:
    with open(path, 'r') as f:
        source = f.read()
    return compile_source(state, source, path=path)

def child_code_objects(state: CompilerState, code_obj: types.CodeType) -> List[types.CodeType]:
    return [const for const in code_obj.co_consts if isinstance(const, types.CodeType)]

def compile_region_node(state: CompilerState, node: ast.AST, table: Any, code_obj: types.CodeType, name: str, name_path: COWList, is_class: bool, label: Optional[RegionLabel]) -> Region:
    """Dispatch to the appropriate region compiler for this AST node."""
    if isinstance(node, ast.GeneratorExp):
        return _compile_genexpr_region(state, node=node, table=table, code_obj=code_obj, name=name, name_path=name_path, is_class=is_class, label=label)
    if isinstance(node, ast.Lambda):
        return _compile_lambda_region(state, node=node, table=table, code_obj=code_obj, name=name, name_path=name_path, is_class=is_class, label=label)
    return _compile_region_ast(state, node=node, table=table, code_obj=code_obj, name=name, name_path=name_path, is_class=is_class, label=label)

def _compile_region_ast(state: CompilerState, node: ast.AST, table: Any, code_obj: types.CodeType, name: str, name_path: COWList, is_class: bool, label: Optional[RegionLabel]) -> Region:
    # Generic region lowering path used for modules, functions, classes, and coroutines.
    nested_regions = []
    builder = BlockBuilder()
    builder.start()
    previous_loop_stack = state.loop_stack
    state.loop_stack = []
    state.region_nested_stacks.append([])
    state.synthetic_region_name_stacks.append({})
    ctx = RegionContext(name=name, name_path=name_path, is_class=is_class, node=node, table=table, code_obj=code_obj, builder=builder, child_tables=list(table.get_children()), child_codes=child_code_objects(state, code_obj))
    body = getattr(node, 'body', ())
    for stmt in body:
        child_regions = lower_stmt(state, ctx, stmt)
        if child_regions:
            nested_regions.extend(child_regions)
    if builder.is_open():
        emit_return_none(state, builder, node)
        builder.finish_block()
    nested_regions.extend(state.region_nested_stacks.pop())
    state.synthetic_region_name_stacks.pop()
    vararg_name, kwarg_name = region_variadic_names(node)
    basic_blocks = builder.finish()
    region = Region(name=name, entry_label=basic_blocks[0].label, label=label, is_class=is_class, basic_blocks=basic_blocks, child_regions=COWList(nested_regions), locals=COWList(code_obj.co_varnames), cells=COWList(code_obj.co_cellvars), freevars=COWList(code_obj.co_freevars), argcount=code_obj.co_argcount, posonlyargcount=getattr(code_obj, 'co_posonlyargcount', 0), kwonlyargcount=code_obj.co_kwonlyargcount, vararg_name=vararg_name, kwarg_name=kwarg_name, flags=code_obj.co_flags)
    state.loop_stack = previous_loop_stack
    return region

def _compile_genexpr_region(state: CompilerState, node: ast.GeneratorExp, table: Any, code_obj: types.CodeType, name: str, name_path: COWList, is_class: bool, label: Optional[RegionLabel]) -> Region:
    # Generator expressions are themselves nested executable regions with yield points.
    builder = BlockBuilder()
    builder.start()
    previous_loop_stack = state.loop_stack
    state.loop_stack = []
    state.region_nested_stacks.append([])
    state.synthetic_region_name_stacks.append({})
    ctx = RegionContext(name=name, name_path=name_path, is_class=is_class, node=node, table=table, code_obj=code_obj, builder=builder, child_tables=list(table.get_children()), child_codes=child_code_objects(state, code_obj))
    after_label = builder.new_label()
    lower_genexpr_generator(state, ctx, node.generators, 0, lambda: emit_genexpr_yield(state, ctx, node.elt), after_label, node)
    builder.start_block(after_label)
    if builder.is_open():
        emit_return_none(state, builder, node)
        builder.finish_block()
    nested_regions = state.region_nested_stacks.pop()
    state.synthetic_region_name_stacks.pop()
    basic_blocks = builder.finish()
    region = Region(name=name, entry_label=basic_blocks[0].label, label=label, is_class=is_class, basic_blocks=basic_blocks, child_regions=COWList(nested_regions), locals=COWList(code_obj.co_varnames), cells=COWList(code_obj.co_cellvars), freevars=COWList(code_obj.co_freevars), argcount=code_obj.co_argcount, posonlyargcount=getattr(code_obj, 'co_posonlyargcount', 0), kwonlyargcount=code_obj.co_kwonlyargcount, flags=code_obj.co_flags)
    state.loop_stack = previous_loop_stack
    return region


def _compile_lambda_region(state: CompilerState, node: ast.Lambda, table: Any, code_obj: types.CodeType, name: str, name_path: COWList, is_class: bool, label: Optional[RegionLabel]) -> Region:
    # Lambdas are expression-bodied nested regions that return their body value directly.
    builder = BlockBuilder()
    builder.start()
    previous_loop_stack = state.loop_stack
    state.loop_stack = []
    state.region_nested_stacks.append([])
    state.synthetic_region_name_stacks.append({})
    ctx = RegionContext(name=name, name_path=name_path, is_class=is_class, node=node, table=table, code_obj=code_obj, builder=builder, child_tables=list(table.get_children()), child_codes=child_code_objects(state, code_obj))
    value = lower_expr(state, ctx, node.body)
    if builder.is_open():
        builder.terminate(attach_meta(state, Return(value=value), node.body))
    nested_regions = state.region_nested_stacks.pop()
    state.synthetic_region_name_stacks.pop()
    vararg_name, kwarg_name = region_variadic_names(node)
    basic_blocks = builder.finish()
    region = Region(name=name, entry_label=basic_blocks[0].label, label=label, is_class=is_class, basic_blocks=basic_blocks, child_regions=COWList(nested_regions), locals=COWList(code_obj.co_varnames), cells=COWList(code_obj.co_cellvars), freevars=COWList(code_obj.co_freevars), argcount=code_obj.co_argcount, posonlyargcount=getattr(code_obj, 'co_posonlyargcount', 0), kwonlyargcount=code_obj.co_kwonlyargcount, vararg_name=vararg_name, kwarg_name=kwarg_name, flags=code_obj.co_flags)
    state.loop_stack = previous_loop_stack
    return region

def emit_genexpr_yield(state, ctx, elt):
    value = lower_expr(state, ctx, elt)
    temp = fresh_temp(state)
    ctx.builder.emit(attach_meta(state, YieldValue(dst=temp, value=value), elt))

def lower_genexpr_generator(state, ctx, generators, index, emit_item, exhaustion_label, owner):
    generator = generators[index]
    iter_name, owns_iter_name = comprehension_iter_name(state, ctx, generator, index)
    header_label = ctx.builder.new_label()
    body_label = ctx.builder.new_label()
    cleanup_label = ctx.builder.new_label()
    if generator.is_async:
        stop_label = ctx.builder.new_label()
        stop_match_label = ctx.builder.new_label()
        stop_nomatch_label = ctx.builder.new_label()
    ctx.builder.terminate(attach_meta(state, Jump(target=header_label), generator))
    ctx.builder.start_block(header_label)
    current_iter = fresh_temp(state)
    ctx.builder.emit(attach_meta(state, LoadName(dst=current_iter, scope=Scope.LOCAL, name=iter_name), generator))
    if generator.is_async:
        next_awaitable = fresh_temp(state)
        ctx.builder.emit(attach_meta(state, GetANext(dst=next_awaitable, aiter=current_iter), generator))
        ctx.builder.emit(attach_meta(state, PushTry(except_label=stop_label), generator))
        value_dst = await_value(state, ctx, next_awaitable, generator)
        ctx.builder.emit(attach_meta(state, PopTry(), generator))
        ctx.builder.terminate(attach_meta(state, Jump(target=body_label), generator))
        ctx.builder.start_block(stop_label)
        current_exc = current_exception_value(state, ctx, generator)
        stop_type = builtin_const_value(state, ctx, builtins.StopAsyncIteration, generator)
        matched = fresh_temp(state)
        ctx.builder.emit(attach_meta(state, CheckExcMatch(dst=matched, exc=current_exc, typ=stop_type), generator))
        ctx.builder.terminate(attach_meta(state, Branch(cond=matched, true_label=stop_match_label, false_label=stop_nomatch_label), generator))
        ctx.builder.start_block(stop_match_label)
        ctx.builder.emit(attach_meta(state, ClearException(), generator))
        ctx.builder.terminate(attach_meta(state, Jump(target=cleanup_label), generator))
        ctx.builder.start_block(stop_nomatch_label)
        ctx.builder.terminate(attach_meta(state, Reraise(), generator))
        ctx.builder.start_block(body_label)
    else:
        value_dst = fresh_temp(state)
        ctx.builder.terminate(attach_meta(state, ForIter(iter_obj=current_iter, value_dst=value_dst, body_label=body_label, exit_label=cleanup_label), generator))
        ctx.builder.start_block(body_label)
    assign_target(state, ctx, generator.target, value_dst)
    for if_expr in generator.ifs:
        next_label = ctx.builder.new_label()
        cond = lower_expr(state, ctx, if_expr)
        ctx.builder.terminate(attach_meta(state, Branch(cond=cond, true_label=next_label, false_label=header_label), if_expr))
        ctx.builder.start_block(next_label)
    if index + 1 < len(generators):
        lower_genexpr_generator(state, ctx, generators, index + 1, emit_item, header_label, owner)
    else:
        emit_item()
        if ctx.builder.is_open():
            ctx.builder.terminate(attach_meta(state, Jump(target=header_label), owner))
    ctx.builder.start_block(cleanup_label)
    if owns_iter_name:
        ctx.builder.emit(attach_meta(state, DeleteName(scope=Scope.LOCAL, name=iter_name), generator))
    ctx.builder.terminate(attach_meta(state, Jump(target=exhaustion_label), generator))

def lower_stmt(state: CompilerState, ctx: RegionContext, stmt: ast.stmt) -> List[Region]:
    """Lower one statement, returning any nested regions created along the way."""
    if isinstance(stmt, ast.FunctionDef):
        return lower_function_def(state, ctx, stmt, is_async=False)
    if isinstance(stmt, ast.AsyncFunctionDef):
        return lower_function_def(state, ctx, stmt, is_async=True)
    if isinstance(stmt, ast.ClassDef):
        return lower_class_def(state, ctx, stmt)
    if isinstance(stmt, ast.Assign):
        value = lower_expr(state, ctx, stmt.value)
        for target in stmt.targets:
            assign_target(state, ctx, target, value)
        return []
    if isinstance(stmt, ast.AnnAssign):
        if stmt.value is None:
            return []
        value = lower_expr(state, ctx, stmt.value)
        assign_target(state, ctx, stmt.target, value)
        return []
    if isinstance(stmt, ast.AugAssign):
        lower_augassign(state, ctx, stmt)
        return []
    if isinstance(stmt, ast.Return):
        value = lower_optional_expr(state, ctx, stmt.value, stmt)
        ctx.builder.terminate(attach_meta(state, Return(value=value), stmt))
        return []
    if isinstance(stmt, ast.Expr):
        lower_expr(state, ctx, stmt.value)
        return []
    if isinstance(stmt, ast.If):
        lower_if(state, ctx, stmt)
        return []
    if isinstance(stmt, ast.For):
        lower_for(state, ctx, stmt)
        return []
    if isinstance(stmt, ast.While):
        lower_while(state, ctx, stmt)
        return []
    if isinstance(stmt, ast.AsyncFor):
        lower_async_for(state, ctx, stmt)
        return []
    if isinstance(stmt, ast.With):
        lower_with(state, ctx, stmt)
        return []
    if isinstance(stmt, ast.AsyncWith):
        lower_async_with(state, ctx, stmt)
        return []
    if isinstance(stmt, ast.TryStar):
        lower_try_star(state, ctx, stmt)
        return []
    if isinstance(stmt, ast.Try):
        lower_try(state, ctx, stmt)
        return []
    if isinstance(stmt, ast.Break):
        lower_break(state, ctx, stmt)
        return []
    if isinstance(stmt, ast.Continue):
        lower_continue(state, ctx, stmt)
        return []
    if isinstance(stmt, ast.Import):
        lower_import(state, ctx, stmt)
        return []
    if isinstance(stmt, ast.ImportFrom):
        lower_import_from(state, ctx, stmt)
        return []
    if isinstance(stmt, ast.Global) or isinstance(stmt, ast.Nonlocal) or isinstance(stmt, ast.Pass):
        return []
    if isinstance(stmt, ast.Raise):
        if stmt.exc is None:
            ctx.builder.terminate(attach_meta(state, Reraise(), stmt))
            return []
        exc = lower_expr(state, ctx, stmt.exc)
        cause = None if stmt.cause is None else lower_expr(state, ctx, stmt.cause)
        ctx.builder.terminate(attach_meta(state, Raise(exc=exc, cause=cause), stmt))
        return []
    if isinstance(stmt, ast.Delete):
        for target in stmt.targets:
            delete_target(state, ctx, target)
        return []
    if isinstance(stmt, ast.Match):
        lower_match(state, ctx, stmt)
        return []
    raise UnsupportedFeature(stmt, 'statement %s is not implemented in AST lowering' % type(stmt).__name__)

def lower_function_def(state, parent_ctx, node, is_async):
    # Build the nested function region first, then wrap it in a runtime function object.
    ensure_simple_arguments(state, node)
    child_table, child_code = take_child_region_inputs(state, parent_ctx, table_type=ChildRegionType.FUNCTION, symtable_name=node.name, code_name=node.name, owner=node)
    child_label = fresh_child_region_label(parent_ctx)
    child_name = child_region_name(state, node.name)
    child_path = child_name_path(state, parent_ctx, node.name, for_class=False)
    nested_region = compile_region_node(state, node=node, table=child_table, code_obj=child_code, name=child_name, name_path=child_path, is_class=False, label=child_label)
    default_values = COWList([lower_expr(state, parent_ctx, value) for value in node.args.defaults])
    kwonly_items = []
    for arg, default in zip(node.args.kwonlyargs, node.args.kw_defaults):
        if default is None:
            continue
        kwonly_items.append((arg.arg, lower_expr(state, parent_ctx, default)))
    func_temp = fresh_temp(state)
    parent_ctx.builder.emit(
        attach_meta(
            state,
            MakeFunction(dst=func_temp, code=child_label, defaults=default_values, kwdefaults=COWList(kwonly_items)),
            node,
        )
    )
    decorated = func_temp
    for decorator in reversed(node.decorator_list):
        decorator_value = lower_expr(state, parent_ctx, decorator)
        call_temp = fresh_temp(state)
        parent_ctx.builder.emit(attach_meta(state, Call(dst=call_temp, callee=decorator_value, args=normal_call_args([decorated]), kwargs=normal_call_kwargs(), flags=0), decorator))
        decorated = call_temp
    scope = scope_for_store(state, parent_ctx, node.name)
    parent_ctx.builder.emit(attach_meta(state, StoreName(src=decorated, scope=scope, name=node.name), node))
    return [nested_region]

def lower_class_def(state, parent_ctx, node):
    # Classes lower as a nested body region plus an explicit BuildClass operation.
    child_table, child_code = take_child_region_inputs(state, parent_ctx, table_type=ChildRegionType.CLASS, symtable_name=node.name, code_name=node.name, owner=node)
    child_label = fresh_child_region_label(parent_ctx)
    child_name = child_region_name(state, node.name)
    child_path = child_name_path(state, parent_ctx, node.name, for_class=True)
    nested_region = compile_region_node(state, node=node, table=child_table, code_obj=child_code, name=child_name, name_path=child_path, is_class=True, label=child_label)
    body_func = fresh_temp(state)
    parent_ctx.builder.emit(attach_meta(state, MakeFunction(dst=body_func, code=child_label), node))
    name_temp = const_value(state, parent_ctx, node.name, node)
    bases = [lower_expr(state, parent_ctx, base) for base in node.bases]
    keywords = []
    for keyword in node.keywords:
        if keyword.arg is None:
            raise UnsupportedFeature(keyword, 'class **kwargs are not implemented in AST lowering')
        keywords.append((keyword.arg, lower_expr(state, parent_ctx, keyword.value)))
    class_temp = fresh_temp(state)
    parent_ctx.builder.emit(attach_meta(state, BuildClass(dst=class_temp, body_func=body_func, name=name_temp, bases=COWList(bases), keywords=COWList(keywords)), node))
    decorated = class_temp
    for decorator in reversed(node.decorator_list):
        decorator_value = lower_expr(state, parent_ctx, decorator)
        call_temp = fresh_temp(state)
        parent_ctx.builder.emit(attach_meta(state, Call(dst=call_temp, callee=decorator_value, args=normal_call_args([decorated]), kwargs=normal_call_kwargs(), flags=0), decorator))
        decorated = call_temp
    scope = scope_for_store(state, parent_ctx, node.name)
    parent_ctx.builder.emit(attach_meta(state, StoreName(src=decorated, scope=scope, name=node.name), node))
    return [nested_region]

def lower_if(state, ctx, stmt):
    cond = lower_expr(state, ctx, stmt.test)
    then_label = ctx.builder.new_label()
    else_label = ctx.builder.new_label()
    end_label = ctx.builder.new_label()
    ctx.builder.terminate(attach_meta(state, Branch(cond=cond, true_label=then_label, false_label=else_label), stmt.test))
    ctx.builder.start_block(then_label)
    for body_stmt in stmt.body:
        nested = lower_stmt(state, ctx, body_stmt)
        if nested:
            raise UnsupportedFeature(body_stmt, 'nested region definitions inside if are not supported in this position')
    then_open = ctx.builder.is_open()
    if then_open:
        ctx.builder.terminate(attach_meta(state, Jump(target=end_label), stmt))
    ctx.builder.start_block(else_label)
    for orelse_stmt in stmt.orelse:
        nested = lower_stmt(state, ctx, orelse_stmt)
        if nested:
            raise UnsupportedFeature(orelse_stmt, 'nested region definitions inside else are not supported in this position')
    else_open = ctx.builder.is_open()
    if else_open:
        ctx.builder.terminate(attach_meta(state, Jump(target=end_label), stmt))
    if then_open or else_open:
        ctx.builder.start_block(end_label)

def push_loop(state, break_label, continue_label):
    state.loop_stack.append((break_label, continue_label))

def pop_loop(state):
    state.loop_stack.pop()

def current_loop(state, node):
    if not state.loop_stack:
        raise UnsupportedFeature(node, '%s outside loop' % type(node).__name__.lower())
    return state.loop_stack[-1]

def lower_break(state, ctx, stmt):
    break_label, _ = current_loop(state, stmt)
    ctx.builder.terminate(attach_meta(state, Escape(target=break_label), stmt))

def lower_continue(state, ctx, stmt):
    _, continue_label = current_loop(state, stmt)
    ctx.builder.terminate(attach_meta(state, Escape(target=continue_label), stmt))

def lower_import(state, ctx, stmt):
    for alias in stmt.names:
        module_temp = fresh_temp(state)
        ctx.builder.emit(attach_meta(state, ImportName(dst=module_temp, module=alias.name, fromlist=COWList(), level=0), stmt))
        store_name = alias.asname or alias.name.split('.', 1)[0]
        scope = scope_for_store(state, ctx, store_name)
        ctx.builder.emit(attach_meta(state, StoreName(src=module_temp, scope=scope, name=store_name), stmt))

def lower_import_from(state, ctx, stmt):
    module_name = stmt.module
    fromlist = [alias.name for alias in stmt.names]
    module_temp = fresh_temp(state)
    ctx.builder.emit(attach_meta(state, ImportName(dst=module_temp, module=module_name, fromlist=COWList(fromlist), level=stmt.level), stmt))
    if len(stmt.names) == 1 and stmt.names[0].name == '*':
        ctx.builder.emit(attach_meta(state, ImportStar(module_obj=module_temp), stmt))
        return
    for alias in stmt.names:
        imported = fresh_temp(state)
        ctx.builder.emit(attach_meta(state, ImportFrom(dst=imported, module_obj=module_temp, name=alias.name), stmt))
        store_name = alias.asname or alias.name
        scope = scope_for_store(state, ctx, store_name)
        ctx.builder.emit(attach_meta(state, StoreName(src=imported, scope=scope, name=store_name), stmt))

def lower_augassign(state, ctx, stmt):
    value = lower_expr(state, ctx, stmt.value)
    op = binary_op(state, stmt.op)
    target = stmt.target
    if isinstance(target, ast.Name):
        current = lower_expr(state, ctx, ast.Name(id=target.id, ctx=ast.Load()))
        result = fresh_temp(state)
        ctx.builder.emit(attach_meta(state, BinaryOp(dst=result, op=op, lhs=current, rhs=value), stmt))
        scope = scope_for_store(state, ctx, target.id)
        ctx.builder.emit(attach_meta(state, StoreName(src=result, scope=scope, name=target.id), stmt))
        return
    if isinstance(target, ast.Attribute):
        obj = lower_expr(state, ctx, target.value)
        current = fresh_temp(state)
        ctx.builder.emit(attach_meta(state, LoadAttr(dst=current, obj=obj, attr_name=target.attr), target))
        result = fresh_temp(state)
        ctx.builder.emit(attach_meta(state, BinaryOp(dst=result, op=op, lhs=current, rhs=value), stmt))
        ctx.builder.emit(attach_meta(state, StoreAttr(obj=obj, attr_name=target.attr, value=result), stmt))
        return
    if isinstance(target, ast.Subscript):
        obj = lower_expr(state, ctx, target.value)
        key = lower_slice_expr(state, ctx, target.slice)
        current = fresh_temp(state)
        ctx.builder.emit(attach_meta(state, LoadItem(dst=current, obj=obj, key=key), target))
        result = fresh_temp(state)
        ctx.builder.emit(attach_meta(state, BinaryOp(dst=result, op=op, lhs=current, rhs=value), stmt))
        ctx.builder.emit(attach_meta(state, StoreItem(obj=obj, key=key, value=result), stmt))
        return
    raise UnsupportedFeature(target, 'augmented assignment target %s is not implemented in AST lowering' % type(target).__name__)

def lower_for(state, ctx, stmt):
    iterable = lower_expr(state, ctx, stmt.iter)
    iter_temp = fresh_temp(state)
    ctx.builder.emit(attach_meta(state, GetIter(dst=iter_temp, iterable=iterable), stmt.iter))
    iter_name = fresh_synthetic_local(state, "for_iter")
    ctx.builder.emit(attach_meta(state, StoreName(src=iter_temp, scope=Scope.LOCAL, name=iter_name), stmt))
    header_label = ctx.builder.new_label()
    body_label = ctx.builder.new_label()
    orelse_label = ctx.builder.new_label() if stmt.orelse else None
    exit_label = orelse_label or ctx.builder.new_label()
    final_label = ctx.builder.new_label() if stmt.orelse else exit_label
    ctx.builder.terminate(attach_meta(state, Jump(target=header_label), stmt))
    ctx.builder.start_block(header_label)
    current_iter = fresh_temp(state)
    ctx.builder.emit(attach_meta(state, LoadName(dst=current_iter, scope=Scope.LOCAL, name=iter_name), stmt))
    value_dst = fresh_temp(state)
    ctx.builder.terminate(attach_meta(state, ForIter(iter_obj=current_iter, value_dst=value_dst, body_label=body_label, exit_label=exit_label), stmt))
    ctx.builder.start_block(body_label)
    push_loop(state, final_label, header_label)
    assign_target(state, ctx, stmt.target, value_dst)
    for body_stmt in stmt.body:
        nested = lower_stmt(state, ctx, body_stmt)
        if nested:
            raise UnsupportedFeature(body_stmt, 'nested region definitions inside for are not supported in this position')
    pop_loop(state, )
    if ctx.builder.is_open():
        ctx.builder.terminate(attach_meta(state, Jump(target=header_label), stmt))
    if stmt.orelse:
        ctx.builder.start_block(orelse_label)
        for orelse_stmt in stmt.orelse:
            nested = lower_stmt(state, ctx, orelse_stmt)
            if nested:
                raise UnsupportedFeature(orelse_stmt, 'nested region definitions inside for-else are not supported in this position')
        if ctx.builder.is_open():
            ctx.builder.terminate(attach_meta(state, Jump(target=final_label), stmt))
        ctx.builder.start_block(final_label)
    else:
        ctx.builder.start_block(exit_label)

def lower_while(state, ctx, stmt):
    cond_label = ctx.builder.new_label()
    body_label = ctx.builder.new_label()
    orelse_label = ctx.builder.new_label() if stmt.orelse else None
    exit_label = orelse_label or ctx.builder.new_label()
    final_label = ctx.builder.new_label() if stmt.orelse else exit_label
    ctx.builder.terminate(attach_meta(state, Jump(target=cond_label), stmt))
    ctx.builder.start_block(cond_label)
    cond = lower_expr(state, ctx, stmt.test)
    ctx.builder.terminate(attach_meta(state, Branch(cond=cond, true_label=body_label, false_label=exit_label), stmt.test))
    ctx.builder.start_block(body_label)
    push_loop(state, final_label, cond_label)
    for body_stmt in stmt.body:
        nested = lower_stmt(state, ctx, body_stmt)
        if nested:
            raise UnsupportedFeature(body_stmt, 'nested region definitions inside while are not supported in this position')
    pop_loop(state, )
    if ctx.builder.is_open():
        ctx.builder.terminate(attach_meta(state, Jump(target=cond_label), stmt))
    if stmt.orelse:
        ctx.builder.start_block(orelse_label)
        for orelse_stmt in stmt.orelse:
            nested = lower_stmt(state, ctx, orelse_stmt)
            if nested:
                raise UnsupportedFeature(orelse_stmt, 'nested region definitions inside while-else are not supported in this position')
        if ctx.builder.is_open():
            ctx.builder.terminate(attach_meta(state, Jump(target=final_label), stmt))
        ctx.builder.start_block(final_label)
    else:
        ctx.builder.start_block(exit_label)

def lower_async_for(state, ctx, stmt):
    # Async iteration is explicit: get __aiter__, await each __anext__, and catch StopAsyncIteration.
    iterable = lower_expr(state, ctx, stmt.iter)
    aiter_temp = fresh_temp(state)
    ctx.builder.emit(attach_meta(state, GetAIter(dst=aiter_temp, iterable=iterable), stmt.iter))
    iter_name = fresh_synthetic_local(state, "async_for_iter")
    ctx.builder.emit(attach_meta(state, StoreName(src=aiter_temp, scope=Scope.LOCAL, name=iter_name), stmt))
    header_label = ctx.builder.new_label()
    body_label = ctx.builder.new_label()
    stop_label = ctx.builder.new_label()
    stop_match_label = ctx.builder.new_label()
    stop_nomatch_label = ctx.builder.new_label()
    orelse_label = ctx.builder.new_label() if stmt.orelse else None
    exit_label = orelse_label or ctx.builder.new_label()
    final_label = ctx.builder.new_label() if stmt.orelse else exit_label
    ctx.builder.terminate(attach_meta(state, Jump(target=header_label), stmt))
    ctx.builder.start_block(header_label)
    current_iter = fresh_temp(state)
    ctx.builder.emit(attach_meta(state, LoadName(dst=current_iter, scope=Scope.LOCAL, name=iter_name), stmt))
    next_awaitable = fresh_temp(state)
    ctx.builder.emit(attach_meta(state, GetANext(dst=next_awaitable, aiter=current_iter), stmt))
    ctx.builder.emit(attach_meta(state, PushTry(except_label=stop_label), stmt))
    next_value = await_value(state, ctx, next_awaitable, stmt)
    ctx.builder.emit(attach_meta(state, PopTry(), stmt))
    ctx.builder.terminate(attach_meta(state, Jump(target=body_label), stmt))
    ctx.builder.start_block(stop_label)
    current_exc = current_exception_value(state, ctx, stmt)
    stop_type = builtin_const_value(state, ctx, builtins.StopAsyncIteration, stmt)
    matched = fresh_temp(state)
    ctx.builder.emit(attach_meta(state, CheckExcMatch(dst=matched, exc=current_exc, typ=stop_type), stmt))
    ctx.builder.terminate(attach_meta(state, Branch(cond=matched, true_label=stop_match_label, false_label=stop_nomatch_label), stmt))
    ctx.builder.start_block(stop_match_label)
    ctx.builder.emit(attach_meta(state, ClearException(), stmt))
    ctx.builder.terminate(attach_meta(state, Jump(target=exit_label), stmt))
    ctx.builder.start_block(stop_nomatch_label)
    ctx.builder.terminate(attach_meta(state, Reraise(), stmt))
    ctx.builder.start_block(body_label)
    push_loop(state, final_label, header_label)
    assign_target(state, ctx, stmt.target, next_value)
    for body_stmt in stmt.body:
        nested = lower_stmt(state, ctx, body_stmt)
        if nested:
            raise UnsupportedFeature(body_stmt, 'nested region definitions inside async for are not supported in this position')
    pop_loop(state, )
    if ctx.builder.is_open():
        ctx.builder.terminate(attach_meta(state, Jump(target=header_label), stmt))
    if stmt.orelse:
        ctx.builder.start_block(orelse_label)
        for orelse_stmt in stmt.orelse:
            nested = lower_stmt(state, ctx, orelse_stmt)
            if nested:
                raise UnsupportedFeature(orelse_stmt, 'nested region definitions inside async for-else are not supported in this position')
        if ctx.builder.is_open():
            ctx.builder.terminate(attach_meta(state, Jump(target=final_label), stmt))
        ctx.builder.start_block(final_label)
    else:
        ctx.builder.start_block(exit_label)

def await_value(state, ctx, value, node):
    awaitable = fresh_temp(state)
    ctx.builder.emit(attach_meta(state, GetAwaitable(dst=awaitable, value=value, where=0), node))
    awaited = fresh_temp(state)
    ctx.builder.emit(attach_meta(state, AwaitValue(dst=awaited, value=awaitable), node))
    return awaited

def call_and_await(state, ctx, callee, args, node):
    call_result = fresh_temp(state)
    ctx.builder.emit(attach_meta(state, Call(dst=call_result, callee=callee, args=normal_call_args(args), kwargs=normal_call_kwargs(), flags=0), node))
    return await_value(state, ctx, call_result, node)

def lower_with(state, ctx, stmt):
    lower_with_items(state, ctx, stmt.items, stmt.body, stmt, is_async=False)

def lower_async_with(state, ctx, stmt):
    lower_with_items(state, ctx, stmt.items, stmt.body, stmt, is_async=True)

def lower_with_items(state, ctx, items, body, owner, is_async=False):
    # Lower nested with-items recursively so each item gets its own synthetic finally path.
    if not items:
        for body_stmt in body:
            nested = lower_stmt(state, ctx, body_stmt)
            if nested:
                raise UnsupportedFeature(body_stmt, 'nested region definitions inside with are not supported in this position')
        return
    item = items[0]
    mgr = lower_expr(state, ctx, item.context_expr)
    exit_attr = '__aexit__' if is_async else '__exit__'
    enter_attr = '__aenter__' if is_async else '__enter__'
    exit_fn = fresh_temp(state)
    ctx.builder.emit(attach_meta(state, LoadAttr(dst=exit_fn, obj=mgr, attr_name=exit_attr), item.context_expr))
    enter_fn = fresh_temp(state)
    ctx.builder.emit(attach_meta(state, LoadAttr(dst=enter_fn, obj=mgr, attr_name=enter_attr), item.context_expr))
    if is_async:
        entered = call_and_await(state, ctx, enter_fn, [], item.context_expr)
    else:
        entered = fresh_temp(state)
        ctx.builder.emit(attach_meta(state, Call(dst=entered, callee=enter_fn, args=normal_call_args(), kwargs=normal_call_kwargs(), flags=0), item.context_expr))
    if item.optional_vars is not None:
        assign_target(state, ctx, item.optional_vars, entered)
    finally_label = ctx.builder.new_label()
    normal_exit_label = ctx.builder.new_label()
    exceptional_exit_label = ctx.builder.new_label()
    suppress_label = ctx.builder.new_label()
    propagate_label = ctx.builder.new_label()
    after_label = ctx.builder.new_label()
    ctx.builder.emit(attach_meta(state, PushTry(finally_label=finally_label), owner))
    lower_with_items(state, ctx, items[1:], body, owner, is_async=is_async)
    if ctx.builder.is_open():
        ctx.builder.emit(attach_meta(state, PopTry(), owner))
        ctx.builder.terminate(attach_meta(state, Jump(target=finally_label), owner))
    ctx.builder.start_block(finally_label)
    current_exc = current_exception_value(state, ctx, owner)
    none_exc = const_value(state, ctx, None, owner)
    is_none = fresh_temp(state)
    ctx.builder.emit(attach_meta(state, CompareOp(dst=is_none, cmp='is', lhs=current_exc, rhs=none_exc), owner))
    ctx.builder.terminate(attach_meta(state, Branch(cond=is_none, true_label=normal_exit_label, false_label=exceptional_exit_label), owner))
    ctx.builder.start_block(normal_exit_label)
    none1 = const_value(state, ctx, None, owner)
    none2 = const_value(state, ctx, None, owner)
    none3 = const_value(state, ctx, None, owner)
    if is_async:
        ignored = call_and_await(state, ctx, exit_fn, [none1, none2, none3], owner)
    else:
        ignored = fresh_temp(state)
        ctx.builder.emit(attach_meta(state, Call(dst=ignored, callee=exit_fn, args=normal_call_args([none1, none2, none3]), kwargs=normal_call_kwargs(), flags=0), owner))
    ctx.builder.emit(attach_meta(state, EndFinally(), owner))
    if ctx.builder.is_open():
        ctx.builder.terminate(attach_meta(state, Jump(target=after_label), owner))
    ctx.builder.start_block(exceptional_exit_label)
    type_name = builtin_const_value(state, ctx, builtins.type, owner)
    exc_type = fresh_temp(state)
    ctx.builder.emit(attach_meta(state, Call(dst=exc_type, callee=type_name, args=normal_call_args([current_exc]), kwargs=normal_call_kwargs(), flags=0), owner))
    traceback = fresh_temp(state)
    ctx.builder.emit(attach_meta(state, LoadAttr(dst=traceback, obj=current_exc, attr_name='__traceback__'), owner))
    if is_async:
        exit_result = call_and_await(state, ctx, exit_fn, [exc_type, current_exc, traceback], owner)
    else:
        exit_result = fresh_temp(state)
        ctx.builder.emit(attach_meta(state, Call(dst=exit_result, callee=exit_fn, args=normal_call_args([exc_type, current_exc, traceback]), kwargs=normal_call_kwargs(), flags=0), owner))
    ctx.builder.terminate(attach_meta(state, Branch(cond=exit_result, true_label=suppress_label, false_label=propagate_label), owner))
    ctx.builder.start_block(suppress_label)
    ctx.builder.emit(attach_meta(state, ClearException(), owner))
    ctx.builder.emit(attach_meta(state, EndFinally(), owner))
    if ctx.builder.is_open():
        ctx.builder.terminate(attach_meta(state, Jump(target=after_label), owner))
    ctx.builder.start_block(propagate_label)
    ctx.builder.emit(attach_meta(state, EndFinally(), owner))
    if ctx.builder.is_open():
        ctx.builder.terminate(attach_meta(state, Jump(target=after_label), owner))
    ctx.builder.start_block(after_label)

def lower_try(state, ctx, stmt):
    # Try statements are represented with explicit synthetic try targets and CFG dispatch blocks.
    if not stmt.handlers and (not stmt.finalbody):
        raise UnsupportedFeature(stmt, 'try without except/finally is not implemented in AST lowering')
    except_dispatch_label = ctx.builder.new_label() if stmt.handlers else None
    finally_label = ctx.builder.new_label() if stmt.finalbody else None
    after_label = ctx.builder.new_label()
    orelse_label = ctx.builder.new_label() if stmt.orelse else after_label
    if stmt.finalbody:
        ctx.builder.emit(attach_meta(state, PushTry(finally_label=finally_label), stmt))
    if stmt.handlers:
        ctx.builder.emit(attach_meta(state, PushTry(except_label=except_dispatch_label), stmt))
    for body_stmt in stmt.body:
        nested = lower_stmt(state, ctx, body_stmt)
        if nested:
            raise UnsupportedFeature(body_stmt, 'nested region definitions inside try are not supported in this position')
    if ctx.builder.is_open():
        if stmt.handlers:
            ctx.builder.emit(attach_meta(state, PopTry(), stmt))
        if stmt.orelse:
            ctx.builder.terminate(attach_meta(state, Jump(target=orelse_label), stmt))
        elif stmt.finalbody:
            ctx.builder.emit(attach_meta(state, PopTry(), stmt))
            ctx.builder.terminate(attach_meta(state, Jump(target=finally_label), stmt))
        else:
            ctx.builder.terminate(attach_meta(state, Jump(target=after_label), stmt))
    if stmt.orelse:
        ctx.builder.start_block(orelse_label)
        for orelse_stmt in stmt.orelse:
            nested = lower_stmt(state, ctx, orelse_stmt)
            if nested:
                raise UnsupportedFeature(orelse_stmt, 'nested region definitions inside else are not supported in this position')
        if ctx.builder.is_open():
            if stmt.finalbody:
                ctx.builder.emit(attach_meta(state, PopTry(), stmt))
                ctx.builder.terminate(attach_meta(state, Jump(target=finally_label), stmt))
            else:
                ctx.builder.terminate(attach_meta(state, Jump(target=after_label), stmt))
    if stmt.handlers:
        ctx.builder.start_block(except_dispatch_label)
        current_exc = current_exception_value(state, ctx, stmt)
        no_match_label = ctx.builder.new_label()
        next_label = None
        for index, handler in enumerate(stmt.handlers):
            is_last = index == len(stmt.handlers) - 1
            body_label = ctx.builder.new_label()
            next_label = no_match_label if is_last else ctx.builder.new_label()
            if handler.type is None:
                ctx.builder.terminate(attach_meta(state, Jump(target=body_label), handler))
            else:
                typ = lower_expr(state, ctx, handler.type)
                match = fresh_temp(state)
                ctx.builder.emit(attach_meta(state, CheckExcMatch(dst=match, exc=current_exc, typ=typ), handler))
                ctx.builder.terminate(attach_meta(state, Branch(cond=match, true_label=body_label, false_label=next_label), handler))
            ctx.builder.start_block(body_label)
            if handler.name:
                scope = scope_for_store(state, ctx, handler.name)
                ctx.builder.emit(attach_meta(state, StoreName(src=current_exc, scope=scope, name=handler.name), handler))
            for handler_stmt in handler.body:
                nested = lower_stmt(state, ctx, handler_stmt)
                if nested:
                    raise UnsupportedFeature(handler_stmt, 'nested region definitions inside except are not supported in this position')
            if ctx.builder.is_open():
                if handler.name:
                    none_temp = const_value(state, ctx, None, handler)
                    scope = scope_for_store(state, ctx, handler.name)
                    ctx.builder.emit(attach_meta(state, StoreName(src=none_temp, scope=scope, name=handler.name), handler))
                    ctx.builder.emit(attach_meta(state, DeleteName(scope=scope, name=handler.name), handler))
                ctx.builder.emit(attach_meta(state, ClearException(), handler))
                if stmt.finalbody:
                    ctx.builder.emit(attach_meta(state, PopTry(), handler))
                    ctx.builder.terminate(attach_meta(state, Jump(target=finally_label), handler))
                else:
                    ctx.builder.terminate(attach_meta(state, Jump(target=after_label), handler))
            if not is_last:
                ctx.builder.start_block(next_label)
        ctx.builder.start_block(no_match_label)
        ctx.builder.terminate(attach_meta(state, Reraise(), stmt))
    if stmt.finalbody:
        ctx.builder.start_block(finally_label)
        for final_stmt in stmt.finalbody:
            nested = lower_stmt(state, ctx, final_stmt)
            if nested:
                raise UnsupportedFeature(final_stmt, 'nested region definitions inside finally are not supported in this position')
        if ctx.builder.is_open():
            ctx.builder.emit(attach_meta(state, EndFinally(), stmt))
            if ctx.builder.is_open():
                ctx.builder.terminate(attach_meta(state, Jump(target=after_label), stmt))
    ctx.builder.start_block(after_label)


def lower_try_star(state, ctx, stmt):
    # Exception groups use the same explicit CFG shape as `try`, but handler tests go through
    # the dedicated exception-group matcher instruction.
    if not stmt.handlers and (not stmt.finalbody):
        raise UnsupportedFeature(stmt, 'try* without except/finally is not implemented in AST lowering')
    except_dispatch_label = ctx.builder.new_label() if stmt.handlers else None
    finally_label = ctx.builder.new_label() if stmt.finalbody else None
    after_label = ctx.builder.new_label()
    orelse_label = ctx.builder.new_label() if stmt.orelse else after_label
    if stmt.finalbody:
        ctx.builder.emit(attach_meta(state, PushTry(finally_label=finally_label), stmt))
    if stmt.handlers:
        ctx.builder.emit(attach_meta(state, PushTry(except_label=except_dispatch_label), stmt))
    for body_stmt in stmt.body:
        nested = lower_stmt(state, ctx, body_stmt)
        if nested:
            raise UnsupportedFeature(body_stmt, 'nested region definitions inside try* are not supported in this position')
    if ctx.builder.is_open():
        if stmt.handlers:
            ctx.builder.emit(attach_meta(state, PopTry(), stmt))
        if stmt.orelse:
            ctx.builder.terminate(attach_meta(state, Jump(target=orelse_label), stmt))
        elif stmt.finalbody:
            ctx.builder.emit(attach_meta(state, PopTry(), stmt))
            ctx.builder.terminate(attach_meta(state, Jump(target=finally_label), stmt))
        else:
            ctx.builder.terminate(attach_meta(state, Jump(target=after_label), stmt))
    if stmt.orelse:
        ctx.builder.start_block(orelse_label)
        for orelse_stmt in stmt.orelse:
            nested = lower_stmt(state, ctx, orelse_stmt)
            if nested:
                raise UnsupportedFeature(orelse_stmt, 'nested region definitions inside try*-else are not supported in this position')
        if ctx.builder.is_open():
            if stmt.finalbody:
                ctx.builder.emit(attach_meta(state, PopTry(), stmt))
                ctx.builder.terminate(attach_meta(state, Jump(target=finally_label), stmt))
            else:
                ctx.builder.terminate(attach_meta(state, Jump(target=after_label), stmt))
    if stmt.handlers:
        ctx.builder.start_block(except_dispatch_label)
        current_exc = current_exception_value(state, ctx, stmt)
        no_match_label = ctx.builder.new_label()
        next_label = None
        for index, handler in enumerate(stmt.handlers):
            is_last = index == len(stmt.handlers) - 1
            body_label = ctx.builder.new_label()
            next_label = no_match_label if is_last else ctx.builder.new_label()
            if handler.type is None:
                ctx.builder.terminate(attach_meta(state, Jump(target=body_label), handler))
            else:
                typ = lower_expr(state, ctx, handler.type)
                match = fresh_temp(state)
                ctx.builder.emit(attach_meta(state, CheckEGMatch(dst=match, exc=current_exc, typ=typ), handler))
                ctx.builder.terminate(attach_meta(state, Branch(cond=match, true_label=body_label, false_label=next_label), handler))
            ctx.builder.start_block(body_label)
            if handler.name:
                scope = scope_for_store(state, ctx, handler.name)
                ctx.builder.emit(attach_meta(state, StoreName(src=current_exc, scope=scope, name=handler.name), handler))
            for handler_stmt in handler.body:
                nested = lower_stmt(state, ctx, handler_stmt)
                if nested:
                    raise UnsupportedFeature(handler_stmt, 'nested region definitions inside except* are not supported in this position')
            if ctx.builder.is_open():
                if handler.name:
                    none_temp = const_value(state, ctx, None, handler)
                    scope = scope_for_store(state, ctx, handler.name)
                    ctx.builder.emit(attach_meta(state, StoreName(src=none_temp, scope=scope, name=handler.name), handler))
                    ctx.builder.emit(attach_meta(state, DeleteName(scope=scope, name=handler.name), handler))
                ctx.builder.emit(attach_meta(state, ClearException(), handler))
                if stmt.finalbody:
                    ctx.builder.emit(attach_meta(state, PopTry(), handler))
                    ctx.builder.terminate(attach_meta(state, Jump(target=finally_label), handler))
                else:
                    ctx.builder.terminate(attach_meta(state, Jump(target=after_label), handler))
            if not is_last:
                ctx.builder.start_block(next_label)
        ctx.builder.start_block(no_match_label)
        ctx.builder.terminate(attach_meta(state, Reraise(), stmt))
    if stmt.finalbody:
        ctx.builder.start_block(finally_label)
        for final_stmt in stmt.finalbody:
            nested = lower_stmt(state, ctx, final_stmt)
            if nested:
                raise UnsupportedFeature(final_stmt, 'nested region definitions inside try*-finally are not supported in this position')
        if ctx.builder.is_open():
            ctx.builder.emit(attach_meta(state, EndFinally(), stmt))
            if ctx.builder.is_open():
                ctx.builder.terminate(attach_meta(state, Jump(target=after_label), stmt))
    ctx.builder.start_block(after_label)


def bind_pattern_name(state, ctx, name, value, node):
    if name is None:
        return
    scope = scope_for_store(state, ctx, name)
    ctx.builder.emit(attach_meta(state, StoreName(src=value, scope=scope, name=name), node))


def emit_call(state, ctx, callee, args, node):
    result = fresh_temp(state)
    ctx.builder.emit(attach_meta(state, Call(dst=result, callee=callee, args=normal_call_args(args), kwargs=normal_call_kwargs(), flags=0), node))
    return result


def emit_builtin_call(state, ctx, builtin_obj, args, node):
    callee = builtin_const_value(state, ctx, builtin_obj, node)
    return emit_call(state, ctx, callee, args, node)


def emit_pattern_length_check(state, ctx, subject, expected, allow_extra, node):
    length = emit_builtin_call(state, ctx, builtins.len, [subject], node)
    wanted = const_value(state, ctx, expected, node)
    matched = fresh_temp(state)
    cmp = '>=' if allow_extra else '=='
    ctx.builder.emit(attach_meta(state, CompareOp(dst=matched, cmp=cmp, lhs=length, rhs=wanted), node))
    return matched


def lower_match(state, ctx, stmt):
    subject = lower_expr(state, ctx, stmt.subject)
    end_label = ctx.builder.new_label()
    for index, case in enumerate(stmt.cases):
        body_label = ctx.builder.new_label()
        failure_label = end_label if index == len(stmt.cases) - 1 else ctx.builder.new_label()
        lower_pattern(state, ctx, case.pattern, subject, body_label, failure_label)
        ctx.builder.start_block(body_label)
        if case.guard is not None:
            guarded_body_label = ctx.builder.new_label()
            guard = lower_expr(state, ctx, case.guard)
            ctx.builder.terminate(attach_meta(state, Branch(cond=guard, true_label=guarded_body_label, false_label=failure_label), case.guard))
            ctx.builder.start_block(guarded_body_label)
        for body_stmt in case.body:
            nested = lower_stmt(state, ctx, body_stmt)
            if nested:
                raise UnsupportedFeature(body_stmt, 'nested region definitions inside match-case are not supported in this position')
        if ctx.builder.is_open():
            ctx.builder.terminate(attach_meta(state, Jump(target=end_label), case.pattern))
        if failure_label is not end_label:
            ctx.builder.start_block(failure_label)
    ctx.builder.start_block(end_label)


def lower_pattern_values(state, ctx, patterns, values, success_label, failure_label, node):
    if len(patterns) != len(values):
        raise UnsupportedFeature(node, 'pattern arity mismatch during AST lowering')
    if not patterns:
        ctx.builder.terminate(attach_meta(state, Jump(target=success_label), node))
        return
    for index, (pattern, value) in enumerate(zip(patterns, values)):
        is_last = index == len(patterns) - 1
        next_label = success_label if is_last else ctx.builder.new_label()
        lower_pattern(state, ctx, pattern, value, next_label, failure_label)
        if not is_last:
            ctx.builder.start_block(next_label)


def lower_pattern(state, ctx, pattern, subject, success_label, failure_label):
    if isinstance(pattern, ast.MatchAs):
        if pattern.pattern is None:
            bind_pattern_name(state, ctx, pattern.name, subject, pattern)
            ctx.builder.terminate(attach_meta(state, Jump(target=success_label), pattern))
            return
        matched_label = ctx.builder.new_label()
        lower_pattern(state, ctx, pattern.pattern, subject, matched_label, failure_label)
        ctx.builder.start_block(matched_label)
        bind_pattern_name(state, ctx, pattern.name, subject, pattern)
        ctx.builder.terminate(attach_meta(state, Jump(target=success_label), pattern))
        return
    if isinstance(pattern, ast.MatchValue):
        wanted = lower_expr(state, ctx, pattern.value)
        matched = fresh_temp(state)
        ctx.builder.emit(attach_meta(state, CompareOp(dst=matched, cmp='==', lhs=subject, rhs=wanted), pattern))
        ctx.builder.terminate(attach_meta(state, Branch(cond=matched, true_label=success_label, false_label=failure_label), pattern))
        return
    if isinstance(pattern, ast.MatchSingleton):
        wanted = const_value(state, ctx, pattern.value, pattern)
        matched = fresh_temp(state)
        ctx.builder.emit(attach_meta(state, CompareOp(dst=matched, cmp='is', lhs=subject, rhs=wanted), pattern))
        ctx.builder.terminate(attach_meta(state, Branch(cond=matched, true_label=success_label, false_label=failure_label), pattern))
        return
    if isinstance(pattern, ast.MatchOr):
        for index, alternative in enumerate(pattern.patterns):
            next_failure = failure_label if index == len(pattern.patterns) - 1 else ctx.builder.new_label()
            lower_pattern(state, ctx, alternative, subject, success_label, next_failure)
            if next_failure is not failure_label:
                ctx.builder.start_block(next_failure)
        return
    if isinstance(pattern, ast.MatchSequence):
        sequence_ok = fresh_temp(state)
        ctx.builder.emit(attach_meta(state, MatchSequence(dst=sequence_ok, value=subject), pattern))
        sequence_label = ctx.builder.new_label()
        ctx.builder.terminate(attach_meta(state, Branch(cond=sequence_ok, true_label=sequence_label, false_label=failure_label), pattern))
        ctx.builder.start_block(sequence_label)
        star_indexes = [index for index, child in enumerate(pattern.patterns) if isinstance(child, ast.MatchStar)]
        if len(star_indexes) > 1:
            raise UnsupportedFeature(pattern, 'multiple starred sequence patterns are not implemented in AST lowering')
        unpack_label = ctx.builder.new_label()
        if not star_indexes:
            length_ok = emit_pattern_length_check(state, ctx, subject, len(pattern.patterns), allow_extra=False, node=pattern)
            ctx.builder.terminate(attach_meta(state, Branch(cond=length_ok, true_label=unpack_label, false_label=failure_label), pattern))
            ctx.builder.start_block(unpack_label)
            values = [fresh_temp(state) for _ in pattern.patterns]
            ctx.builder.emit(attach_meta(state, Unpack(src=subject, dsts=COWList(values)), pattern))
            lower_pattern_values(state, ctx, pattern.patterns, values, success_label, failure_label, pattern)
            return
        star_index = star_indexes[0]
        minimum_size = len(pattern.patterns) - 1
        length_ok = emit_pattern_length_check(state, ctx, subject, minimum_size, allow_extra=True, node=pattern)
        ctx.builder.terminate(attach_meta(state, Branch(cond=length_ok, true_label=unpack_label, false_label=failure_label), pattern))
        ctx.builder.start_block(unpack_label)
        before_values = [fresh_temp(state) for _ in pattern.patterns[:star_index]]
        star_value = fresh_temp(state)
        after_values = [fresh_temp(state) for _ in pattern.patterns[star_index + 1:]]
        ctx.builder.emit(attach_meta(state, Unpack(src=subject, dsts=COWList(before_values + [star_value] + after_values), star_index=star_index), pattern))
        values = before_values + [star_value] + after_values
        lower_pattern_values(state, ctx, pattern.patterns, values, success_label, failure_label, pattern)
        return
    if isinstance(pattern, ast.MatchStar):
        bind_pattern_name(state, ctx, pattern.name, subject, pattern)
        ctx.builder.terminate(attach_meta(state, Jump(target=success_label), pattern))
        return
    if isinstance(pattern, ast.MatchMapping):
        if pattern.rest is not None:
            raise UnsupportedFeature(pattern, 'mapping rest patterns are not implemented in AST lowering')
        mapping_ok = fresh_temp(state)
        ctx.builder.emit(attach_meta(state, MatchMapping(dst=mapping_ok, value=subject), pattern))
        mapping_label = ctx.builder.new_label()
        ctx.builder.terminate(attach_meta(state, Branch(cond=mapping_ok, true_label=mapping_label, false_label=failure_label), pattern))
        ctx.builder.start_block(mapping_label)
        if not pattern.keys:
            ctx.builder.terminate(attach_meta(state, Jump(target=success_label), pattern))
            return
        key_values = [lower_expr(state, ctx, key) for key in pattern.keys]
        keys_tuple = build_tuple(state, ctx, key_values, pattern)
        matched_items = fresh_temp(state)
        ctx.builder.emit(attach_meta(state, MatchKeys(dst=matched_items, mapping=subject, keys=keys_tuple), pattern))
        none_value = const_value(state, ctx, None, pattern)
        found = fresh_temp(state)
        ctx.builder.emit(attach_meta(state, CompareOp(dst=found, cmp='is not', lhs=matched_items, rhs=none_value), pattern))
        values_label = ctx.builder.new_label()
        ctx.builder.terminate(attach_meta(state, Branch(cond=found, true_label=values_label, false_label=failure_label), pattern))
        ctx.builder.start_block(values_label)
        values = [fresh_temp(state) for _ in pattern.patterns]
        ctx.builder.emit(attach_meta(state, Unpack(src=matched_items, dsts=COWList(values)), pattern))
        lower_pattern_values(state, ctx, pattern.patterns, values, success_label, failure_label, pattern)
        return
    if isinstance(pattern, ast.MatchClass):
        cls = lower_expr(state, ctx, pattern.cls)
        matched = fresh_temp(state)
        ctx.builder.emit(attach_meta(state, MatchClass(dst=matched, value=subject, cls=cls, attr_names=COWList(pattern.kwd_attrs), positional_count=len(pattern.patterns)), pattern))
        attrs_label = ctx.builder.new_label()
        ctx.builder.terminate(attach_meta(state, Branch(cond=matched, true_label=attrs_label, false_label=failure_label), pattern))
        ctx.builder.start_block(attrs_label)
        values = []
        patterns = []
        if pattern.patterns:
            match_args = fresh_temp(state)
            ctx.builder.emit(attach_meta(state, LoadAttr(dst=match_args, obj=cls, attr_name='__match_args__'), pattern))
            getattr_fn = builtin_const_value(state, ctx, builtins.getattr, pattern)
            for index, positional_pattern in enumerate(pattern.patterns):
                position = const_value(state, ctx, index, pattern)
                attr_name = fresh_temp(state)
                ctx.builder.emit(attach_meta(state, LoadItem(dst=attr_name, obj=match_args, key=position), pattern))
                attr_value = emit_call(state, ctx, getattr_fn, [subject, attr_name], pattern)
                values.append(attr_value)
                patterns.append(positional_pattern)
        for attr_name, keyword_pattern in zip(pattern.kwd_attrs, pattern.kwd_patterns):
            attr_value = fresh_temp(state)
            ctx.builder.emit(attach_meta(state, LoadAttr(dst=attr_value, obj=subject, attr_name=attr_name), pattern))
            values.append(attr_value)
            patterns.append(keyword_pattern)
        if not patterns:
            ctx.builder.terminate(attach_meta(state, Jump(target=success_label), pattern))
            return
        lower_pattern_values(state, ctx, patterns, values, success_label, failure_label, pattern)
        return
    raise UnsupportedFeature(pattern, 'pattern %s is not implemented in AST lowering' % type(pattern).__name__)


def lower_ifexp(state, ctx, expr):
    result_name = fresh_synthetic_local(state, "ifexp_result")
    then_label = ctx.builder.new_label()
    else_label = ctx.builder.new_label()
    end_label = ctx.builder.new_label()
    cond = lower_expr(state, ctx, expr.test)
    ctx.builder.terminate(attach_meta(state, Branch(cond=cond, true_label=then_label, false_label=else_label), expr.test))
    ctx.builder.start_block(then_label)
    then_value = lower_expr(state, ctx, expr.body)
    ctx.builder.emit(attach_meta(state, StoreName(src=then_value, scope=Scope.LOCAL, name=result_name), expr.body))
    ctx.builder.terminate(attach_meta(state, Jump(target=end_label), expr.body))
    ctx.builder.start_block(else_label)
    else_value = lower_expr(state, ctx, expr.orelse)
    ctx.builder.emit(attach_meta(state, StoreName(src=else_value, scope=Scope.LOCAL, name=result_name), expr.orelse))
    ctx.builder.terminate(attach_meta(state, Jump(target=end_label), expr.orelse))
    ctx.builder.start_block(end_label)
    result = fresh_temp(state)
    ctx.builder.emit(attach_meta(state, LoadName(dst=result, scope=Scope.LOCAL, name=result_name), expr))
    ctx.builder.emit(attach_meta(state, DeleteName(scope=Scope.LOCAL, name=result_name), expr))
    return result

def lower_bool_op(state, ctx, expr):
    if not expr.values:
        raise UnsupportedFeature(expr, 'empty boolean operation is not valid')
    if len(expr.values) == 1:
        return lower_expr(state, ctx, expr.values[0])
    result_name = fresh_synthetic_local(state, "boolop_result")
    end_label = ctx.builder.new_label()
    is_and = isinstance(expr.op, ast.And)
    for value_expr in expr.values[:-1]:
        value = lower_expr(state, ctx, value_expr)
        ctx.builder.emit(attach_meta(state, StoreName(src=value, scope=Scope.LOCAL, name=result_name), value_expr))
        next_label = ctx.builder.new_label()
        if is_and:
            ctx.builder.terminate(attach_meta(state, Branch(cond=value, true_label=next_label, false_label=end_label), value_expr))
        else:
            ctx.builder.terminate(attach_meta(state, Branch(cond=value, true_label=end_label, false_label=next_label), value_expr))
        ctx.builder.start_block(next_label)
    last_value = lower_expr(state, ctx, expr.values[-1])
    ctx.builder.emit(attach_meta(state, StoreName(src=last_value, scope=Scope.LOCAL, name=result_name), expr.values[-1]))
    ctx.builder.terminate(attach_meta(state, Jump(target=end_label), expr))
    ctx.builder.start_block(end_label)
    result = fresh_temp(state)
    ctx.builder.emit(attach_meta(state, LoadName(dst=result, scope=Scope.LOCAL, name=result_name), expr))
    ctx.builder.emit(attach_meta(state, DeleteName(scope=Scope.LOCAL, name=result_name), expr))
    return result

def emit_method_call(state, ctx, obj, method_name, args, node):
    method = fresh_temp(state)
    ctx.builder.emit(attach_meta(state, LoadAttr(dst=method, obj=obj, attr_name=method_name), node))
    result = fresh_temp(state)
    ctx.builder.emit(attach_meta(state, Call(dst=result, callee=method, args=normal_call_args(args), kwargs=normal_call_kwargs(), flags=0), node))
    return result


def comprehension_iter_name(state, ctx, generator, index):
    # Generator-expression regions receive the outer iterator as an implicit first argument.
    if index == 0 and isinstance(ctx.node, ast.GeneratorExp) and ctx.code_obj.co_argcount:
        return ctx.code_obj.co_varnames[0], False
    iterable = lower_expr(state, ctx, generator.iter)
    iter_temp = fresh_temp(state)
    if generator.is_async:
        ctx.builder.emit(attach_meta(state, GetAIter(dst=iter_temp, iterable=iterable), generator.iter))
    else:
        ctx.builder.emit(attach_meta(state, GetIter(dst=iter_temp, iterable=iterable), generator.iter))
    iter_name = fresh_synthetic_local(state, "comprehension_iter")
    ctx.builder.emit(attach_meta(state, StoreName(src=iter_temp, scope=Scope.LOCAL, name=iter_name), generator))
    return iter_name, True


def lower_comprehension(state, ctx, expr, generators, emit_item, result):
    # Save any outer bindings shadowed by the comprehension targets, then restore them afterwards.
    if not generators:
        raise UnsupportedFeature(expr, 'comprehension without generators is not valid')
    saved_names = snapshot_comprehension_target_names(state, ctx, generators, expr)
    after_label = ctx.builder.new_label()
    lower_comprehension_generator(state, ctx, generators, 0, emit_item, after_label, expr)
    ctx.builder.start_block(after_label)
    restore_comprehension_target_names(state, ctx, saved_names, expr)
    return result


def lower_comprehension_generator(state, ctx, generators, index, emit_item, exhaustion_label, owner):
    # The sync and async cases share the same CFG shape; async just inserts await + exception edges.
    generator = generators[index]
    iter_name, owns_iter_name = comprehension_iter_name(state, ctx, generator, index)
    header_label = ctx.builder.new_label()
    body_label = ctx.builder.new_label()
    cleanup_label = ctx.builder.new_label()
    if generator.is_async:
        stop_label = ctx.builder.new_label()
        stop_match_label = ctx.builder.new_label()
        stop_nomatch_label = ctx.builder.new_label()
    ctx.builder.terminate(attach_meta(state, Jump(target=header_label), generator))
    ctx.builder.start_block(header_label)
    current_iter = fresh_temp(state)
    ctx.builder.emit(attach_meta(state, LoadName(dst=current_iter, scope=Scope.LOCAL, name=iter_name), generator))
    if generator.is_async:
        next_awaitable = fresh_temp(state)
        ctx.builder.emit(attach_meta(state, GetANext(dst=next_awaitable, aiter=current_iter), generator))
        ctx.builder.emit(attach_meta(state, PushTry(except_label=stop_label), generator))
        value_dst = await_value(state, ctx, next_awaitable, generator)
        ctx.builder.emit(attach_meta(state, PopTry(), generator))
        ctx.builder.terminate(attach_meta(state, Jump(target=body_label), generator))
        ctx.builder.start_block(stop_label)
        current_exc = current_exception_value(state, ctx, generator)
        stop_type = builtin_const_value(state, ctx, builtins.StopAsyncIteration, generator)
        matched = fresh_temp(state)
        ctx.builder.emit(attach_meta(state, CheckExcMatch(dst=matched, exc=current_exc, typ=stop_type), generator))
        ctx.builder.terminate(attach_meta(state, Branch(cond=matched, true_label=stop_match_label, false_label=stop_nomatch_label), generator))
        ctx.builder.start_block(stop_match_label)
        ctx.builder.emit(attach_meta(state, ClearException(), generator))
        ctx.builder.terminate(attach_meta(state, Jump(target=cleanup_label), generator))
        ctx.builder.start_block(stop_nomatch_label)
        ctx.builder.terminate(attach_meta(state, Reraise(), generator))
        ctx.builder.start_block(body_label)
    else:
        value_dst = fresh_temp(state)
        ctx.builder.terminate(attach_meta(state, ForIter(iter_obj=current_iter, value_dst=value_dst, body_label=body_label, exit_label=cleanup_label), generator))
        ctx.builder.start_block(body_label)
    assign_target(state, ctx, generator.target, value_dst)
    for if_expr in generator.ifs:
        next_label = ctx.builder.new_label()
        cond = lower_expr(state, ctx, if_expr)
        ctx.builder.terminate(attach_meta(state, Branch(cond=cond, true_label=next_label, false_label=header_label), if_expr))
        ctx.builder.start_block(next_label)
    if index + 1 < len(generators):
        lower_comprehension_generator(state, ctx, generators, index + 1, emit_item, header_label, owner)
    else:
        emit_item()
        if ctx.builder.is_open():
            ctx.builder.terminate(attach_meta(state, Jump(target=header_label), owner))
    ctx.builder.start_block(cleanup_label)
    if owns_iter_name:
        ctx.builder.emit(attach_meta(state, DeleteName(scope=Scope.LOCAL, name=iter_name), generator))
    ctx.builder.terminate(attach_meta(state, Jump(target=exhaustion_label), generator))


def lower_list_comp(state, ctx, expr):
    result = fresh_temp(state)
    ctx.builder.emit(attach_meta(state, BuildList(dst=result, items=COWList()), expr))

    def emit_item():
        item = lower_expr(state, ctx, expr.elt)
        emit_method_call(state, ctx, result, 'append', [item], expr)

    return lower_comprehension(state, ctx, expr, expr.generators, emit_item, result)


def lower_set_comp(state, ctx, expr):
    result = fresh_temp(state)
    ctx.builder.emit(attach_meta(state, BuildSet(dst=result, items=COWList()), expr))

    def emit_item():
        item = lower_expr(state, ctx, expr.elt)
        emit_method_call(state, ctx, result, 'add', [item], expr)

    return lower_comprehension(state, ctx, expr, expr.generators, emit_item, result)


def lower_dict_comp(state, ctx, expr):
    result = fresh_temp(state)
    ctx.builder.emit(attach_meta(state, BuildMap(dst=result, items=COWList()), expr))

    def emit_item():
        key = lower_expr(state, ctx, expr.key)
        value = lower_expr(state, ctx, expr.value)
        ctx.builder.emit(attach_meta(state, StoreItem(obj=result, key=key, value=value), expr))

    return lower_comprehension(state, ctx, expr, expr.generators, emit_item, result)


def lower_generator_exp(state, ctx, expr):
    # Generator expressions lower to an explicit nested region plus a call that seeds the outer iterator.
    child_table, child_code = take_child_region_inputs(state, ctx, table_type=ChildRegionType.FUNCTION, symtable_name='genexpr', code_name='<genexpr>', owner=expr)
    child_label = fresh_child_region_label(ctx)
    child_name = child_region_name(state, '<genexpr>')
    child_path = child_name_path(state, ctx, '<genexpr>', for_class=False)
    nested_region = compile_region_node(state, node=expr, table=child_table, code_obj=child_code, name=child_name, name_path=child_path, is_class=False, label=child_label)
    state.region_nested_stacks[-1].append(nested_region)
    outer_iterable = lower_expr(state, ctx, expr.generators[0].iter)
    outer_iter = fresh_temp(state)
    if expr.generators[0].is_async:
        ctx.builder.emit(attach_meta(state, GetAIter(dst=outer_iter, iterable=outer_iterable), expr.generators[0].iter))
    else:
        ctx.builder.emit(attach_meta(state, GetIter(dst=outer_iter, iterable=outer_iterable), expr.generators[0].iter))
    func = fresh_temp(state)
    ctx.builder.emit(attach_meta(state, MakeFunction(dst=func, code=child_label), expr))
    call = fresh_temp(state)
    ctx.builder.emit(attach_meta(state, Call(dst=call, callee=func, args=normal_call_args([outer_iter]), kwargs=normal_call_kwargs(), flags=0), expr))
    return call


def lower_lambda(state, ctx, expr):
    ensure_simple_arguments(state, expr)
    child_table, child_code = take_child_region_inputs(state, ctx, table_type=ChildRegionType.FUNCTION, symtable_name='lambda', code_name='<lambda>', owner=expr)
    child_label = fresh_child_region_label(ctx)
    child_name = child_region_name(state, '<lambda>')
    child_path = child_name_path(state, ctx, '<lambda>', for_class=False)
    nested_region = compile_region_node(state, node=expr, table=child_table, code_obj=child_code, name=child_name, name_path=child_path, is_class=False, label=child_label)
    state.region_nested_stacks[-1].append(nested_region)
    default_values = COWList([lower_expr(state, ctx, value) for value in expr.args.defaults])
    kwonly_items = []
    for arg, default in zip(expr.args.kwonlyargs, expr.args.kw_defaults):
        if default is None:
            continue
        kwonly_items.append((arg.arg, lower_expr(state, ctx, default)))
    temp = fresh_temp(state)
    ctx.builder.emit(attach_meta(state, MakeFunction(dst=temp, code=child_label, defaults=default_values, kwdefaults=COWList(kwonly_items)), expr))
    return temp

def snapshot_comprehension_target_names(state, ctx, generators, owner):
    # Preserve any outer names shadowed by comprehension targets so the enclosing scope sees no leak.
    saved = []
    names = sorted({name for generator in generators for name in target_names(state, generator.target)})
    for name in names:
        scope = scope_for_store(state, ctx, name)
        present_name = fresh_synthetic_local(state, "saved_present")
        value_name = fresh_synthetic_local(state, "saved_value")
        missing_label = ctx.builder.new_label()
        after_label = ctx.builder.new_label()
        ctx.builder.emit(attach_meta(state, PushTry(except_label=missing_label), owner))
        loaded = fresh_temp(state)
        ctx.builder.emit(attach_meta(state, LoadName(dst=loaded, scope=scope, name=name), owner))
        ctx.builder.emit(attach_meta(state, PopTry(), owner))
        present_true = const_value(state, ctx, True, owner)
        ctx.builder.emit(attach_meta(state, StoreName(src=present_true, scope=Scope.LOCAL, name=present_name), owner))
        ctx.builder.emit(attach_meta(state, StoreName(src=loaded, scope=Scope.LOCAL, name=value_name), owner))
        ctx.builder.terminate(attach_meta(state, Jump(target=after_label), owner))
        ctx.builder.start_block(missing_label)
        ctx.builder.emit(attach_meta(state, ClearException(), owner))
        present_false = const_value(state, ctx, False, owner)
        missing_value = const_value(state, ctx, None, owner)
        ctx.builder.emit(attach_meta(state, StoreName(src=present_false, scope=Scope.LOCAL, name=present_name), owner))
        ctx.builder.emit(attach_meta(state, StoreName(src=missing_value, scope=Scope.LOCAL, name=value_name), owner))
        ctx.builder.terminate(attach_meta(state, Jump(target=after_label), owner))
        ctx.builder.start_block(after_label)
        saved.append((scope, name, present_name, value_name))
    return saved


def restore_comprehension_target_names(state, ctx, saved_names, owner):
    # Restore or delete the saved outer bindings after the comprehension region finishes.
    for scope, name, present_name, value_name in saved_names:
        present_temp = fresh_temp(state)
        value_temp = fresh_temp(state)
        ctx.builder.emit(attach_meta(state, LoadName(dst=present_temp, scope=Scope.LOCAL, name=present_name), owner))
        ctx.builder.emit(attach_meta(state, LoadName(dst=value_temp, scope=Scope.LOCAL, name=value_name), owner))
        restore_label = ctx.builder.new_label()
        delete_label = ctx.builder.new_label()
        next_label = ctx.builder.new_label()
        delete_missing_label = ctx.builder.new_label()
        ctx.builder.terminate(attach_meta(state, Branch(cond=present_temp, true_label=restore_label, false_label=delete_label), owner))
        ctx.builder.start_block(restore_label)
        ctx.builder.emit(attach_meta(state, StoreName(src=value_temp, scope=scope, name=name), owner))
        ctx.builder.terminate(attach_meta(state, Jump(target=next_label), owner))
        ctx.builder.start_block(delete_label)
        ctx.builder.emit(attach_meta(state, PushTry(except_label=delete_missing_label), owner))
        ctx.builder.emit(attach_meta(state, DeleteName(scope=scope, name=name), owner))
        ctx.builder.emit(attach_meta(state, PopTry(), owner))
        ctx.builder.terminate(attach_meta(state, Jump(target=next_label), owner))
        ctx.builder.start_block(delete_missing_label)
        ctx.builder.emit(attach_meta(state, ClearException(), owner))
        ctx.builder.terminate(attach_meta(state, Jump(target=next_label), owner))
        ctx.builder.start_block(next_label)
        ctx.builder.emit(attach_meta(state, DeleteName(scope=Scope.LOCAL, name=present_name), owner))
        ctx.builder.emit(attach_meta(state, DeleteName(scope=Scope.LOCAL, name=value_name), owner))


def target_names(state, target):
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, ast.Starred):
        return target_names(state, target.value)
    if isinstance(target, (ast.Tuple, ast.List)):
        names = []
        for child in target.elts:
            names.extend(target_names(state, child))
        return names
    return []


def lower_container_items(state, ctx, elts):
    items = []
    for elt in elts:
        if isinstance(elt, ast.Starred):
            items.append(UnpackedTemporaryValue(lower_expr(state, ctx, elt.value)))
        else:
            items.append(lower_expr(state, ctx, elt))
    return items


def lower_compare_expr(state, ctx, expr):
    if len(expr.ops) == 1 and len(expr.comparators) == 1:
        lhs = lower_expr(state, ctx, expr.left)
        rhs = lower_expr(state, ctx, expr.comparators[0])
        temp = fresh_temp(state)
        ctx.builder.emit(attach_meta(state, CompareOp(dst=temp, cmp=compare_op(state, expr.ops[0]), lhs=lhs, rhs=rhs), expr))
        return temp

    current_name = fresh_synthetic_local(state, "compare_current")
    result_name = fresh_synthetic_local(state, "compare_result")
    false_label = ctx.builder.new_label()
    end_label = ctx.builder.new_label()

    first_value = lower_expr(state, ctx, expr.left)
    ctx.builder.emit(attach_meta(state, StoreName(src=first_value, scope=Scope.LOCAL, name=current_name), expr.left))

    for index, (op_node, rhs_expr) in enumerate(zip(expr.ops, expr.comparators)):
        lhs = fresh_temp(state)
        ctx.builder.emit(attach_meta(state, LoadName(dst=lhs, scope=Scope.LOCAL, name=current_name), rhs_expr))
        rhs = lower_expr(state, ctx, rhs_expr)
        cmp_result = fresh_temp(state)
        ctx.builder.emit(attach_meta(state, CompareOp(dst=cmp_result, cmp=compare_op(state, op_node), lhs=lhs, rhs=rhs), rhs_expr))
        is_last = index == len(expr.ops) - 1
        if is_last:
            true_label = ctx.builder.new_label()
            ctx.builder.terminate(attach_meta(state, Branch(cond=cmp_result, true_label=true_label, false_label=false_label), rhs_expr))
            ctx.builder.start_block(true_label)
            true_value = const_value(state, ctx, True, rhs_expr)
            ctx.builder.emit(attach_meta(state, StoreName(src=true_value, scope=Scope.LOCAL, name=result_name), rhs_expr))
            ctx.builder.terminate(attach_meta(state, Jump(target=end_label), rhs_expr))
        else:
            next_label = ctx.builder.new_label()
            ctx.builder.emit(attach_meta(state, StoreName(src=rhs, scope=Scope.LOCAL, name=current_name), rhs_expr))
            ctx.builder.terminate(attach_meta(state, Branch(cond=cmp_result, true_label=next_label, false_label=false_label), rhs_expr))
            ctx.builder.start_block(next_label)

    ctx.builder.start_block(false_label)
    false_value = const_value(state, ctx, False, expr)
    ctx.builder.emit(attach_meta(state, StoreName(src=false_value, scope=Scope.LOCAL, name=result_name), expr))
    ctx.builder.terminate(attach_meta(state, Jump(target=end_label), expr))

    ctx.builder.start_block(end_label)
    result = fresh_temp(state)
    ctx.builder.emit(attach_meta(state, LoadName(dst=result, scope=Scope.LOCAL, name=result_name), expr))
    ctx.builder.emit(attach_meta(state, DeleteName(scope=Scope.LOCAL, name=current_name), expr))
    ctx.builder.emit(attach_meta(state, DeleteName(scope=Scope.LOCAL, name=result_name), expr))
    return result


def lower_expr(state: CompilerState, ctx: RegionContext, expr: ast.AST) -> TemporaryValue:
    """Lower one expression and return the IR value holding its result."""
    if isinstance(expr, ast.Constant):
        return const_value(state, ctx, expr.value, expr)
    if isinstance(expr, ast.NamedExpr):
        value = lower_expr(state, ctx, expr.value)
        if not isinstance(expr.target, ast.Name):
            raise UnsupportedFeature(expr.target, 'named-expression target %s is not implemented in AST lowering' % type(expr.target).__name__)
        scope = scope_for_store(state, ctx, expr.target.id)
        ctx.builder.emit(attach_meta(state, StoreName(src=value, scope=scope, name=expr.target.id), expr.target))
        return value
    if isinstance(expr, ast.Lambda):
        return lower_lambda(state, ctx, expr)
    if isinstance(expr, ast.Name):
        temp = fresh_temp(state)
        scope = scope_for_load(state, ctx, expr.id)
        ctx.builder.emit(attach_meta(state, LoadName(dst=temp, scope=scope, name=expr.id), expr))
        return temp
    if isinstance(expr, ast.Attribute):
        obj = lower_expr(state, ctx, expr.value)
        temp = fresh_temp(state)
        ctx.builder.emit(attach_meta(state, LoadAttr(dst=temp, obj=obj, attr_name=expr.attr), expr))
        return temp
    if isinstance(expr, ast.Subscript):
        obj = lower_expr(state, ctx, expr.value)
        key = lower_slice_expr(state, ctx, expr.slice)
        temp = fresh_temp(state)
        ctx.builder.emit(attach_meta(state, LoadItem(dst=temp, obj=obj, key=key), expr))
        return temp
    if isinstance(expr, ast.Call):
        callee = lower_expr(state, ctx, expr.func)
        args = []
        for arg in expr.args:
            if isinstance(arg, ast.Starred):
                args.append(UnpackedTemporaryValue(lower_expr(state, ctx, arg.value)))
            else:
                args.append(lower_expr(state, ctx, arg))
        kwargs = []
        for keyword in expr.keywords:
            kwargs.append((keyword.arg, lower_expr(state, ctx, keyword.value)))
        temp = fresh_temp(state)
        ctx.builder.emit(attach_meta(state, Call(dst=temp, callee=callee, args=COWList(args), kwargs=COWList(kwargs), flags=0), expr))
        return temp
    if isinstance(expr, ast.BoolOp):
        return lower_bool_op(state, ctx, expr)
    if isinstance(expr, ast.IfExp):
        return lower_ifexp(state, ctx, expr)
    if isinstance(expr, ast.Tuple):
        items = lower_container_items(state, ctx, expr.elts)
        return build_tuple(state, ctx, items, expr)
    if isinstance(expr, ast.List):
        items = lower_container_items(state, ctx, expr.elts)
        temp = fresh_temp(state)
        ctx.builder.emit(attach_meta(state, BuildList(dst=temp, items=COWList(items)), expr))
        return temp
    if isinstance(expr, ast.Dict):
        items = []
        for key, value in zip(expr.keys, expr.values):
            if key is None:
                items.append((None, lower_expr(state, ctx, value)))
            else:
                items.append((lower_expr(state, ctx, key), lower_expr(state, ctx, value)))
        temp = fresh_temp(state)
        ctx.builder.emit(attach_meta(state, BuildMap(dst=temp, items=COWList(items)), expr))
        return temp
    if isinstance(expr, ast.ListComp):
        return lower_list_comp(state, ctx, expr)
    if isinstance(expr, ast.JoinedStr):
        parts = [lower_expr(state, ctx, value) for value in expr.values]
        temp = fresh_temp(state)
        ctx.builder.emit(attach_meta(state, BuildString(dst=temp, parts=COWList(parts)), expr))
        return temp
    if isinstance(expr, ast.FormattedValue):
        value = lower_expr(state, ctx, expr.value)
        conversion = None
        if expr.conversion == ord('s'):
            conversion = 'str'
        elif expr.conversion == ord('r'):
            conversion = 'repr'
        elif expr.conversion == ord('a'):
            conversion = 'ascii'
        elif expr.conversion not in (-1, None):
            raise UnsupportedFeature(expr, 'formatted-value conversion %r is not implemented in AST lowering' % (expr.conversion,))
        spec = None if expr.format_spec is None else lower_expr(state, ctx, expr.format_spec)
        temp = fresh_temp(state)
        ctx.builder.emit(attach_meta(state, FormatValue(dst=temp, value=value, conversion=conversion, spec=spec), expr))
        return temp
    if isinstance(expr, ast.SetComp):
        return lower_set_comp(state, ctx, expr)
    if isinstance(expr, ast.DictComp):
        return lower_dict_comp(state, ctx, expr)
    if isinstance(expr, ast.GeneratorExp):
        return lower_generator_exp(state, ctx, expr)
    if isinstance(expr, ast.Set):
        items = lower_container_items(state, ctx, expr.elts)
        temp = fresh_temp(state)
        ctx.builder.emit(attach_meta(state, BuildSet(dst=temp, items=COWList(items)), expr))
        return temp
    if isinstance(expr, ast.BinOp):
        lhs = lower_expr(state, ctx, expr.left)
        rhs = lower_expr(state, ctx, expr.right)
        temp = fresh_temp(state)
        ctx.builder.emit(attach_meta(state, BinaryOp(dst=temp, op=binary_op(state, expr.op), lhs=lhs, rhs=rhs), expr))
        return temp
    if isinstance(expr, ast.UnaryOp):
        src = lower_expr(state, ctx, expr.operand)
        temp = fresh_temp(state)
        ctx.builder.emit(attach_meta(state, UnaryOp(dst=temp, op=unary_op(state, expr.op), src=src), expr))
        return temp
    if isinstance(expr, ast.Compare):
        return lower_compare_expr(state, ctx, expr)
    if isinstance(expr, ast.Yield):
        value = lower_optional_expr(state, ctx, expr.value, expr)
        temp = fresh_temp(state)
        ctx.builder.emit(attach_meta(state, YieldValue(dst=temp, value=value), expr))
        return temp
    if isinstance(expr, ast.YieldFrom):
        value = lower_expr(state, ctx, expr.value)
        temp = fresh_temp(state)
        ctx.builder.emit(attach_meta(state, YieldFrom(dst=temp, value=value), expr))
        return temp
    if isinstance(expr, ast.Await):
        value = lower_expr(state, ctx, expr.value)
        awaitable = fresh_temp(state)
        ctx.builder.emit(attach_meta(state, GetAwaitable(dst=awaitable, value=value, where=0), expr))
        temp = fresh_temp(state)
        ctx.builder.emit(attach_meta(state, AwaitValue(dst=temp, value=awaitable), expr))
        return temp
    raise UnsupportedFeature(expr, 'expression %s is not implemented in AST lowering' % type(expr).__name__)

def current_exception_value(state, ctx, node):
    temp = fresh_temp(state)
    ctx.builder.emit(attach_meta(state, CurrentException(dst=temp), node))
    return temp

def lower_optional_expr(state, ctx, expr, owner):
    if expr is None:
        return const_value(state, ctx, None, owner)
    return lower_expr(state, ctx, expr)

def lower_slice_expr(state, ctx, slice_node):
    if isinstance(slice_node, ast.Slice):
        start = lower_optional_expr(state, ctx, slice_node.lower, slice_node)
        stop = lower_optional_expr(state, ctx, slice_node.upper, slice_node)
        step = None if slice_node.step is None else lower_expr(state, ctx, slice_node.step)
        temp = fresh_temp(state)
        ctx.builder.emit(attach_meta(state, BuildSlice(dst=temp, start=start, stop=stop, step=step), slice_node))
        return temp
    return lower_expr(state, ctx, slice_node)

def assign_target(state, ctx, target, value):
    if isinstance(target, ast.Name):
        scope = scope_for_store(state, ctx, target.id)
        ctx.builder.emit(attach_meta(state, StoreName(src=value, scope=scope, name=target.id), target))
        return
    if isinstance(target, ast.Attribute):
        obj = lower_expr(state, ctx, target.value)
        ctx.builder.emit(attach_meta(state, StoreAttr(obj=obj, attr_name=target.attr, value=value), target))
        return
    if isinstance(target, ast.Subscript):
        obj = lower_expr(state, ctx, target.value)
        key = lower_slice_expr(state, ctx, target.slice)
        ctx.builder.emit(attach_meta(state, StoreItem(obj=obj, key=key, value=value), target))
        return
    if isinstance(target, (ast.Tuple, ast.List)):
        assign_sequence_target(state, ctx, target, value)
        return
    raise UnsupportedFeature(target, 'assignment target %s is not implemented in AST lowering' % type(target).__name__)

def assign_sequence_target(state, ctx, target, value):
    starred = [index for index, elt in enumerate(target.elts) if isinstance(elt, ast.Starred)]
    if len(starred) > 1:
        raise UnsupportedFeature(target, 'multiple starred assignment targets are not implemented')
    if not starred:
        dsts = [fresh_temp(state) for _ in target.elts]
        ctx.builder.emit(attach_meta(state, Unpack(src=value, dsts=COWList(dsts)), target))
        for child, dst in zip(target.elts, dsts):
            assign_target(state, ctx, child, dst)
        return
    star_index = starred[0]
    before_dsts = [fresh_temp(state) for _ in target.elts[:star_index]]
    star_dst = fresh_temp(state)
    after_dsts = [fresh_temp(state) for _ in target.elts[star_index + 1:]]
    ctx.builder.emit(attach_meta(state, Unpack(src=value, dsts=COWList(before_dsts + [star_dst] + after_dsts), star_index=star_index), target))
    for child, dst in zip(target.elts[:star_index], before_dsts):
        assign_target(state, ctx, child, dst)
    assign_target(state, ctx, target.elts[star_index].value, star_dst)
    for child, dst in zip(target.elts[star_index + 1:], after_dsts):
        assign_target(state, ctx, child, dst)

def delete_target(state, ctx, target):
    if isinstance(target, ast.Name):
        scope = scope_for_store(state, ctx, target.id)
        ctx.builder.emit(attach_meta(state, DeleteName(scope=scope, name=target.id), target))
        return
    if isinstance(target, ast.Attribute):
        obj = lower_expr(state, ctx, target.value)
        ctx.builder.emit(attach_meta(state, DeleteAttr(obj=obj, attr_name=target.attr), target))
        return
    if isinstance(target, ast.Subscript):
        obj = lower_expr(state, ctx, target.value)
        key = lower_slice_expr(state, ctx, target.slice)
        ctx.builder.emit(attach_meta(state, DeleteItem(obj=obj, key=key), target))
        return
    if isinstance(target, (ast.Tuple, ast.List)):
        for child in target.elts:
            delete_target(state, ctx, child)
        return
    raise UnsupportedFeature(target, 'delete target %s is not implemented in AST lowering' % type(target).__name__)

def take_child_region_inputs(state, ctx, table_type: ChildRegionType, symtable_name: str, code_name: str, owner):
    """Find the next nested symbol-table child and code object for a region.

    AST lowering needs both pieces of metadata for nested executable regions such as
    functions, classes, and comprehension/genexpr bodies:

    - the `symtable` child provides scope information
    - the compiled child code object provides flags, locals, cells, and freevars

    Matching is done in source order using `ctx.next_child_table` and
    `ctx.next_child_code` rather than by global lookup. Callers provide the exact
    symtable and code-object names to match so this helper does not need to rewrite
    names like `<genexpr>`.
    """
    table = None
    for index in range(ctx.next_child_table, len(ctx.child_tables)):
        candidate = ctx.child_tables[index]
        if candidate.get_type() == table_type.value and candidate.get_name() == symtable_name:
            table = candidate
            ctx.next_child_table = index + 1
            break
    if table is None:
        raise UnsupportedFeature(owner, 'missing nested symbol-table child for region')

    code_obj = None
    for index in range(ctx.next_child_code, len(ctx.child_codes)):
        candidate = ctx.child_codes[index]
        if candidate.co_name == code_name:
            code_obj = candidate
            ctx.next_child_code = index + 1
            break
    if code_obj is None:
        raise UnsupportedFeature(owner, 'missing nested code object for region')
    return (table, code_obj)

def child_region_name(state, base_name):
    counts = state.synthetic_region_name_stacks[-1]
    count = counts.get(base_name, 0) + 1
    counts[base_name] = count
    if count == 1:
        return base_name
    return '%s#%d' % (base_name, count)

def child_name_path(state, parent_ctx, child_name, for_class):
    parent_path = [item for item in parent_ctx.name_path if item != '<module>']
    if not parent_path:
        return COWList([child_name])
    if for_class or parent_ctx.is_class:
        return COWList(parent_path + [child_name])
    return COWList(parent_path + ['<locals>', child_name])

def scope_for_load(state, ctx, name):
    return scope_for_name(state, ctx, name)

def scope_for_store(state, ctx, name):
    return scope_for_name(state, ctx, name)

def scope_for_name(state, ctx, name):
    # Reconstruct Python's local/global/name/cell addressing mode for this symbol.
    if ctx.is_class:
        symbol = lookup_symbol(state, ctx.table, name)
        if symbol is not None and symbol.is_declared_global():
            return Scope.GLOBAL
        return Scope.NAME
    if ctx.name == '<module>':
        symbol = lookup_symbol(state, ctx.table, name)
        if symbol is not None and symbol.is_declared_global():
            return Scope.GLOBAL
        return Scope.NAME
    symbol = lookup_symbol(state, ctx.table, name)
    if symbol is not None and symbol.is_declared_global():
        return Scope.GLOBAL
    if symbol is not None and symbol.is_nonlocal():
        return Scope.CELL
    if name in ctx.code_obj.co_freevars or name in ctx.code_obj.co_cellvars:
        return Scope.CELL
    if name in ctx.code_obj.co_varnames:
        return Scope.LOCAL
    return Scope.GLOBAL

def lookup_symbol(state, table, name):
    if name not in table.get_identifiers():
        return None
    return table.lookup(name)


def region_variadic_names(node):
    if not hasattr(node, 'args'):
        return (None, None)
    vararg = None if node.args.vararg is None else node.args.vararg.arg
    kwarg = None if node.args.kwarg is None else node.args.kwarg.arg
    return (vararg, kwarg)


def ensure_simple_arguments(state, node):
    return None

def const_value(state, ctx, value, node):
    temp = fresh_temp(state)
    ctx.builder.emit(attach_meta(state, Const(dst=temp, value=value), node))
    return temp


def builtin_const_value(state, ctx, value, node):
    return const_value(state, ctx, value, node)


def normal_call_args(args=()):
    return COWList(args)


def normal_call_kwargs(kwargs=()):
    return COWList(kwargs)


def build_tuple(state, ctx, items, node):
    temp = fresh_temp(state)
    ctx.builder.emit(attach_meta(state, BuildTuple(dst=temp, items=COWList(items)), node))
    return temp

def emit_return_none(state, builder, node):
    temp = fresh_temp(state)
    builder.emit(attach_meta(state, Const(dst=temp, value=None), node))
    builder.emit(attach_meta(state, Return(value=temp), node))

def attach_meta(state: CompilerState, instruction: Any, node: ast.AST) -> Any:
    # Keep source spans as optional metadata so the executable IR stays simple.
    span = SourceSpan(lineno=getattr(node, 'lineno', None), end_lineno=getattr(node, 'end_lineno', None), col_offset=getattr(node, 'col_offset', None), end_col_offset=getattr(node, 'end_col_offset', None))
    return attrs.evolve(instruction, span=span)

def binary_op(state, op):
    mapping = {ast.Add: '+', ast.Sub: '-', ast.Mult: '*', ast.Div: '/', ast.FloorDiv: '//', ast.Mod: '%', ast.Pow: '**', ast.LShift: '<<', ast.RShift: '>>', ast.BitAnd: '&', ast.BitOr: '|', ast.BitXor: '^', ast.MatMult: '@'}
    for cls, name in mapping.items():
        if isinstance(op, cls):
            return name
    raise UnsupportedFeature(op, 'binary operator %s is not implemented in AST lowering' % type(op).__name__)

def unary_op(state, op):
    mapping = {ast.UAdd: '+', ast.USub: '-', ast.Not: 'not', ast.Invert: '~'}
    for cls, name in mapping.items():
        if isinstance(op, cls):
            return name
    raise UnsupportedFeature(op, 'unary operator %s is not implemented in AST lowering' % type(op).__name__)

def compare_op(state, op):
    mapping = {ast.Lt: '<', ast.LtE: '<=', ast.Eq: '==', ast.NotEq: '!=', ast.Gt: '>', ast.GtE: '>=', ast.Is: 'is', ast.IsNot: 'is not', ast.In: 'in', ast.NotIn: 'not in'}
    for cls, name in mapping.items():
        if isinstance(op, cls):
            return name
    raise UnsupportedFeature(op, 'compare operator %s is not implemented in AST lowering' % type(op).__name__)



# ---------------------------------------------------------------------------
# IR interpreter
# ---------------------------------------------------------------------------


"""Execute the project IR.

The interpreter is intentionally semantic rather than optimized. It executes Region blocks,
maintains explicit temp/local/cell state, and implements the extra control-flow machinery used
by the AST frontend such as try/finally cleanup and structured escape targets.
"""

# Sentinel distinct from user-visible None.
_UNSET = object()


class IRHook(object):
    # Optional observer interface used by tests and debugging tools.
    def on_enter_frame(self, frame):
        pass

    def on_exit_frame(self, frame, result=None, exception=None):
        pass

    def before_instruction(self, frame, instr):
        pass

    def after_instruction(self, frame, instr, result=_UNSET):
        pass

    def on_exception(self, frame, instr, exception):
        pass


# Mutable box used to model Python closure cells.
@attrs.define
class Cell:
    value: Any = _UNSET


class VerboseTraceHook(IRHook):
    # Simple stderr trace of executed IR instructions.
    def before_instruction(self, frame, instr):
        location = "%s L%s:%s" % (frame.function.__qualname__, frame.block_label.index, frame.instr_index)
        print("%s  %s" % (location, render_instruction(instr, indent="")), file=sys.stderr)


@attrs.define
class Frame:
    # Runtime state for one executing region. Besides locals/cells/temps it also tracks
    # pending control transfers that must survive through finally blocks.
    interpreter: "IRInterpreter"
    function: "IRFunction"
    function_ir: Region
    globals: Dict[str, Any]
    locals: Dict[str, Any]
    cells: Dict[str, Cell]
    temps: Dict[TemporaryValue, Any] = attrs.field(factory=dict)
    block_label: Optional[BasicBlockLabel] = None
    instr_index: int = 0
    finished: bool = False
    return_value: Any = None
    current_exception: Optional[BaseException] = None
    exc_stack: list = attrs.field(factory=list)
    pending_send_value: Any = _UNSET
    try_stack: list = attrs.field(factory=list)
    pending_return_value: Any = _UNSET
    pending_jump_label: Any = _UNSET


class IRFunction(object):
    # Runtime callable wrapper around one Region.
    def __init__(self, interpreter, region_ir, globals_dict, closure_cells=None, qualname=None, preloaded_locals=None):
        self.interpreter = interpreter
        self.function_ir = region_ir
        self.region_ir = region_ir
        self.globals = globals_dict
        self.closure_cells = dict(closure_cells or {})
        self.preloaded_locals = dict(preloaded_locals or {})
        self.defaults = None
        self.kwdefaults = None
        self.annotations = {}
        self.__name__ = region_ir.name.split("#", 1)[0]
        self.__qualname__ = qualname or self.__name__
        self.__defaults__ = self.defaults
        self.__kwdefaults__ = self.kwdefaults
        self.__annotations__ = self.annotations
        self.__globals__ = globals_dict

    def __repr__(self):
        return "<IRFunction %s>" % (self.__qualname__,)

    def __call__(self, *args, **kwargs):
        return self.interpreter.call_function(self, args, kwargs)

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return BoundIRMethod(self, obj)


class BoundIRMethod(object):
    # Small descriptor wrapper so IR-defined functions behave like Python methods.
    def __init__(self, function, instance):
        self.function = function
        self.instance = instance
        self.__name__ = function.__name__
        self.__qualname__ = function.__qualname__

    def __call__(self, *args, **kwargs):
        return self.function(self.instance, *args, **kwargs)


class IRGenerator(object):
    # Adapter exposing generator protocol on top of a suspended IR frame.
    def __init__(self, interpreter, frame):
        self.interpreter = interpreter
        self.frame = frame
        self.closed = False

    def __iter__(self):
        return self

    def __next__(self):
        return self.send(None)

    def send(self, value):
        if self.closed:
            raise StopIteration
        event = self.interpreter.resume_frame(self.frame, send_value=value)
        kind = event[0]
        if kind == "yield":
            return event[1]
        if kind == "return":
            self.closed = True
            raise StopIteration(event[1])
        raise RuntimeError("unexpected generator event %r" % (event,))


class IRCoroutine(collections.abc.Coroutine):
    # Adapter exposing coroutine protocol on top of a suspended IR frame.
    def __init__(self, interpreter, frame):
        self.interpreter = interpreter
        self.frame = frame
        self.done = False

    def __await__(self):
        return self

    def __iter__(self):
        return self

    def __next__(self):
        return self.send(None)

    def send(self, value):
        if self.done:
            raise StopIteration(None)
        kind, result = self.interpreter.resume_frame(self.frame, send_value=value)
        if kind == "yield":
            return result
        if kind == "return":
            self.done = True
            raise StopIteration(result)
        raise RuntimeError("unexpected coroutine event %r" % ((kind, result),))

    def throw(self, typ, val=None, tb=None):
        self.done = True
        if val is None:
            if isinstance(typ, BaseException):
                raise typ
            raise typ()
        raise val

    def close(self):
        self.done = True


class IRAsyncGenerator(object):
    # Adapter exposing async-generator protocol on top of a suspended IR frame.
    def __init__(self, interpreter, frame):
        self.interpreter = interpreter
        self.frame = frame
        self.closed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.closed:
            raise StopAsyncIteration
        kind, value = self.interpreter.resume_frame(self.frame, send_value=None)
        if kind == "yield":
            return value
        if kind == "return":
            self.closed = True
            raise StopAsyncIteration
        raise RuntimeError("unexpected async generator event %r" % ((kind, value),))


class IRInterpreter(object):
    # Small tree interpreter for nested Region trees.
    def __init__(self, module_ir, hooks=(), module_name="__main__", module_path=None, search_path=None):
        self.module_ir = module_ir
        self.hooks = tuple(hooks)
        self.module_name = module_name
        self.module_path = None if module_path is None else os.path.abspath(module_path)
        self.search_path = self.normalize_search_path(search_path, self.module_path)
        self.module_ir_cache = {}
        self.last_completed_frame = None

    def normalize_search_path(self, search_path, module_path):
        raw_entries = []
        if module_path is not None:
            raw_entries.append(os.path.dirname(module_path))
        if search_path is None:
            raw_entries.extend(sys.path)
        else:
            raw_entries.extend(search_path)

        normalized_entries = []
        seen = set()
        for entry in raw_entries:
            normalized = os.path.abspath(entry or os.getcwd())
            if normalized in seen:
                continue
            seen.add(normalized)
            normalized_entries.append(normalized)
        return tuple(normalized_entries)

    def run_module_region(self, module_ir, globals_dict, locals_dict=None, qualname="<module>"):
        if locals_dict is None:
            locals_dict = globals_dict
        module_function = IRFunction(self, module_ir, globals_dict, qualname=qualname)
        frame = Frame(
            interpreter=self,
            function=module_function,
            function_ir=module_ir,
            globals=globals_dict,
            locals=locals_dict,
            cells={},
            block_label=module_ir.entry_label,
            instr_index=0,
        )
        self.run_to_completion(frame)
        return locals_dict

    def exec(self, globals=None, locals=None):
        # Execute the top-level module region in a fresh frame.
        if globals is None:
            globals = {}
        if locals is None:
            locals = globals
        if "__builtins__" not in globals:
            globals["__builtins__"] = builtins.__dict__

        globals.setdefault("__name__", self.module_name)
        module_path = self.module_path
        if module_path is not None:
            globals.setdefault("__file__", module_path)
        # Keep exec()-style behavior here: do not synthesize import-module metadata
        # such as __package__ or __path__ when executing an arbitrary namespace.
        globals["__build_class__"] = self.build_class
        return self.run_module_region(self.module_ir, globals, locals)

    def resolve_absolute_import_name(self, frame, module_name, level):
        module_name = "" if module_name is None else module_name
        if level == 0:
            return module_name

        package_name = frame.globals.get("__package__")
        if package_name is None:
            current_name = frame.globals.get("__name__", "")
            if "__path__" in frame.globals:
                package_name = current_name
            else:
                package_name = current_name.rpartition(".")[0]
        if not package_name:
            raise ImportError("attempted relative import with no known parent package")
        return importlib.util.resolve_name("." * level + module_name, package_name)

    def bind_submodule(self, fullname, module):
        parent_name, _, child_name = fullname.rpartition(".")
        if not parent_name:
            return
        parent_module = sys.modules.get(parent_name)
        if parent_module is not None:
            setattr(parent_module, child_name, module)

    def find_module_spec(self, fullname):
        parent_name, _, _ = fullname.rpartition(".")
        search_path = list(self.search_path)
        if parent_name:
            parent_module = self.import_absolute_module(parent_name)
            search_path = getattr(parent_module, "__path__", None)
            if search_path is None:
                raise ModuleNotFoundError("No module named %s; %s is not a package" % (fullname, parent_name), name=fullname)
        return importlib.machinery.PathFinder.find_spec(fullname, search_path)

    def is_ir_source_spec(self, spec):
        if spec is None or spec.origin in (None, "built-in", "frozen"):
            return False
        return isinstance(spec.loader, importlib.machinery.SourceFileLoader) and spec.origin.endswith(".py")

    def load_module_ir(self, path):
        path = os.path.abspath(path)
        cached = self.module_ir_cache.get(path)
        if cached is not None:
            return cached
        module_ir = compile_file(new_compiler_state(), path)
        self.module_ir_cache[path] = module_ir
        return module_ir

    def load_ir_module_from_spec(self, fullname, spec):
        existing = sys.modules.get(fullname)
        if existing is not None:
            return existing

        module = importlib.util.module_from_spec(spec)
        if getattr(module, "__file__", None) is not None:
            module.__file__ = os.path.abspath(module.__file__)
        if getattr(module, "__path__", None) is not None:
            module.__path__ = [os.path.abspath(entry) for entry in module.__path__]
        sys.modules[fullname] = module
        self.bind_submodule(fullname, module)

        try:
            module.__dict__.setdefault("__builtins__", builtins.__dict__)
            module.__dict__["__build_class__"] = self.build_class
            module_ir = self.load_module_ir(spec.origin)
            self.run_module_region(module_ir, module.__dict__)
        except BaseException:
            if sys.modules.get(fullname) is module:
                del sys.modules[fullname]
            raise
        return module

    def load_python_module_from_spec(self, fullname, spec):
        existing = sys.modules.get(fullname)
        if existing is not None:
            return existing

        module = importlib.util.module_from_spec(spec)
        if getattr(module, "__file__", None) is not None:
            module.__file__ = os.path.abspath(module.__file__)
        if getattr(module, "__path__", None) is not None:
            module.__path__ = [os.path.abspath(entry) for entry in module.__path__]
        sys.modules[fullname] = module
        self.bind_submodule(fullname, module)

        try:
            if spec.loader is not None:
                spec.loader.exec_module(module)
        except BaseException:
            if sys.modules.get(fullname) is module:
                del sys.modules[fullname]
            raise
        return module

    def import_absolute_module(self, fullname):
        existing = sys.modules.get(fullname)
        if existing is not None:
            return existing

        spec = self.find_module_spec(fullname)
        if spec is None:
            return importlib.import_module(fullname)
        if self.is_ir_source_spec(spec):
            return self.load_ir_module_from_spec(fullname, spec)
        return self.load_python_module_from_spec(fullname, spec)

    def ensure_fromlist(self, module, fromlist):
        if not fromlist or "*" in fromlist:
            return
        package_path = getattr(module, "__path__", None)
        if package_path is None:
            return
        for name in fromlist:
            if hasattr(module, name):
                continue
            fullname = "%s.%s" % (module.__name__, name)
            try:
                self.import_absolute_module(fullname)
            except ModuleNotFoundError:
                continue

    def import_module(self, frame, module_name, fromlist, level):
        absolute_name = self.resolve_absolute_import_name(frame, module_name, level)
        module = self.import_absolute_module(absolute_name)
        self.ensure_fromlist(module, fromlist)
        if fromlist:
            return module
        top_level_name = absolute_name.split(".", 1)[0]
        return sys.modules.get(top_level_name, module)

    def build_class(self, body_function, name, *bases, **kwargs):
        # Execute the lowered class body like a function, then hand its namespace to the metaclass.
        metaclass = kwargs.pop("metaclass", type)
        namespace = {}
        body_function.__qualname__ = name
        body_function.preloaded_locals = {
            "__module__": body_function.__globals__.get("__name__", "__main__"),
            "__qualname__": name,
        }
        body_function.__call__()
        class_frame = self.last_completed_frame
        namespace.update(class_frame.locals)
        return metaclass(name, bases or (object,), namespace, **kwargs)

    def call_function(self, function, args, kwargs):
        # Pick the runtime wrapper that matches the code object's generator/coroutine flags.
        flags = function.region_ir.flags
        frame = self.make_frame(function, args, kwargs)

        if flags & inspect.CO_ASYNC_GENERATOR:
            return IRAsyncGenerator(self, frame)
        if flags & inspect.CO_COROUTINE:
            return IRCoroutine(self, frame)
        if flags & inspect.CO_GENERATOR:
            return IRGenerator(self, frame)
        return self.run_to_completion(frame)

    def make_frame(self, function, args, kwargs):
        # Materialize locals and closure cells for one function invocation.
        locals_dict = self.bind_arguments(function, args, kwargs)
        locals_dict.update(function.preloaded_locals)
        cells = dict(function.closure_cells)
        for name in function.region_ir.cells:
            if name not in cells:
                cells[name] = Cell(locals_dict.get(name, _UNSET))
        return Frame(
            interpreter=self,
            function=function,
            function_ir=function.function_ir,
            globals=function.globals,
            locals=locals_dict,
            cells=cells,
            block_label=function.function_ir.entry_label,
            instr_index=0,
        )

    def bind_arguments(self, function, args, kwargs):
        # Bind Python arguments using the structural signature recorded on the Region.
        region = function.region_ir
        positional = region.argcount
        posonly = region.posonlyargcount
        kwonly = region.kwonlyargcount
        names = list(region.locals[: positional + kwonly])
        bound = {}
        kwargs = dict(kwargs)

        positional_names = names[:positional]
        posonly_names = positional_names[:posonly]
        kwonly_names = names[positional: positional + kwonly]

        if len(args) > len(positional_names) and region.vararg_name is None:
            raise TypeError("too many positional arguments for %s" % (function.__qualname__,))

        consumed_positional = positional_names[: min(len(args), len(positional_names))]
        for name, value in zip(consumed_positional, args):
            bound[name] = value

        posonly_keyword_names = [name for name in posonly_names if name in kwargs]
        if posonly_keyword_names:
            raise TypeError("positional-only arguments passed by keyword for %s: %s" % (function.__qualname__, sorted(posonly_keyword_names)))

        duplicate_names = [name for name in consumed_positional if name in kwargs]
        if duplicate_names:
            raise TypeError("multiple values for arguments for %s: %s" % (function.__qualname__, sorted(duplicate_names)))

        for name in positional_names[len(consumed_positional):]:
            if name in kwargs:
                bound[name] = kwargs.pop(name)

        if function.defaults:
            default_names = positional_names[-len(function.defaults):]
            for name, value in zip(default_names, function.defaults):
                bound.setdefault(name, value)

        for name in positional_names:
            if name not in bound:
                raise TypeError("missing argument %r for %s" % (name, function.__qualname__))

        extra_args = args[len(positional_names):]
        if region.vararg_name is not None:
            bound[region.vararg_name] = tuple(extra_args)

        for name in kwonly_names:
            if name in kwargs:
                bound[name] = kwargs.pop(name)
            elif function.kwdefaults and name in function.kwdefaults:
                bound[name] = function.kwdefaults[name]
            else:
                raise TypeError("missing keyword-only argument %r for %s" % (name, function.__qualname__))

        if region.kwarg_name is not None:
            bound[region.kwarg_name] = dict(kwargs)
        elif kwargs:
            raise TypeError("unexpected keyword arguments for %s: %s" % (function.__qualname__, sorted(kwargs)))

        return bound

    def run_to_completion(self, frame):
        # Fully execute a frame that is not expected to yield.
        self.last_completed_frame = None
        event = self.resume_frame(frame, send_value=None)
        if event[0] != "return":
            raise RuntimeError("frame yielded unexpectedly: %r" % (event,))
        self.last_completed_frame = frame
        return event[1]

    def resume_frame(self, frame, send_value=None):
        # Drive one frame until it returns or yields.
        if frame.finished:
            return ("return", frame.return_value)
        if frame.block_label is None:
            frame.block_label = frame.function_ir.entry_label
        frame.pending_send_value = send_value

        for hook in self.hooks:
            hook.on_enter_frame(frame)

        try:
            while not frame.finished:
                block = self.get_block(frame.function_ir, frame.block_label)
                if frame.instr_index >= len(block.instructions):
                    next_label = self.fallthrough_label(frame.function_ir, frame.block_label)
                    if next_label is None:
                        frame.finished = True
                        break
                    frame.block_label = next_label
                    frame.instr_index = 0
                    continue

                instr = block.instructions[frame.instr_index]
                for hook in self.hooks:
                    hook.before_instruction(frame, instr)

                try:
                    event = self.execute_instruction(frame, instr)
                except BaseException as exc:
                    for hook in self.hooks:
                        hook.on_exception(frame, instr, exc)
                    handled = self.handle_exception(frame, exc)
                    if handled is None:
                        frame.current_exception = exc
                        raise
                    event = handled

                for hook in self.hooks:
                    hook.after_instruction(frame, instr, event if event is not None else _UNSET)

                if event is None:
                    frame.instr_index += 1
                    continue

                kind = event[0]
                if kind == "jump":
                    frame.block_label = event[1]
                    frame.instr_index = 0
                    continue
                if kind == "yield":
                    frame.instr_index += 1
                    return event
                if kind == "return":
                    frame.finished = True
                    frame.return_value = event[1]
                    return event
                raise RuntimeError("unknown execution event %r" % (event,))

            return ("return", frame.return_value)
        finally:
            for hook in self.hooks:
                hook.on_exit_frame(frame, result=frame.return_value)

    def get_block(self, function_ir, label):
        for block in function_ir.basic_blocks:
            if block.label == label:
                return block
        raise KeyError("unknown block label %r in %s" % (label, function_ir.name))

    def fallthrough_label(self, function_ir, label):
        basic_blocks = list(function_ir.basic_blocks)
        for index, block in enumerate(basic_blocks):
            if block.label == label and index + 1 < len(basic_blocks):
                return basic_blocks[index + 1].label
        return None

    def resolve_value(self, frame, value):
        if isinstance(value, TemporaryValue):
            return frame.temps[value]
        if isinstance(value, BasicBlockLabel):
            return value
        if isinstance(value, RegionLabel):
            return value
        return value

    def get_child_region(self, function_ir, label):
        for child_region in function_ir.child_regions:
            if child_region.label == label:
                return child_region
        raise KeyError("unknown child region label %r in %s" % (label, function_ir.name))

    def store_temp(self, frame, temp, value):
        frame.temps[temp] = value
        return value

    def execute_instruction(self, frame, instr):
        # Main interpreter dispatch for one IR instruction.
        if isinstance(instr, Const):
            return self.exec_value_instr(frame, instr, instr.value)

        if isinstance(instr, LoadName):
            value = self.load_var(frame, instr.scope, instr.name)
            return self.exec_value_instr(frame, instr, value)

        if isinstance(instr, StoreName):
            self.store_var(frame, instr.scope, instr.name, self.resolve_value(frame, instr.src))
            return None

        if isinstance(instr, DeleteName):
            self.delete_var(frame, instr.scope, instr.name)
            return None

        if isinstance(instr, UnaryOp):
            return self.exec_value_instr(frame, instr, self.apply_unary(instr.op, self.resolve_value(frame, instr.src)))

        if isinstance(instr, BinaryOp):
            lhs = self.resolve_value(frame, instr.lhs)
            rhs = self.resolve_value(frame, instr.rhs)
            return self.exec_value_instr(frame, instr, self.apply_binary(instr.op, lhs, rhs))

        if isinstance(instr, CompareOp):
            lhs = self.resolve_value(frame, instr.lhs)
            rhs = self.resolve_value(frame, instr.rhs)
            return self.exec_value_instr(frame, instr, self.apply_compare(instr.cmp, lhs, rhs))

        if isinstance(instr, LoadAttr):
            obj = self.resolve_value(frame, instr.obj)
            return self.exec_value_instr(frame, instr, getattr(obj, instr.attr_name))

        if isinstance(instr, StoreAttr):
            setattr(self.resolve_value(frame, instr.obj), instr.attr_name, self.resolve_value(frame, instr.value))
            return None

        if isinstance(instr, DeleteAttr):
            delattr(self.resolve_value(frame, instr.obj), instr.attr_name)
            return None

        if isinstance(instr, LoadItem):
            obj = self.resolve_value(frame, instr.obj)
            key = self.resolve_value(frame, instr.key)
            return self.exec_value_instr(frame, instr, obj[key])

        if isinstance(instr, StoreItem):
            obj = self.resolve_value(frame, instr.obj)
            key = self.resolve_value(frame, instr.key)
            value = self.resolve_value(frame, instr.value)
            obj[key] = value
            return None

        if isinstance(instr, DeleteItem):
            del self.resolve_value(frame, instr.obj)[self.resolve_value(frame, instr.key)]
            return None

        if isinstance(instr, BuildTuple):
            built = []
            for item in instr.items:
                if isinstance(item, UnpackedTemporaryValue):
                    built.extend(self.resolve_value(frame, item.value))
                else:
                    built.append(self.resolve_value(frame, item))
            return self.exec_value_instr(frame, instr, tuple(built))

        if isinstance(instr, BuildList):
            built = []
            for item in instr.items:
                if isinstance(item, UnpackedTemporaryValue):
                    built.extend(self.resolve_value(frame, item.value))
                else:
                    built.append(self.resolve_value(frame, item))
            return self.exec_value_instr(frame, instr, built)

        if isinstance(instr, BuildSet):
            built = set()
            for item in instr.items:
                if isinstance(item, UnpackedTemporaryValue):
                    built.update(self.resolve_value(frame, item.value))
                else:
                    built.add(self.resolve_value(frame, item))
            return self.exec_value_instr(frame, instr, built)

        if isinstance(instr, BuildMap):
            built = {}
            for key, value in instr.items:
                if key is None:
                    built.update(dict(self.resolve_value(frame, value)))
                else:
                    built[self.resolve_value(frame, key)] = self.resolve_value(frame, value)
            return self.exec_value_instr(frame, instr, built)

        if isinstance(instr, BuildSlice):
            return self.exec_value_instr(
                frame,
                instr,
                slice(
                    self.resolve_value(frame, instr.start),
                    self.resolve_value(frame, instr.stop),
                    None if instr.step is None else self.resolve_value(frame, instr.step),
                ),
            )

        if isinstance(instr, BuildString):
            return self.exec_value_instr(frame, instr, "".join(str(self.resolve_value(frame, part)) for part in instr.parts))

        if isinstance(instr, FormatValue):
            value = self.resolve_value(frame, instr.value)
            if instr.conversion == "repr":
                value = repr(value)
            elif instr.conversion == "ascii":
                value = ascii(value)
            else:
                value = str(value)
            if instr.spec is not None:
                value = format(value, self.resolve_value(frame, instr.spec))
            return self.exec_value_instr(frame, instr, value)

        if isinstance(instr, Unpack):
            values = list(self.resolve_value(frame, instr.src))
            if instr.star_index is None:
                if len(values) != len(instr.dsts):
                    raise ValueError("unpack mismatch")
                for dst, value in zip(instr.dsts, values):
                    frame.temps[dst] = value
                return None
            if instr.star_index < 0 or instr.star_index >= len(instr.dsts):
                raise ValueError("invalid unpack star index")
            before_count = instr.star_index
            after_count = len(instr.dsts) - before_count - 1
            if len(values) < before_count + after_count:
                raise ValueError("unpack mismatch")
            for dst, value in zip(instr.dsts[:before_count], values[:before_count]):
                frame.temps[dst] = value
            frame.temps[instr.dsts[instr.star_index]] = values[before_count: len(values) - after_count]
            for dst, value in zip(instr.dsts[before_count + 1 :], values[len(values) - after_count:]):
                frame.temps[dst] = value
            return None

        if isinstance(instr, Call):
            callee = self.resolve_value(frame, instr.callee)
            args = []
            for arg in instr.args:
                if isinstance(arg, UnpackedTemporaryValue):
                    args.extend(self.resolve_value(frame, arg.value))
                else:
                    args.append(self.resolve_value(frame, arg))
            kwargs = {}
            for name, value in instr.kwargs:
                resolved = self.resolve_value(frame, value)
                if name is None:
                    for key, item in dict(resolved).items():
                        if key in kwargs:
                            raise TypeError("multiple values for keyword argument %r" % (key,))
                        kwargs[key] = item
                else:
                    if name in kwargs:
                        raise TypeError("multiple values for keyword argument %r" % (name,))
                    kwargs[name] = resolved
            return self.exec_value_instr(frame, instr, callee(*args, **kwargs))

        if isinstance(instr, ImportName):
            module = self.import_module(frame, instr.module, list(instr.fromlist), instr.level)
            return self.exec_value_instr(frame, instr, module)

        if isinstance(instr, ImportFrom):
            module_obj = self.resolve_value(frame, instr.module_obj)
            return self.exec_value_instr(frame, instr, getattr(module_obj, instr.name))

        if isinstance(instr, ImportStar):
            frame.locals.update(vars(self.resolve_value(frame, instr.module_obj)))
            return None

        if isinstance(instr, MakeFunction):
            region = self.get_child_region(frame.function_ir, instr.code)
            closure = {}
            for name in region.freevars:
                if name in frame.cells:
                    closure[name] = frame.cells[name]
            child_name = region.name.split("#", 1)[0]
            parent_qualname = frame.function.__qualname__
            if frame.function_ir.name == "<module>":
                qualname = child_name
            elif frame.function_ir.is_class:
                qualname = "%s.%s" % (parent_qualname, child_name)
            else:
                qualname = "%s.<locals>.%s" % (parent_qualname, child_name)
            fn = IRFunction(self, region, frame.globals, closure_cells=closure, qualname=qualname)
            if instr.defaults:
                fn.defaults = tuple(self.resolve_value(frame, value) for value in instr.defaults)
                fn.__defaults__ = fn.defaults
            if instr.kwdefaults:
                fn.kwdefaults = {name: self.resolve_value(frame, value) for name, value in instr.kwdefaults}
                fn.__kwdefaults__ = fn.kwdefaults
            return self.exec_value_instr(frame, instr, fn)

        if isinstance(instr, BuildClass):
            body = self.resolve_value(frame, instr.body_func)
            name = self.resolve_value(frame, instr.name)
            bases = [self.resolve_value(frame, base) for base in instr.bases]
            keywords = {name: self.resolve_value(frame, value) for name, value in instr.keywords}
            return self.exec_value_instr(frame, instr, self.build_class(body, name, *bases, **keywords))

        if isinstance(instr, GetIter):
            return self.exec_value_instr(frame, instr, iter(self.resolve_value(frame, instr.iterable)))

        if isinstance(instr, ForIter):
            iterator = self.resolve_value(frame, instr.iter_obj)
            try:
                value = next(iterator)
            except StopIteration:
                return ("jump", instr.exit_label)
            frame.temps[instr.value_dst] = value
            return ("jump", instr.body_label)

        if isinstance(instr, GetAIter):
            return self.exec_value_instr(frame, instr, self.resolve_value(frame, instr.iterable).__aiter__())

        if isinstance(instr, GetANext):
            return self.exec_value_instr(frame, instr, self.resolve_value(frame, instr.aiter).__anext__())

        if isinstance(instr, GetAwaitable):
            value = self.resolve_value(frame, instr.value)
            return self.exec_value_instr(frame, instr, value if inspect.isawaitable(value) else value)

        if isinstance(instr, YieldValue):
            yielded = self.resolve_value(frame, instr.value)
            sent_value = None if frame.pending_send_value is _UNSET else frame.pending_send_value
            self.store_temp(frame, instr.dst, sent_value)
            frame.pending_send_value = _UNSET
            return ("yield", yielded)

        if isinstance(instr, YieldFrom):
            yielded = self.resolve_value(frame, instr.value)
            self.store_temp(frame, instr.dst, None)
            return ("yield", yielded)

        if isinstance(instr, AwaitValue):
            value = self.resolve_value(frame, instr.value)
            return self.exec_value_instr(frame, instr, self.await_sync(value))

        if isinstance(instr, CurrentException):
            return self.exec_value_instr(frame, instr, frame.current_exception)

        if isinstance(instr, Raise):
            exc = self.normalize_exception_for_raise(self.resolve_value(frame, instr.exc))
            if instr.cause is not None:
                cause = self.normalize_exception_for_raise(self.resolve_value(frame, instr.cause), allow_none=True)
                exc.__cause__ = cause
                exc.__suppress_context__ = True
            raise exc

        if isinstance(instr, Reraise):
            if frame.current_exception is None:
                raise RuntimeError("no current exception to reraise")
            raise frame.current_exception

        if isinstance(instr, CheckExcMatch):
            exc = self.resolve_value(frame, instr.exc)
            typ = self.resolve_value(frame, instr.typ)
            matched = isinstance(exc, typ) if isinstance(typ, type) else False
            return self.exec_value_instr(frame, instr, matched)

        if isinstance(instr, CheckEGMatch):
            return self.exec_value_instr(frame, instr, False)

        if isinstance(instr, PushTry):
            frame.try_stack.append({
                "except_label": instr.except_label,
                "finally_label": instr.finally_label,
            })
            return None

        if isinstance(instr, PopTry):
            if frame.try_stack:
                frame.try_stack.pop()
            return None

        if isinstance(instr, ClearException):
            frame.current_exception = None
            return None

        if isinstance(instr, EndFinally):
            return self.end_finally(frame)

        if isinstance(instr, Escape):
            return self.handle_escape(frame, instr.target)

        if isinstance(instr, Jump):
            return ("jump", instr.target)

        if isinstance(instr, Branch):
            cond = self.resolve_value(frame, instr.cond)
            return ("jump", instr.true_label if cond else instr.false_label)

        if isinstance(instr, Return):
            value = self.resolve_value(frame, instr.value)
            frame.current_exception = None
            return self.handle_return(frame, value)

        if isinstance(instr, MatchMapping):
            value = self.resolve_value(frame, instr.value)
            return self.exec_value_instr(frame, instr, isinstance(value, Mapping))

        if isinstance(instr, MatchSequence):
            value = self.resolve_value(frame, instr.value)
            return self.exec_value_instr(frame, instr, isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)))

        if isinstance(instr, MatchKeys):
            mapping = self.resolve_value(frame, instr.mapping)
            keys = self.resolve_value(frame, instr.keys)
            try:
                result = tuple(mapping[key] for key in keys)
            except Exception:
                result = None
            return self.exec_value_instr(frame, instr, result)

        if isinstance(instr, MatchClass):
            value = self.resolve_value(frame, instr.value)
            cls = self.resolve_value(frame, instr.cls)
            return self.exec_value_instr(frame, instr, isinstance(value, cls))

        raise NotImplementedError("unsupported IR instruction: %r" % (instr,))

    def exec_value_instr(self, frame, instr, value):
        self.store_temp(frame, instr.dst, value)
        return None

    def handle_exception(self, frame, exception):
        # Redirect an exception to the nearest active synthetic try target, if any.
        frame.current_exception = exception
        for index in range(len(frame.try_stack) - 1, -1, -1):
            entry = frame.try_stack[index]
            target = entry.get("except_label") or entry.get("finally_label")
            if target is None:
                continue
            del frame.try_stack[index:]
            return ("jump", target)
        return None

    def handle_return(self, frame, value):
        for index in range(len(frame.try_stack) - 1, -1, -1):
            entry = frame.try_stack[index]
            target = entry.get("finally_label")
            if target is None:
                continue
            frame.pending_return_value = value
            del frame.try_stack[index:]
            return ("jump", target)
        frame.pending_return_value = _UNSET
        frame.pending_jump_label = _UNSET
        return ("return", value)

    def handle_escape(self, frame, target_label):
        # `Escape` is used for structured `break` / `continue`. It behaves like a jump, except
        # that it must thread through pending finally blocks before reaching its destination.
        for index in range(len(frame.try_stack) - 1, -1, -1):
            entry = frame.try_stack[index]
            finally_label = entry.get("finally_label")
            if finally_label is None:
                continue
            frame.pending_jump_label = target_label
            del frame.try_stack[index:]
            return ("jump", finally_label)
        frame.pending_jump_label = _UNSET
        frame.pending_return_value = _UNSET
        return ("jump", target_label)

    def end_finally(self, frame):
        # Finish a finally region by resuming the highest-priority pending control effect:
        # exception, return, or structured jump.
        if frame.current_exception is not None:
            exc = frame.current_exception
            handled = self.handle_exception(frame, exc)
            if handled is not None:
                return handled
            raise exc
        if frame.pending_return_value is not _UNSET:
            value = frame.pending_return_value
            frame.pending_return_value = _UNSET
            return self.handle_return(frame, value)
        if frame.pending_jump_label is not _UNSET:
            label = frame.pending_jump_label
            frame.pending_jump_label = _UNSET
            return self.handle_escape(frame, label)
        return None

    def await_sync(self, value):
        if isinstance(value, IRCoroutine):
            iterator = value.__await__()
            send_value = None
            while True:
                try:
                    yielded = iterator.send(send_value)
                except StopIteration as stop:
                    return stop.value
                send_value = self.await_sync(yielded)
        if inspect.isawaitable(value):
            iterator = value.__await__()
            send_value = None
            while True:
                try:
                    yielded = iterator.send(send_value)
                except StopIteration as stop:
                    return stop.value
                if inspect.isawaitable(yielded) or isinstance(yielded, IRCoroutine):
                    send_value = self.await_sync(yielded)
                else:
                    send_value = yielded
        return value

    def load_var(self, frame, scope, name):
        # Variable lookup follows the explicit scope chosen during lowering.
        if scope == Scope.LOCAL:
            if name in frame.locals:
                return frame.locals[name]
            if name in frame.cells and frame.cells[name].value is not _UNSET:
                return frame.cells[name].value
            raise NameError(name)

        if scope == Scope.CELL:
            if name in frame.cells and frame.cells[name].value is not _UNSET:
                return frame.cells[name].value
            raise NameError(name)

        if scope == Scope.GLOBAL:
            if name in frame.globals:
                return frame.globals[name]
            return self.load_builtin(frame, name)

        if scope == Scope.NAME:
            if name in frame.locals:
                return frame.locals[name]
            if name in frame.globals:
                return frame.globals[name]
            return self.load_builtin(frame, name)

        raise NotImplementedError("unknown scope %r" % (scope,))

    def store_var(self, frame, scope, name, value):
        if scope == Scope.GLOBAL:
            frame.globals[name] = value
            return
        if scope == Scope.CELL:
            frame.cells.setdefault(name, Cell())
            frame.cells[name].value = value
            return
        frame.locals[name] = value

    def delete_var(self, frame, scope, name):
        if scope == Scope.GLOBAL:
            del frame.globals[name]
            return
        if scope == Scope.CELL:
            if name in frame.cells:
                frame.cells[name].value = _UNSET
                return
            raise NameError(name)
        del frame.locals[name]

    def load_builtin(self, frame, name):
        builtins_obj = frame.globals.get("__builtins__", builtins.__dict__)
        if isinstance(builtins_obj, dict):
            if name in builtins_obj:
                return builtins_obj[name]
        else:
            if hasattr(builtins_obj, name):
                return getattr(builtins_obj, name)
        raise NameError(name)

    def normalize_exception_for_raise(self, value, allow_none=False):
        # Match Python's rule that `raise` accepts an exception instance or exception class.
        if allow_none and value is None:
            return None
        if isinstance(value, BaseException):
            return value
        if isinstance(value, type) and issubclass(value, BaseException):
            return value()
        raise TypeError("exceptions must derive from BaseException")

    def apply_unary(self, op, value):
        if op == "+":
            return +value
        if op == "-":
            return -value
        if op == "not":
            return not value
        if op == "~":
            return ~value
        raise NotImplementedError("unsupported unary op %r" % (op,))

    def apply_binary(self, op, lhs, rhs):
        table = {
            "+": operator.add,
            "-": operator.sub,
            "*": operator.mul,
            "/": operator.truediv,
            "//": operator.floordiv,
            "%": operator.mod,
            "**": operator.pow,
            "<<": operator.lshift,
            ">>": operator.rshift,
            "&": operator.and_,
            "|": operator.or_,
            "^": operator.xor,
            "@": operator.matmul,
        }
        if op in table:
            return table[op](lhs, rhs)
        raise NotImplementedError("unsupported binary op %r" % (op,))

    def apply_compare(self, cmp, lhs, rhs):
        table = {
            "<": operator.lt,
            "<=": operator.le,
            "==": operator.eq,
            "!=": operator.ne,
            ">": operator.gt,
            ">=": operator.ge,
            "is": operator.is_,
            "is not": operator.is_not,
            "in": lambda a, b: a in b,
            "not in": lambda a, b: a not in b,
        }
        if cmp not in table:
            raise NotImplementedError("unsupported compare op %r" % (cmp,))
        return table[cmp](lhs, rhs)

