# Copyright (c) 2026 Jifeng Wu
# Licensed under the Apache-2.0 License. See LICENSE file in the project root for full license information.
"""AST-to-IR lowering compiler.

Converts Python AST nodes into the pyssa Region IR.  The compiler is
structured as explicit lowering functions with explicit parameters
rather than a large class-based visitor.
"""

import ast
import builtins
import symtable
import sys
import types
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import attrs
from cowlist import COWList

from .ir import (
    BasicBlockLabel,
    RegionLabel,
    Scope,
    SourceSpan,
    SyntheticLocal,
    SyntheticLocalPurpose,
    TemporaryValue,
    UnaryOperator,
    BinaryOperator,
    ComparisonOperator,
    FormatConversion,
    CodeFlag,
    UnpackedTemporaryValue,
    Const,
    LoadName,
    StoreName,
    DeleteName,
    Annotate,
    UnaryOp,
    BinaryOp,
    CompareOp,
    LoadAttr,
    StoreAttr,
    DeleteAttr,
    LoadItem,
    StoreItem,
    DeleteItem,
    BuildTuple,
    BuildList,
    BuildSet,
    BuildMap,
    BuildSlice,
    BuildString,
    FormatValue,
    Unpack,
    Call,
    ImportName,
    ImportFrom,
    ImportStar,
    MakeFunction,
    BuildClass,
    MakeTypeAlias,
    TypeParam,
    TypeParamKind,
    GetIter,
    ForIter,
    GetAIter,
    GetANext,
    GetAwaitable,
    YieldValue,
    YieldFrom,
    AwaitValue,
    CurrentException,
    Raise,
    Reraise,
    CheckExcMatch,
    CheckEGMatch,
    PushTry,
    PopTry,
    ClearException,
    EndFinally,
    Escape,
    Jump,
    Branch,
    Return,
    MatchMapping,
    MatchSequence,
    MatchKeys,
    MatchClass,
    BasicBlock,
    Region,
)

# ---------------------------------------------------------------------------
# Error used when the frontend reaches syntax it still does not lower.
# ---------------------------------------------------------------------------


# ``ast.TypeAlias`` (PEP 695 `type X = ...`) only exists on Python 3.12+.
if sys.version_info >= (3, 12):
    TypeAliasNode = ast.TypeAlias
    TypeVarNode = ast.TypeVar
    ParamSpecNode = ast.ParamSpec
    TypeVarTupleNode = ast.TypeVarTuple
else:
    TypeAliasNode = None
    TypeVarNode = None
    ParamSpecNode = None
    TypeVarTupleNode = None

# Assignment expressions arrived in 3.8, structural pattern matching in 3.10,
# and ``except*`` in 3.11. Guard attribute access so this module still imports
# and lowers older syntax on earlier interpreters.
if sys.version_info >= (3, 8):
    NamedExprNode = ast.NamedExpr
else:
    NamedExprNode = None

if sys.version_info >= (3, 10):
    MatchNode = ast.Match
else:
    MatchNode = None

if sys.version_info >= (3, 11):
    TryStarNode = ast.TryStar
else:
    TryStarNode = None


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


# ---------------------------------------------------------------------------
# Per-region lowering context
# ---------------------------------------------------------------------------


class RegionContext:
    """Per-region state shared by lowering helpers.

    The ``code_obj`` comes from Python's own compiler and is used only for
    metadata / scope shape, not for bytecode lowering.

    This is mutable lowering state, so it is a plain class with regular
    containers rather than a frozen ``attrs`` type.
    """

    def __init__(
        self,
        name: str,
        name_path: COWList,
        is_class: bool,
        node: ast.AST,
        table: Any,
        code_obj: types.CodeType,
        builder: "BlockBuilder",
        child_tables: Optional[List[Any]] = None,
        child_codes: Optional[List[types.CodeType]] = None,
    ) -> None:
        self.name = name
        self.name_path = name_path
        self.is_class = is_class
        self.node = node
        self.table = table
        self.code_obj = code_obj
        self.builder = builder
        self.child_tables: List[Any] = [] if child_tables is None else child_tables
        self.child_codes: List[types.CodeType] = (
            [] if child_codes is None else child_codes
        )
        self.used_child_tables: Set[int] = set()
        self.used_child_codes: Set[int] = set()
        self.next_child_region_label: int = 0


# ---------------------------------------------------------------------------
# CFG construction helper
# ---------------------------------------------------------------------------


class BlockBuilder:
    """Helper for building CFG blocks incrementally.

    Lowering maintains an explicit basic-block graph keyed by labels and
    later materializes the final ordered block sequence for the IR.
    """

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
        block = BasicBlock(
            label=self.current_label,
            instructions=COWList(self.current_instructions),
        )
        self.basic_blocks.append(block)
        self.blocks_by_label[self.current_label] = block
        self.block_successors[self.current_label] = COWList(
            self.successors_for_block(block)
        )
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


# ---------------------------------------------------------------------------
# Child region kind
# ---------------------------------------------------------------------------


class ChildRegionType(str, Enum):
    """Kinds of nested symbol-table regions used by the compiler."""

    FUNCTION = "function"
    CLASS = "class"
    TYPE_PARAMETERS = "type parameters"


class CleanupKind(str, Enum):
    """Kinds of pending cleanup contexts (finally or with-exit)."""

    TRY = "try"
    WITH = "with"


class EarlyExitKind(str, Enum):
    """Kinds of early exits routed through pending cleanups."""

    RETURN = "return"
    BREAK = "break"
    CONTINUE = "continue"


class TryKind(str, Enum):
    """Kinds of try statement lowered by the shared CFG builder."""

    NORMAL = "try"
    EXCEPTION_GROUP = "try*"


# ---------------------------------------------------------------------------
# Compiler state and entrypoints
# ---------------------------------------------------------------------------


class CompilerState:
    """Cross-region compiler state.  Intentionally small and explicit.

    Mutable lowering state, so a plain class with regular lists rather than a
    frozen ``attrs`` type.
    """

    def __init__(self) -> None:
        self.temp_index: int = 0
        self.synthetic_local_index: int = 0
        self.loop_stack: List[Tuple[BasicBlockLabel, BasicBlockLabel]] = []
        self.finally_stack: List["CleanupContext"] = []
        self.region_nested_stacks: List[List[Region]] = []
        self.synthetic_region_name_stacks: List[Dict[str, int]] = []


class CleanupContext:
    """Pending finally/exit cleanup used to route early returns and escapes.

    Mutable lowering state, so a plain class with a regular list rather than a
    frozen ``attrs`` type.
    """

    def __init__(
        self,
        kind: CleanupKind,
        owner: ast.AST,
        pop_count: int = 0,
        finalbody: Sequence[ast.stmt] = (),
        exit_fn: Optional[TemporaryValue] = None,
        is_async: bool = False,
    ) -> None:
        self.kind = kind
        self.owner = owner
        self.pop_count = pop_count
        self.exits: List[Tuple[BasicBlockLabel, EarlyExitKind, Any, int]] = []
        self.finalbody = finalbody
        self.exit_fn = exit_fn
        self.is_async = is_async


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


def fresh_synthetic_local(
    state: CompilerState,
    purpose: SyntheticLocalPurpose = SyntheticLocalPurpose.GENERAL,
) -> SyntheticLocal:
    local = SyntheticLocal(index=state.synthetic_local_index, purpose=purpose)
    state.synthetic_local_index += 1
    return local


def synthetic_local_name(local: SyntheticLocal) -> str:
    """Render a compiler synthetic local as a real ``str`` name operand.

    Synthetic locals must not collide with user identifiers, so they use the
    reserved ``<synthetic:...>`` prefix (Python identifiers cannot contain
    ``<``).  The same object is always rendered to the same string, so the
    store/load/delete pairs still agree while satisfying the ``name: str``
    contract on name instructions.
    """
    return "<synthetic:%s:%d>" % (local.purpose.value, local.index)


def compile_source(state: CompilerState, source: str, path: str = "<ast>") -> Region:
    """Compile one source string into the top-level Region."""
    tree = ast.parse(source, filename=path, mode="exec")
    root_table = symtable.symtable(source, path, "exec")
    root_code = compile(source, path, "exec")
    return compile_region_node(
        state,
        node=tree,
        table=root_table,
        code_obj=root_code,
        name="<module>",
        name_path=COWList(["<module>"]),
        is_class=False,
        label=None,
    )


def compile_file(state: CompilerState, path: str) -> Region:
    with open(path, "r") as f:
        source = f.read()
    return compile_source(state, source, path=path)


def child_code_objects(
    state: CompilerState, code_obj: types.CodeType
) -> List[types.CodeType]:
    return [const for const in code_obj.co_consts if isinstance(const, types.CodeType)]


def finish_region(
    builder: BlockBuilder,
    name: str,
    label: Optional[RegionLabel],
    is_class: bool,
    code_obj: types.CodeType,
    nested_regions: List[Region],
    vararg_name: Optional[str] = None,
    kwarg_name: Optional[str] = None,
) -> Region:
    """Materialize a finished region from its builder and code metadata."""
    basic_blocks = builder.finish()
    return Region(
        name=name,
        entry_label=basic_blocks[0].label,
        label=label,
        is_class=is_class,
        basic_blocks=basic_blocks,
        child_regions=COWList(nested_regions),
        locals=COWList(code_obj.co_varnames),
        cells=COWList(code_obj.co_cellvars),
        freevars=COWList(code_obj.co_freevars),
        argcount=code_obj.co_argcount,
        posonlyargcount=getattr(code_obj, "co_posonlyargcount", 0),
        kwonlyargcount=code_obj.co_kwonlyargcount,
        vararg_name=vararg_name,
        kwarg_name=kwarg_name,
    )


# ---------------------------------------------------------------------------
# AST -> IR compilation (everything below)
# ---------------------------------------------------------------------------
# This file contains the bulk of the AST -> IR compiler, organized as
# explicit lowering functions.  All code below was previously in pyssa.py
# and is preserved verbatim except for updated imports.
# ---------------------------------------------------------------------------
def compile_region_node(
    state: CompilerState,
    node: ast.AST,
    table: Any,
    code_obj: types.CodeType,
    name: str,
    name_path: COWList,
    is_class: bool,
    label: Optional[RegionLabel],
) -> Region:
    """Dispatch to the appropriate region compiler for this AST node."""
    if isinstance(node, ast.GeneratorExp):
        return compile_genexpr_region(
            state,
            node=node,
            table=table,
            code_obj=code_obj,
            name=name,
            name_path=name_path,
            is_class=is_class,
            label=label,
        )
    if isinstance(node, ast.Lambda):
        return compile_lambda_region(
            state,
            node=node,
            table=table,
            code_obj=code_obj,
            name=name,
            name_path=name_path,
            is_class=is_class,
            label=label,
        )
    return compile_region_ast(
        state,
        node=node,
        table=table,
        code_obj=code_obj,
        name=name,
        name_path=name_path,
        is_class=is_class,
        label=label,
    )


def compile_region_ast(
    state: CompilerState,
    node: ast.AST,
    table: Any,
    code_obj: types.CodeType,
    name: str,
    name_path: COWList,
    is_class: bool,
    label: Optional[RegionLabel],
) -> Region:
    # Generic region lowering path used for modules, functions, classes, and coroutines.
    nested_regions = []
    builder = BlockBuilder()
    builder.start()
    previous_loop_stack = state.loop_stack
    previous_finally_stack = state.finally_stack
    state.loop_stack = []
    state.finally_stack = []
    state.region_nested_stacks.append([])
    state.synthetic_region_name_stacks.append({})
    ctx = RegionContext(
        name=name,
        name_path=name_path,
        is_class=is_class,
        node=node,
        table=table,
        code_obj=code_obj,
        builder=builder,
        child_tables=list(table.get_children()),
        child_codes=child_code_objects(state, code_obj),
    )
    body = getattr(node, "body", ())
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
    state.loop_stack = previous_loop_stack
    state.finally_stack = previous_finally_stack
    return finish_region(
        builder, name, label, is_class, code_obj, nested_regions,
        vararg_name, kwarg_name,
    )


def compile_genexpr_region(
    state: CompilerState,
    node: ast.GeneratorExp,
    table: Any,
    code_obj: types.CodeType,
    name: str,
    name_path: COWList,
    is_class: bool,
    label: Optional[RegionLabel],
) -> Region:
    # Generator expressions are themselves nested executable regions with yield points.
    builder = BlockBuilder()
    builder.start()
    previous_loop_stack = state.loop_stack
    previous_finally_stack = state.finally_stack
    state.loop_stack = []
    state.finally_stack = []
    state.region_nested_stacks.append([])
    state.synthetic_region_name_stacks.append({})
    ctx = RegionContext(
        name=name,
        name_path=name_path,
        is_class=is_class,
        node=node,
        table=table,
        code_obj=code_obj,
        builder=builder,
        child_tables=list(table.get_children()),
        child_codes=child_code_objects(state, code_obj),
    )
    after_label = builder.new_label()
    lower_genexpr_generator(
        state,
        ctx,
        node.generators,
        0,
        lambda: emit_genexpr_yield(state, ctx, node.elt),
        after_label,
        node,
    )
    builder.start_block(after_label)
    if builder.is_open():
        emit_return_none(state, builder, node)
        builder.finish_block()
    nested_regions = state.region_nested_stacks.pop()
    state.synthetic_region_name_stacks.pop()
    state.loop_stack = previous_loop_stack
    state.finally_stack = previous_finally_stack
    return finish_region(builder, name, label, is_class, code_obj, nested_regions)


def compile_lambda_region(
    state: CompilerState,
    node: ast.Lambda,
    table: Any,
    code_obj: types.CodeType,
    name: str,
    name_path: COWList,
    is_class: bool,
    label: Optional[RegionLabel],
) -> Region:
    # Lambdas are expression-bodied nested regions that return their body value directly.
    builder = BlockBuilder()
    builder.start()
    previous_loop_stack = state.loop_stack
    previous_finally_stack = state.finally_stack
    state.loop_stack = []
    state.finally_stack = []
    state.region_nested_stacks.append([])
    state.synthetic_region_name_stacks.append({})
    ctx = RegionContext(
        name=name,
        name_path=name_path,
        is_class=is_class,
        node=node,
        table=table,
        code_obj=code_obj,
        builder=builder,
        child_tables=list(table.get_children()),
        child_codes=child_code_objects(state, code_obj),
    )
    value = lower_expr(state, ctx, node.body)
    if builder.is_open():
        builder.terminate(attach_meta(state, Return(value=value), node.body))
    nested_regions = state.region_nested_stacks.pop()
    state.synthetic_region_name_stacks.pop()
    vararg_name, kwarg_name = region_variadic_names(node)
    state.loop_stack = previous_loop_stack
    state.finally_stack = previous_finally_stack
    return finish_region(
        builder, name, label, is_class, code_obj, nested_regions,
        vararg_name, kwarg_name,
    )


def emit_genexpr_yield(
    state: CompilerState, ctx: RegionContext, elt: ast.AST
) -> None:
    value = lower_expr(state, ctx, elt)
    temp = fresh_temp(state)
    ctx.builder.emit(attach_meta(state, YieldValue(dst=temp, value=value), elt))


def lower_genexpr_generator(
    state: CompilerState,
    ctx: RegionContext,
    generators: List[ast.comprehension],
    index: int,
    emit_item: Any,
    exhaustion_label: BasicBlockLabel,
    owner: ast.AST,
) -> None:
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
    ctx.builder.emit(
        attach_meta(
            state,
            LoadName(dst=current_iter, scope=Scope.LOCAL, name=iter_name),
            generator,
        )
    )
    if generator.is_async:
        next_awaitable = fresh_temp(state)
        ctx.builder.emit(
            attach_meta(
                state, GetANext(dst=next_awaitable, aiter=current_iter), generator
            )
        )
        ctx.builder.emit(
            attach_meta(state, PushTry(except_label=stop_label), generator)
        )
        value_dst = await_value(state, ctx, next_awaitable, generator)
        ctx.builder.emit(attach_meta(state, PopTry(), generator))
        ctx.builder.terminate(attach_meta(state, Jump(target=body_label), generator))
        ctx.builder.start_block(stop_label)
        current_exc = current_exception_value(state, ctx, generator)
        stop_type = builtin_const_value(
            state, ctx, builtins.StopAsyncIteration, generator
        )
        matched = fresh_temp(state)
        ctx.builder.emit(
            attach_meta(
                state,
                CheckExcMatch(dst=matched, exc=current_exc, typ=stop_type),
                generator,
            )
        )
        ctx.builder.terminate(
            attach_meta(
                state,
                Branch(
                    cond=matched,
                    true_label=stop_match_label,
                    false_label=stop_nomatch_label,
                ),
                generator,
            )
        )
        ctx.builder.start_block(stop_match_label)
        ctx.builder.emit(attach_meta(state, ClearException(), generator))
        ctx.builder.terminate(attach_meta(state, Jump(target=cleanup_label), generator))
        ctx.builder.start_block(stop_nomatch_label)
        ctx.builder.terminate(attach_meta(state, Reraise(), generator))
        ctx.builder.start_block(body_label)
    else:
        value_dst = fresh_temp(state)
        ctx.builder.terminate(
            attach_meta(
                state,
                ForIter(
                    iter_obj=current_iter,
                    value_dst=value_dst,
                    body_label=body_label,
                    exit_label=cleanup_label,
                ),
                generator,
            )
        )
        ctx.builder.start_block(body_label)
    assign_target(state, ctx, generator.target, value_dst)
    for if_expr in generator.ifs:
        next_label = ctx.builder.new_label()
        cond = lower_expr(state, ctx, if_expr)
        ctx.builder.terminate(
            attach_meta(
                state,
                Branch(cond=cond, true_label=next_label, false_label=header_label),
                if_expr,
            )
        )
        ctx.builder.start_block(next_label)
    if index + 1 < len(generators):
        lower_genexpr_generator(
            state, ctx, generators, index + 1, emit_item, header_label, owner
        )
    else:
        emit_item()
        if ctx.builder.is_open():
            ctx.builder.terminate(attach_meta(state, Jump(target=header_label), owner))
    ctx.builder.start_block(cleanup_label)
    if owns_iter_name:
        ctx.builder.emit(
            attach_meta(state, DeleteName(scope=Scope.LOCAL, name=iter_name), generator)
        )
    ctx.builder.terminate(attach_meta(state, Jump(target=exhaustion_label), generator))


def lower_stmt_list(
    state: CompilerState, ctx: RegionContext, stmts: Sequence[ast.stmt]
) -> List[Region]:
    nested_regions = []
    for stmt in stmts:
        child_regions = lower_stmt(state, ctx, stmt)
        if child_regions:
            nested_regions.extend(child_regions)
    return nested_regions


def lower_stmt(
    state: CompilerState, ctx: RegionContext, stmt: ast.stmt
) -> List[Region]:
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
        if stmt.value is not None:
            value = lower_expr(state, ctx, stmt.value)
            assign_target(state, ctx, stmt.target, value)
        obj_label, obj_region = lower_expression_region(
            state, ctx, stmt.target, "<annotation-target>"
        )
        ann_label, ann_region = lower_expression_region(
            state, ctx, stmt.annotation, "<annotation>"
        )
        ctx.builder.emit(
            attach_meta(
                state, Annotate(obj=obj_label, annotation=ann_label), stmt
            )
        )
        return [obj_region, ann_region]
    if TypeAliasNode is not None and isinstance(stmt, TypeAliasNode):
        # PEP 695 `type X = ...`: the alias value is lazy, so it goes in its own
        # nested region; the alias object is built and bound by name.
        if not isinstance(stmt.name, ast.Name):
            raise UnsupportedFeature(stmt, "type alias target is not a name")
        value_label, value_region = lower_expression_region(
            state, ctx, stmt.value, "<type-alias-value>"
        )
        type_params, type_param_regions = lower_type_params(state, ctx, stmt)
        alias_temp = fresh_temp(state)
        ctx.builder.emit(
            attach_meta(
                state,
                MakeTypeAlias(
                    dst=alias_temp,
                    name=stmt.name.id,
                    value=value_label,
                    type_params=COWList(type_params),
                ),
                stmt,
            )
        )
        scope = scope_for_store(state, ctx, stmt.name.id)
        ctx.builder.emit(
            attach_meta(
                state,
                StoreName(src=alias_temp, scope=scope, name=stmt.name.id),
                stmt,
            )
        )
        return [value_region] + type_param_regions
    if isinstance(stmt, ast.AugAssign):
        lower_augassign(state, ctx, stmt)
        return []
    if isinstance(stmt, ast.Return):
        value = lower_optional_expr(state, ctx, stmt.value, stmt)
        emit_exit(state, ctx, EarlyExitKind.RETURN, value, stmt)
        return []
    if isinstance(stmt, ast.Expr):
        lower_expr(state, ctx, stmt.value)
        return []
    if isinstance(stmt, ast.If):
        return lower_if(state, ctx, stmt)
    if isinstance(stmt, ast.For):
        return lower_for(state, ctx, stmt)
    if isinstance(stmt, ast.While):
        return lower_while(state, ctx, stmt)
    if isinstance(stmt, ast.AsyncFor):
        return lower_async_for(state, ctx, stmt)
    if isinstance(stmt, ast.With):
        return lower_with(state, ctx, stmt)
    if isinstance(stmt, ast.AsyncWith):
        return lower_async_with(state, ctx, stmt)
    if TryStarNode is not None and isinstance(stmt, TryStarNode):
        return lower_try_star(state, ctx, stmt)
    if isinstance(stmt, ast.Try):
        return lower_try(state, ctx, stmt)
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
    if (
        isinstance(stmt, ast.Global)
        or isinstance(stmt, ast.Nonlocal)
        or isinstance(stmt, ast.Pass)
    ):
        return []
    if isinstance(stmt, ast.Raise):
        if stmt.exc is None:
            ctx.builder.terminate(attach_meta(state, Reraise(), stmt))
            return []
        exc = lower_expr(state, ctx, stmt.exc)
        cause = None if stmt.cause is None else lower_expr(state, ctx, stmt.cause)
        ctx.builder.terminate(attach_meta(state, Raise(exc=exc, cause=cause), stmt))
        return []
    if isinstance(stmt, ast.Assert):
        lower_assert(state, ctx, stmt)
        return []
    if isinstance(stmt, ast.Delete):
        for target in stmt.targets:
            delete_target(state, ctx, target)
        return []
    if MatchNode is not None and isinstance(stmt, MatchNode):
        return lower_match(state, ctx, stmt)
    raise UnsupportedFeature(
        stmt, "statement %s is not implemented in AST lowering" % type(stmt).__name__
    )


def lower_expression_region(
    state: CompilerState, ctx: RegionContext, expr: ast.AST, name: str
) -> Tuple[RegionLabel, Region]:
    """Lower a single expression into a self-contained child region.

    Used for annotation targets and annotation values: the expression is
    lowered to ordinary IR (resolving names against the *enclosing* scope,
    since an annotation is not a separate Python scope) and returned as a
    nested region ending in ``Return``.  Nothing is emitted into ``ctx``'s
    block, so the annotation never runs unless an interpreter evaluates the
    region explicitly.
    """
    label = fresh_child_region_label(ctx)
    builder = BlockBuilder()
    builder.start()
    previous_finally_stack = state.finally_stack
    state.finally_stack = []
    state.region_nested_stacks.append([])
    state.synthetic_region_name_stacks.append({})
    sub_ctx = RegionContext(
        name=ctx.name,
        name_path=ctx.name_path,
        is_class=ctx.is_class,
        node=expr,
        table=ctx.table,
        code_obj=ctx.code_obj,
        builder=builder,
        child_tables=list(ctx.table.get_children()),
        child_codes=child_code_objects(state, ctx.code_obj),
    )
    value = lower_expr(state, sub_ctx, expr)
    builder.terminate(attach_meta(state, Return(value=value), expr))
    nested_regions = state.region_nested_stacks.pop()
    state.synthetic_region_name_stacks.pop()
    state.finally_stack = previous_finally_stack
    basic_blocks = builder.finish()
    region = Region(
        name=name,
        entry_label=basic_blocks[0].label,
        label=label,
        is_class=False,
        basic_blocks=basic_blocks,
        child_regions=COWList(nested_regions),
    )
    return label, region


def lower_function_def(
    state: CompilerState,
    parent_ctx: RegionContext,
    node: ast.AST,
    is_async: bool,
) -> List[Region]:
    # Build the nested function region first, then wrap it in a runtime function object.
    child_table, child_code = take_child_region_inputs(
        state,
        parent_ctx,
        table_type=ChildRegionType.FUNCTION,
        symtable_name=node.name,
        code_name=node.name,
        owner=node,
    )
    child_label = fresh_child_region_label(parent_ctx)
    child_name = child_region_name(state, node.name)
    child_path = child_name_path(state, parent_ctx, node.name, for_class=False)
    nested_region = compile_region_node(
        state,
        node=node,
        table=child_table,
        code_obj=child_code,
        name=child_name,
        name_path=child_path,
        is_class=False,
        label=child_label,
    )
    # Decorator expressions are evaluated before defaults/annotations and in
    # source order; they are applied after the function is built, bottom-up.
    decorator_values = [lower_expr(state, parent_ctx, d) for d in node.decorator_list]
    default_values = COWList(
        [lower_expr(state, parent_ctx, value) for value in node.args.defaults]
    )
    kwonly_items = []
    for arg, default in zip(node.args.kwonlyargs, node.args.kw_defaults):
        if default is None:
            continue
        kwonly_items.append((arg.arg, lower_expr(state, parent_ctx, default)))
    # Annotations are lowered into their own nested expression regions so
    # signatures are lazy: they are never evaluated when the enclosing region
    # runs, which keeps forward references (common in stubs) from failing.
    annotation_items = []
    annotation_regions = []
    all_annotated_args = list(node.args.posonlyargs) + list(node.args.args)
    if node.args.vararg is not None:
        all_annotated_args.append(node.args.vararg)
    all_annotated_args.extend(node.args.kwonlyargs)
    if node.args.kwarg is not None:
        all_annotated_args.append(node.args.kwarg)
    for arg in all_annotated_args:
        if arg.annotation is not None:
            label, region = lower_expression_region(
                state, parent_ctx, arg.annotation, "<annotation>"
            )
            annotation_items.append((arg.arg, label))
            annotation_regions.append(region)
    if node.returns is not None:
        label, region = lower_expression_region(
            state, parent_ctx, node.returns, "<annotation>"
        )
        annotation_items.append(("return", label))
        annotation_regions.append(region)
    type_params, type_param_regions = lower_type_params(state, parent_ctx, node)
    func_temp = fresh_temp(state)
    parent_ctx.builder.emit(
        attach_meta(
            state,
            MakeFunction(
                dst=func_temp,
                code=child_label,
                defaults=default_values,
                kwdefaults=COWList(kwonly_items),
                annotations=COWList(annotation_items),
                type_params=COWList(type_params),
                flags=CodeFlag(child_code.co_flags),
            ),
            node,
        )
    )
    decorated = func_temp
    for decorator_value in reversed(decorator_values):
        call_temp = fresh_temp(state)
        parent_ctx.builder.emit(
            attach_meta(
                state,
                Call(
                    dst=call_temp,
                    callee=decorator_value,
                    args=normal_call_args([decorated]),
                    kwargs=normal_call_kwargs(),
                ),
                node,
            )
        )
        decorated = call_temp
    scope = scope_for_store(state, parent_ctx, node.name)
    parent_ctx.builder.emit(
        attach_meta(state, StoreName(src=decorated, scope=scope, name=node.name), node)
    )
    return [nested_region] + annotation_regions + type_param_regions


def lower_class_def(
    state: CompilerState, parent_ctx: RegionContext, node: ast.ClassDef
) -> List[Region]:
    # Classes lower as a nested body region plus an explicit BuildClass operation.
    child_table, child_code = take_child_region_inputs(
        state,
        parent_ctx,
        table_type=ChildRegionType.CLASS,
        symtable_name=node.name,
        code_name=node.name,
        owner=node,
    )
    child_label = fresh_child_region_label(parent_ctx)
    child_name = child_region_name(state, node.name)
    child_path = child_name_path(state, parent_ctx, node.name, for_class=True)
    nested_region = compile_region_node(
        state,
        node=node,
        table=child_table,
        code_obj=child_code,
        name=child_name,
        name_path=child_path,
        is_class=True,
        label=child_label,
    )
    # Decorator expressions are evaluated before bases/keywords and in source
    # order; they are applied after the class is built, bottom-up.
    decorator_values = [lower_expr(state, parent_ctx, d) for d in node.decorator_list]
    body_func = fresh_temp(state)
    parent_ctx.builder.emit(
        attach_meta(
            state,
            MakeFunction(
                dst=body_func, code=child_label, flags=CodeFlag(child_code.co_flags)
            ),
            node,
        )
    )
    name_temp = const_value(state, parent_ctx, node.name, node)
    bases = [lower_expr(state, parent_ctx, base) for base in node.bases]
    keywords = []
    for keyword in node.keywords:
        if keyword.arg is None:
            raise UnsupportedFeature(
                keyword, "class **kwargs are not implemented in AST lowering"
            )
        keywords.append((keyword.arg, lower_expr(state, parent_ctx, keyword.value)))
    type_params, type_param_regions = lower_type_params(state, parent_ctx, node)
    class_temp = fresh_temp(state)
    parent_ctx.builder.emit(
        attach_meta(
            state,
            BuildClass(
                dst=class_temp,
                body_func=body_func,
                name=name_temp,
                bases=COWList(bases),
                keywords=COWList(keywords),
                type_params=COWList(type_params),
            ),
            node,
        )
    )
    decorated = class_temp
    for decorator_value in reversed(decorator_values):
        call_temp = fresh_temp(state)
        parent_ctx.builder.emit(
            attach_meta(
                state,
                Call(
                    dst=call_temp,
                    callee=decorator_value,
                    args=normal_call_args([decorated]),
                    kwargs=normal_call_kwargs(),
                ),
                node,
            )
        )
        decorated = call_temp
    scope = scope_for_store(state, parent_ctx, node.name)
    parent_ctx.builder.emit(
        attach_meta(state, StoreName(src=decorated, scope=scope, name=node.name), node)
    )
    return [nested_region] + type_param_regions


def lower_if(
    state: CompilerState, ctx: RegionContext, stmt: ast.If
) -> List[Region]:
    nested_regions = []
    cond = lower_expr(state, ctx, stmt.test)
    then_label = ctx.builder.new_label()
    else_label = ctx.builder.new_label()
    end_label = ctx.builder.new_label()
    ctx.builder.terminate(
        attach_meta(
            state,
            Branch(cond=cond, true_label=then_label, false_label=else_label),
            stmt.test,
        )
    )
    ctx.builder.start_block(then_label)
    nested_regions.extend(lower_stmt_list(state, ctx, stmt.body))
    then_open = ctx.builder.is_open()
    if then_open:
        ctx.builder.terminate(attach_meta(state, Jump(target=end_label), stmt))
    ctx.builder.start_block(else_label)
    nested_regions.extend(lower_stmt_list(state, ctx, stmt.orelse))
    else_open = ctx.builder.is_open()
    if else_open:
        ctx.builder.terminate(attach_meta(state, Jump(target=end_label), stmt))
    if then_open or else_open:
        ctx.builder.start_block(end_label)
    return nested_regions


def lower_assert(
    state: CompilerState, ctx: RegionContext, stmt: ast.Assert
) -> None:
    # Match Python's compile-time handling of assert under optimization.
    if not __debug__:
        return
    cond = lower_expr(state, ctx, stmt.test)
    continue_label = ctx.builder.new_label()
    raise_label = ctx.builder.new_label()
    ctx.builder.terminate(
        attach_meta(
            state,
            Branch(cond=cond, true_label=continue_label, false_label=raise_label),
            stmt.test,
        )
    )
    ctx.builder.start_block(raise_label)
    assertion_error = builtin_const_value(state, ctx, builtins.AssertionError, stmt)
    exc = assertion_error
    if stmt.msg is not None:
        msg = lower_expr(state, ctx, stmt.msg)
        exc = fresh_temp(state)
        ctx.builder.emit(
            attach_meta(
                state,
                Call(
                    dst=exc,
                    callee=assertion_error,
                    args=normal_call_args([msg]),
                    kwargs=normal_call_kwargs(),
                ),
                stmt.msg,
            )
        )
    ctx.builder.terminate(attach_meta(state, Raise(exc=exc), stmt))
    ctx.builder.start_block(continue_label)


def push_loop(
    state: CompilerState,
    break_label: BasicBlockLabel,
    continue_label: BasicBlockLabel,
) -> None:
    state.loop_stack.append((break_label, continue_label))


def pop_loop(state: CompilerState) -> None:
    state.loop_stack.pop()


def current_loop(
    state: CompilerState, node: ast.AST
) -> Tuple[BasicBlockLabel, BasicBlockLabel]:
    if not state.loop_stack:
        raise UnsupportedFeature(node, "%s outside loop" % type(node).__name__.lower())
    return state.loop_stack[-1]


def emit_exit(
    state: CompilerState,
    ctx: RegionContext,
    kind: EarlyExitKind,
    payload: Any,
    node: ast.AST,
) -> None:
    """Terminate a block with an early exit, routing it through any active
    finally/with cleanup first.

    When a cleanup context is active, the exit is deferred: we jump to a
    dedicated cleanup block and record the exit so the enclosing ``try``/
    ``with`` lowering can run its cleanup and then re-emit the exit (which may
    be intercepted again by an outer cleanup).
    """
    cleanup = state.finally_stack[-1] if state.finally_stack else None
    if cleanup is not None:
        cleanup_label = ctx.builder.new_label()
        cleanup.exits.append((cleanup_label, kind, payload, cleanup.pop_count))
        ctx.builder.terminate(attach_meta(state, Jump(target=cleanup_label), node))
        return
    if kind == EarlyExitKind.RETURN:
        ctx.builder.terminate(attach_meta(state, Return(value=payload), node))
        return
    if kind in (EarlyExitKind.BREAK, EarlyExitKind.CONTINUE):
        ctx.builder.terminate(attach_meta(state, Escape(target=payload), node))
        return
    raise UnsupportedFeature(node, "unknown early exit kind %r" % kind)


def process_cleanup_exits(
    state: CompilerState, ctx: RegionContext, cleanup: "CleanupContext"
) -> List[Region]:
    """Emit one cleanup block per deferred early exit."""
    nested_regions = []
    original_used_child_tables = set(ctx.used_child_tables)
    original_used_child_codes = set(ctx.used_child_codes)
    for cleanup_label, kind, payload, pop_count in cleanup.exits:
        ctx.builder.start_block(cleanup_label)
        for _ in range(pop_count):
            ctx.builder.emit(attach_meta(state, PopTry(), cleanup.owner))
        if cleanup.kind == CleanupKind.TRY:
            # A finally body is emitted independently for each early-exit path.
            # Reuse its symbol-table/code inputs for each CFG copy; the normal
            # exceptional path below will consume them permanently.
            ctx.used_child_tables = set(original_used_child_tables)
            ctx.used_child_codes = set(original_used_child_codes)
            nested_regions.extend(lower_stmt_list(state, ctx, cleanup.finalbody))
        elif cleanup.kind == CleanupKind.WITH:
            none1 = const_value(state, ctx, None, cleanup.owner)
            none2 = const_value(state, ctx, None, cleanup.owner)
            none3 = const_value(state, ctx, None, cleanup.owner)
            if cleanup.is_async:
                call_and_await(
                    state, ctx, cleanup.exit_fn, [none1, none2, none3], cleanup.owner
                )
            else:
                ignored = fresh_temp(state)
                ctx.builder.emit(
                    attach_meta(
                        state,
                        Call(
                            dst=ignored,
                            callee=cleanup.exit_fn,
                            args=normal_call_args([none1, none2, none3]),
                            kwargs=normal_call_kwargs(),
                        ),
                        cleanup.owner,
                    )
                )
        if ctx.builder.is_open():
            emit_exit(state, ctx, kind, payload, cleanup.owner)
    ctx.used_child_tables = original_used_child_tables
    ctx.used_child_codes = original_used_child_codes
    return nested_regions


def lower_break(
    state: CompilerState, ctx: RegionContext, stmt: ast.Break
) -> None:
    break_label, _ = current_loop(state, stmt)
    emit_exit(state, ctx, EarlyExitKind.BREAK, break_label, stmt)


def lower_continue(
    state: CompilerState, ctx: RegionContext, stmt: ast.Continue
) -> None:
    _, continue_label = current_loop(state, stmt)
    emit_exit(state, ctx, EarlyExitKind.CONTINUE, continue_label, stmt)


def lower_import(
    state: CompilerState, ctx: RegionContext, stmt: ast.Import
) -> None:
    for alias in stmt.names:
        # ``import a.b`` binds the top-level package ``a`` (and Python also
        # imports ``a.b`` as a side effect); ``import a.b as c`` binds ``c`` to
        # ``a.b`` itself.  We keep the IR simple and bind the package for the
        # plain dotted form, matching Python's visible name binding.
        if alias.asname is not None:
            import_module = alias.name
            store_name = alias.asname
        else:
            import_module = alias.name.split(".", 1)[0]
            store_name = import_module
            if import_module != alias.name:
                # Import the complete dotted path for its package-loading side
                # effects, then bind the top-level package below.
                imported_submodule = fresh_temp(state)
                ctx.builder.emit(
                    attach_meta(
                        state,
                        ImportName(
                            dst=imported_submodule,
                            module=alias.name,
                            fromlist=COWList(),
                            level=0,
                        ),
                        stmt,
                    )
                )
        module_temp = fresh_temp(state)
        ctx.builder.emit(
            attach_meta(
                state,
                ImportName(
                    dst=module_temp, module=import_module, fromlist=COWList(), level=0
                ),
                stmt,
            )
        )
        scope = scope_for_store(state, ctx, store_name)
        ctx.builder.emit(
            attach_meta(
                state, StoreName(src=module_temp, scope=scope, name=store_name), stmt
            )
        )


def lower_import_from(
    state: CompilerState, ctx: RegionContext, stmt: ast.ImportFrom
) -> None:
    module_name = stmt.module
    fromlist = [alias.name for alias in stmt.names]
    module_temp = fresh_temp(state)
    ctx.builder.emit(
        attach_meta(
            state,
            ImportName(
                dst=module_temp,
                module=module_name,
                fromlist=COWList(fromlist),
                level=stmt.level,
            ),
            stmt,
        )
    )
    if len(stmt.names) == 1 and stmt.names[0].name == "*":
        ctx.builder.emit(attach_meta(state, ImportStar(module_obj=module_temp), stmt))
        return
    for alias in stmt.names:
        imported = fresh_temp(state)
        ctx.builder.emit(
            attach_meta(
                state,
                ImportFrom(dst=imported, module_obj=module_temp, name=alias.name),
                stmt,
            )
        )
        store_name = alias.asname or alias.name
        scope = scope_for_store(state, ctx, store_name)
        ctx.builder.emit(
            attach_meta(
                state, StoreName(src=imported, scope=scope, name=store_name), stmt
            )
        )


def lower_augassign(
    state: CompilerState, ctx: RegionContext, stmt: ast.AugAssign
) -> None:
    # Python evaluates the target (object and key) and reads its current value
    # *before* evaluating the right-hand side.  Lower the RHS last so the
    # generated IR preserves that order.
    op = binary_op(state, stmt.op)
    target = stmt.target
    if isinstance(target, ast.Name):
        current = lower_expr(state, ctx, ast.Name(id=target.id, ctx=ast.Load()))
        value = lower_expr(state, ctx, stmt.value)
        result = fresh_temp(state)
        ctx.builder.emit(
            attach_meta(
                state, BinaryOp(dst=result, op=op, lhs=current, rhs=value), stmt
            )
        )
        scope = scope_for_store(state, ctx, target.id)
        ctx.builder.emit(
            attach_meta(state, StoreName(src=result, scope=scope, name=target.id), stmt)
        )
        return
    if isinstance(target, ast.Attribute):
        obj = lower_expr(state, ctx, target.value)
        current = fresh_temp(state)
        ctx.builder.emit(
            attach_meta(
                state, LoadAttr(dst=current, obj=obj, attr_name=target.attr), target
            )
        )
        value = lower_expr(state, ctx, stmt.value)
        result = fresh_temp(state)
        ctx.builder.emit(
            attach_meta(
                state, BinaryOp(dst=result, op=op, lhs=current, rhs=value), stmt
            )
        )
        ctx.builder.emit(
            attach_meta(
                state, StoreAttr(obj=obj, attr_name=target.attr, value=result), stmt
            )
        )
        return
    if isinstance(target, ast.Subscript):
        obj = lower_expr(state, ctx, target.value)
        key = lower_slice_expr(state, ctx, target.slice)
        current = fresh_temp(state)
        ctx.builder.emit(
            attach_meta(state, LoadItem(dst=current, obj=obj, key=key), target)
        )
        value = lower_expr(state, ctx, stmt.value)
        result = fresh_temp(state)
        ctx.builder.emit(
            attach_meta(
                state, BinaryOp(dst=result, op=op, lhs=current, rhs=value), stmt
            )
        )
        ctx.builder.emit(
            attach_meta(state, StoreItem(obj=obj, key=key, value=result), stmt)
        )
        return
    raise UnsupportedFeature(
        target,
        "augmented assignment target %s is not implemented in AST lowering"
        % type(target).__name__,
    )


def lower_for(
    state: CompilerState, ctx: RegionContext, stmt: ast.For
) -> List[Region]:
    nested_regions = []
    iterable = lower_expr(state, ctx, stmt.iter)
    iter_temp = fresh_temp(state)
    ctx.builder.emit(
        attach_meta(state, GetIter(dst=iter_temp, iterable=iterable), stmt.iter)
    )
    iter_name = synthetic_local_name(fresh_synthetic_local(state, SyntheticLocalPurpose.FOR_ITER))
    ctx.builder.emit(
        attach_meta(
            state, StoreName(src=iter_temp, scope=Scope.LOCAL, name=iter_name), stmt
        )
    )
    header_label = ctx.builder.new_label()
    body_label = ctx.builder.new_label()
    orelse_label = ctx.builder.new_label() if stmt.orelse else None
    exit_label = orelse_label or ctx.builder.new_label()
    final_label = ctx.builder.new_label() if stmt.orelse else exit_label
    ctx.builder.terminate(attach_meta(state, Jump(target=header_label), stmt))
    ctx.builder.start_block(header_label)
    current_iter = fresh_temp(state)
    ctx.builder.emit(
        attach_meta(
            state, LoadName(dst=current_iter, scope=Scope.LOCAL, name=iter_name), stmt
        )
    )
    value_dst = fresh_temp(state)
    ctx.builder.terminate(
        attach_meta(
            state,
            ForIter(
                iter_obj=current_iter,
                value_dst=value_dst,
                body_label=body_label,
                exit_label=exit_label,
            ),
            stmt,
        )
    )
    ctx.builder.start_block(body_label)
    push_loop(state, final_label, header_label)
    assign_target(state, ctx, stmt.target, value_dst)
    nested_regions.extend(lower_stmt_list(state, ctx, stmt.body))
    pop_loop(
        state,
    )
    if ctx.builder.is_open():
        ctx.builder.terminate(attach_meta(state, Jump(target=header_label), stmt))
    if stmt.orelse:
        ctx.builder.start_block(orelse_label)
        nested_regions.extend(lower_stmt_list(state, ctx, stmt.orelse))
        if ctx.builder.is_open():
            ctx.builder.terminate(attach_meta(state, Jump(target=final_label), stmt))
        ctx.builder.start_block(final_label)
    else:
        ctx.builder.start_block(exit_label)
    return nested_regions


def lower_while(
    state: CompilerState, ctx: RegionContext, stmt: ast.While
) -> List[Region]:
    nested_regions = []
    cond_label = ctx.builder.new_label()
    body_label = ctx.builder.new_label()
    orelse_label = ctx.builder.new_label() if stmt.orelse else None
    exit_label = orelse_label or ctx.builder.new_label()
    final_label = ctx.builder.new_label() if stmt.orelse else exit_label
    ctx.builder.terminate(attach_meta(state, Jump(target=cond_label), stmt))
    ctx.builder.start_block(cond_label)
    cond = lower_expr(state, ctx, stmt.test)
    ctx.builder.terminate(
        attach_meta(
            state,
            Branch(cond=cond, true_label=body_label, false_label=exit_label),
            stmt.test,
        )
    )
    ctx.builder.start_block(body_label)
    push_loop(state, final_label, cond_label)
    nested_regions.extend(lower_stmt_list(state, ctx, stmt.body))
    pop_loop(
        state,
    )
    if ctx.builder.is_open():
        ctx.builder.terminate(attach_meta(state, Jump(target=cond_label), stmt))
    if stmt.orelse:
        ctx.builder.start_block(orelse_label)
        nested_regions.extend(lower_stmt_list(state, ctx, stmt.orelse))
        if ctx.builder.is_open():
            ctx.builder.terminate(attach_meta(state, Jump(target=final_label), stmt))
        ctx.builder.start_block(final_label)
    else:
        ctx.builder.start_block(exit_label)
    return nested_regions


def lower_async_for(
    state: CompilerState, ctx: RegionContext, stmt: ast.AsyncFor
) -> List[Region]:
    nested_regions = []
    # Async iteration is explicit: get __aiter__, await each __anext__, and catch StopAsyncIteration.
    iterable = lower_expr(state, ctx, stmt.iter)
    aiter_temp = fresh_temp(state)
    ctx.builder.emit(
        attach_meta(state, GetAIter(dst=aiter_temp, iterable=iterable), stmt.iter)
    )
    iter_name = synthetic_local_name(fresh_synthetic_local(state, SyntheticLocalPurpose.ASYNC_FOR_ITER))
    ctx.builder.emit(
        attach_meta(
            state, StoreName(src=aiter_temp, scope=Scope.LOCAL, name=iter_name), stmt
        )
    )
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
    ctx.builder.emit(
        attach_meta(
            state, LoadName(dst=current_iter, scope=Scope.LOCAL, name=iter_name), stmt
        )
    )
    next_awaitable = fresh_temp(state)
    ctx.builder.emit(
        attach_meta(state, GetANext(dst=next_awaitable, aiter=current_iter), stmt)
    )
    ctx.builder.emit(attach_meta(state, PushTry(except_label=stop_label), stmt))
    next_value = await_value(state, ctx, next_awaitable, stmt)
    ctx.builder.emit(attach_meta(state, PopTry(), stmt))
    ctx.builder.terminate(attach_meta(state, Jump(target=body_label), stmt))
    ctx.builder.start_block(stop_label)
    current_exc = current_exception_value(state, ctx, stmt)
    stop_type = builtin_const_value(state, ctx, builtins.StopAsyncIteration, stmt)
    matched = fresh_temp(state)
    ctx.builder.emit(
        attach_meta(
            state, CheckExcMatch(dst=matched, exc=current_exc, typ=stop_type), stmt
        )
    )
    ctx.builder.terminate(
        attach_meta(
            state,
            Branch(
                cond=matched,
                true_label=stop_match_label,
                false_label=stop_nomatch_label,
            ),
            stmt,
        )
    )
    ctx.builder.start_block(stop_match_label)
    ctx.builder.emit(attach_meta(state, ClearException(), stmt))
    ctx.builder.terminate(attach_meta(state, Jump(target=exit_label), stmt))
    ctx.builder.start_block(stop_nomatch_label)
    ctx.builder.terminate(attach_meta(state, Reraise(), stmt))
    ctx.builder.start_block(body_label)
    push_loop(state, final_label, header_label)
    assign_target(state, ctx, stmt.target, next_value)
    nested_regions.extend(lower_stmt_list(state, ctx, stmt.body))
    pop_loop(
        state,
    )
    if ctx.builder.is_open():
        ctx.builder.terminate(attach_meta(state, Jump(target=header_label), stmt))
    if stmt.orelse:
        ctx.builder.start_block(orelse_label)
        nested_regions.extend(lower_stmt_list(state, ctx, stmt.orelse))
        if ctx.builder.is_open():
            ctx.builder.terminate(attach_meta(state, Jump(target=final_label), stmt))
        ctx.builder.start_block(final_label)
    else:
        ctx.builder.start_block(exit_label)
    return nested_regions


def await_value(
    state: CompilerState,
    ctx: RegionContext,
    value: TemporaryValue,
    node: ast.AST,
) -> TemporaryValue:
    awaitable = fresh_temp(state)
    ctx.builder.emit(
        attach_meta(state, GetAwaitable(dst=awaitable, value=value), node)
    )
    awaited = fresh_temp(state)
    ctx.builder.emit(attach_meta(state, AwaitValue(dst=awaited, value=awaitable), node))
    return awaited


def call_and_await(
    state: CompilerState,
    ctx: RegionContext,
    callee: TemporaryValue,
    args: List[TemporaryValue],
    node: ast.AST,
) -> TemporaryValue:
    call_result = fresh_temp(state)
    ctx.builder.emit(
        attach_meta(
            state,
            Call(
                dst=call_result,
                callee=callee,
                args=normal_call_args(args),
                kwargs=normal_call_kwargs(),
            ),
            node,
        )
    )
    return await_value(state, ctx, call_result, node)


def lower_with(
    state: CompilerState, ctx: RegionContext, stmt: ast.With
) -> List[Region]:
    return lower_with_items(state, ctx, stmt.items, stmt.body, stmt, is_async=False)


def lower_async_with(
    state: CompilerState, ctx: RegionContext, stmt: ast.AsyncWith
) -> List[Region]:
    return lower_with_items(state, ctx, stmt.items, stmt.body, stmt, is_async=True)


def lower_with_items(
    state: CompilerState,
    ctx: RegionContext,
    items: List[ast.withitem],
    body: List[ast.stmt],
    owner: ast.AST,
    is_async: bool = False,
) -> List[Region]:
    nested_regions = []
    # Lower nested with-items recursively so each item gets its own synthetic finally path.
    if not items:
        return lower_stmt_list(state, ctx, body)
    item = items[0]
    mgr = lower_expr(state, ctx, item.context_expr)
    exit_attr = "__aexit__" if is_async else "__exit__"
    enter_attr = "__aenter__" if is_async else "__enter__"
    exit_fn = fresh_temp(state)
    ctx.builder.emit(
        attach_meta(
            state,
            LoadAttr(dst=exit_fn, obj=mgr, attr_name=exit_attr),
            item.context_expr,
        )
    )
    enter_fn = fresh_temp(state)
    ctx.builder.emit(
        attach_meta(
            state,
            LoadAttr(dst=enter_fn, obj=mgr, attr_name=enter_attr),
            item.context_expr,
        )
    )
    if is_async:
        entered = call_and_await(state, ctx, enter_fn, [], item.context_expr)
    else:
        entered = fresh_temp(state)
        ctx.builder.emit(
            attach_meta(
                state,
                Call(
                    dst=entered,
                    callee=enter_fn,
                    args=normal_call_args(),
                    kwargs=normal_call_kwargs(),
                ),
                item.context_expr,
            )
        )
    if item.optional_vars is not None:
        assign_target(state, ctx, item.optional_vars, entered)
    finally_label = ctx.builder.new_label()
    normal_exit_label = ctx.builder.new_label()
    exceptional_exit_label = ctx.builder.new_label()
    suppress_label = ctx.builder.new_label()
    propagate_label = ctx.builder.new_label()
    after_label = ctx.builder.new_label()
    ctx.builder.emit(attach_meta(state, PushTry(finally_label=finally_label), owner))
    cleanup = CleanupContext(
        kind=CleanupKind.WITH,
        owner=owner,
        pop_count=1,
        exit_fn=exit_fn,
        is_async=is_async,
    )
    state.finally_stack.append(cleanup)
    nested_regions.extend(
        lower_with_items(state, ctx, items[1:], body, owner, is_async=is_async)
    )
    if ctx.builder.is_open():
        ctx.builder.emit(attach_meta(state, PopTry(), owner))
        ctx.builder.terminate(attach_meta(state, Jump(target=finally_label), owner))
    state.finally_stack.pop()
    nested_regions.extend(process_cleanup_exits(state, ctx, cleanup))
    ctx.builder.start_block(finally_label)
    current_exc = current_exception_value(state, ctx, owner)
    none_exc = const_value(state, ctx, None, owner)
    is_none = fresh_temp(state)
    ctx.builder.emit(
        attach_meta(
            state,
            CompareOp(
                dst=is_none,
                cmp=ComparisonOperator.IS,
                lhs=current_exc,
                rhs=none_exc,
            ),
            owner,
        )
    )
    ctx.builder.terminate(
        attach_meta(
            state,
            Branch(
                cond=is_none,
                true_label=normal_exit_label,
                false_label=exceptional_exit_label,
            ),
            owner,
        )
    )
    ctx.builder.start_block(normal_exit_label)
    none1 = const_value(state, ctx, None, owner)
    none2 = const_value(state, ctx, None, owner)
    none3 = const_value(state, ctx, None, owner)
    if is_async:
        ignored = call_and_await(state, ctx, exit_fn, [none1, none2, none3], owner)
    else:
        ignored = fresh_temp(state)
        ctx.builder.emit(
            attach_meta(
                state,
                Call(
                    dst=ignored,
                    callee=exit_fn,
                    args=normal_call_args([none1, none2, none3]),
                    kwargs=normal_call_kwargs(),
                ),
                owner,
            )
        )
    ctx.builder.emit(attach_meta(state, EndFinally(), owner))
    if ctx.builder.is_open():
        ctx.builder.terminate(attach_meta(state, Jump(target=after_label), owner))
    ctx.builder.start_block(exceptional_exit_label)
    type_name = builtin_const_value(state, ctx, builtins.type, owner)
    exc_type = fresh_temp(state)
    ctx.builder.emit(
        attach_meta(
            state,
            Call(
                dst=exc_type,
                callee=type_name,
                args=normal_call_args([current_exc]),
                kwargs=normal_call_kwargs(),
            ),
            owner,
        )
    )
    traceback = fresh_temp(state)
    ctx.builder.emit(
        attach_meta(
            state,
            LoadAttr(dst=traceback, obj=current_exc, attr_name="__traceback__"),
            owner,
        )
    )
    if is_async:
        exit_result = call_and_await(
            state, ctx, exit_fn, [exc_type, current_exc, traceback], owner
        )
    else:
        exit_result = fresh_temp(state)
        ctx.builder.emit(
            attach_meta(
                state,
                Call(
                    dst=exit_result,
                    callee=exit_fn,
                    args=normal_call_args([exc_type, current_exc, traceback]),
                    kwargs=normal_call_kwargs(),
                ),
                owner,
            )
        )
    ctx.builder.terminate(
        attach_meta(
            state,
            Branch(
                cond=exit_result, true_label=suppress_label, false_label=propagate_label
            ),
            owner,
        )
    )
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
    return nested_regions


def lower_try_common(
    state: CompilerState,
    ctx: RegionContext,
    stmt: ast.Try,
    kind: TryKind,
) -> List[Region]:
    nested_regions = []
    match_cls = (
        CheckEGMatch if kind == TryKind.EXCEPTION_GROUP else CheckExcMatch
    )
    # Try statements are represented with explicit synthetic try targets and CFG dispatch blocks.
    if not stmt.handlers and (not stmt.finalbody):
        raise UnsupportedFeature(
            stmt,
            "%s without except/finally is not implemented in AST lowering"
            % kind.value,
        )
    except_dispatch_label = ctx.builder.new_label() if stmt.handlers else None
    finally_label = ctx.builder.new_label() if stmt.finalbody else None
    after_label = ctx.builder.new_label()
    orelse_label = ctx.builder.new_label() if stmt.orelse else after_label
    if stmt.finalbody:
        ctx.builder.emit(attach_meta(state, PushTry(finally_label=finally_label), stmt))
    if stmt.handlers:
        ctx.builder.emit(
            attach_meta(state, PushTry(except_label=except_dispatch_label), stmt)
        )
    cleanup = None
    if stmt.finalbody:
        pop_count = (1 if stmt.finalbody else 0) + (1 if stmt.handlers else 0)
        cleanup = CleanupContext(
            kind=CleanupKind.TRY,
            owner=stmt,
            pop_count=pop_count,
            finalbody=stmt.finalbody,
        )
        state.finally_stack.append(cleanup)
    nested_regions.extend(lower_stmt_list(state, ctx, stmt.body))
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
        if cleanup is not None:
            # The except handler was popped before entering the else suite.
            cleanup.pop_count = 1
        nested_regions.extend(lower_stmt_list(state, ctx, stmt.orelse))
        if ctx.builder.is_open():
            if stmt.finalbody:
                ctx.builder.emit(attach_meta(state, PopTry(), stmt))
                ctx.builder.terminate(
                    attach_meta(state, Jump(target=finally_label), stmt)
                )
            else:
                ctx.builder.terminate(
                    attach_meta(state, Jump(target=after_label), stmt)
                )
    if stmt.handlers:
        ctx.builder.start_block(except_dispatch_label)
        if cleanup is not None:
            # Exception dispatch consumed the except handler; only the
            # surrounding finally handler remains active in handler bodies.
            cleanup.pop_count = 1
        current_exc = current_exception_value(state, ctx, stmt)
        no_match_label = ctx.builder.new_label()
        next_label = None
        for index, handler in enumerate(stmt.handlers):
            is_last = index == len(stmt.handlers) - 1
            body_label = ctx.builder.new_label()
            next_label = no_match_label if is_last else ctx.builder.new_label()
            if handler.type is None:
                ctx.builder.terminate(
                    attach_meta(state, Jump(target=body_label), handler)
                )
            else:
                typ = lower_expr(state, ctx, handler.type)
                match = fresh_temp(state)
                ctx.builder.emit(
                    attach_meta(
                        state,
                        match_cls(dst=match, exc=current_exc, typ=typ),
                        handler,
                    )
                )
                ctx.builder.terminate(
                    attach_meta(
                        state,
                        Branch(
                            cond=match, true_label=body_label, false_label=next_label
                        ),
                        handler,
                    )
                )
            ctx.builder.start_block(body_label)
            if handler.name:
                scope = scope_for_store(state, ctx, handler.name)
                ctx.builder.emit(
                    attach_meta(
                        state,
                        StoreName(src=current_exc, scope=scope, name=handler.name),
                        handler,
                    )
                )
            nested_regions.extend(lower_stmt_list(state, ctx, handler.body))
            if ctx.builder.is_open():
                if handler.name:
                    none_temp = const_value(state, ctx, None, handler)
                    scope = scope_for_store(state, ctx, handler.name)
                    ctx.builder.emit(
                        attach_meta(
                            state,
                            StoreName(src=none_temp, scope=scope, name=handler.name),
                            handler,
                        )
                    )
                    ctx.builder.emit(
                        attach_meta(
                            state, DeleteName(scope=scope, name=handler.name), handler
                        )
                    )
                ctx.builder.emit(attach_meta(state, ClearException(), handler))
                if stmt.finalbody:
                    ctx.builder.emit(attach_meta(state, PopTry(), handler))
                    ctx.builder.terminate(
                        attach_meta(state, Jump(target=finally_label), handler)
                    )
                else:
                    ctx.builder.terminate(
                        attach_meta(state, Jump(target=after_label), handler)
                    )
            if not is_last:
                ctx.builder.start_block(next_label)
        ctx.builder.start_block(no_match_label)
        ctx.builder.terminate(attach_meta(state, Reraise(), stmt))
    if cleanup is not None:
        state.finally_stack.pop()
        nested_regions.extend(process_cleanup_exits(state, ctx, cleanup))
    if stmt.finalbody:
        ctx.builder.start_block(finally_label)
        nested_regions.extend(lower_stmt_list(state, ctx, stmt.finalbody))
        if ctx.builder.is_open():
            ctx.builder.emit(attach_meta(state, EndFinally(), stmt))
            if ctx.builder.is_open():
                ctx.builder.terminate(
                    attach_meta(state, Jump(target=after_label), stmt)
                )
    ctx.builder.start_block(after_label)
    return nested_regions


def lower_try(
    state: CompilerState, ctx: RegionContext, stmt: ast.Try
) -> List[Region]:
    return lower_try_common(state, ctx, stmt, TryKind.NORMAL)


def lower_try_star(
    state: CompilerState, ctx: RegionContext, stmt: ast.Try
) -> List[Region]:
    return lower_try_common(state, ctx, stmt, TryKind.EXCEPTION_GROUP)


def bind_pattern_name(
    state: CompilerState,
    ctx: RegionContext,
    bindings: List[Tuple[Scope, str, TemporaryValue]],
    bound_names: List[Tuple[Scope, str]],
    name: Optional[str],
    value: TemporaryValue,
    node: ast.AST,
) -> None:
    if name is None:
        return
    scope = scope_for_store(state, ctx, name)
    # Pattern captures are deferred until the whole case (or, for ``|``
    # alternatives, the whole alternative) has matched.  ``bound_names`` is
    # used to clean up on guard failure.
    bindings.append((scope, name, value))
    bound_names.append((scope, name))


def emit_pattern_bindings(
    state: CompilerState,
    ctx: RegionContext,
    bindings: List[Tuple[Scope, str, TemporaryValue]],
    node: ast.AST,
) -> None:
    for scope, name, value in bindings:
        ctx.builder.emit(
            attach_meta(state, StoreName(src=value, scope=scope, name=name), node)
        )


def emit_delete_names(
    state: CompilerState,
    ctx: RegionContext,
    bound_names: List[Tuple[Scope, str]],
    target: BasicBlockLabel,
    node: ast.AST,
) -> None:
    """Emit tolerant deletes for guard-failure cleanup, then jump to *target*.

    Deletion is wrapped in a synthetic try/except so a name captured by a
    different ``|`` alternative (and therefore never actually bound on the
    failing path) does not abort the cleanup.
    """
    for scope, name in bound_names:
        missing_label = ctx.builder.new_label()
        next_label = ctx.builder.new_label()
        ctx.builder.emit(
            attach_meta(state, PushTry(except_label=missing_label), node)
        )
        ctx.builder.emit(
            attach_meta(state, DeleteName(scope=scope, name=name), node)
        )
        ctx.builder.emit(attach_meta(state, PopTry(), node))
        ctx.builder.terminate(attach_meta(state, Jump(target=next_label), node))
        ctx.builder.start_block(missing_label)
        ctx.builder.emit(attach_meta(state, ClearException(), node))
        ctx.builder.terminate(attach_meta(state, Jump(target=next_label), node))
        ctx.builder.start_block(next_label)
    ctx.builder.terminate(attach_meta(state, Jump(target=target), node))


def emit_call(
    state: CompilerState,
    ctx: RegionContext,
    callee: TemporaryValue,
    args: List[TemporaryValue],
    node: ast.AST,
) -> TemporaryValue:
    result = fresh_temp(state)
    ctx.builder.emit(
        attach_meta(
            state,
            Call(
                dst=result,
                callee=callee,
                args=normal_call_args(args),
                kwargs=normal_call_kwargs(),
            ),
            node,
        )
    )
    return result


def emit_builtin_call(
    state: CompilerState,
    ctx: RegionContext,
    builtin_obj: Any,
    args: List[TemporaryValue],
    node: ast.AST,
) -> TemporaryValue:
    callee = builtin_const_value(state, ctx, builtin_obj, node)
    return emit_call(state, ctx, callee, args, node)


def emit_pattern_length_check(
    state: CompilerState,
    ctx: RegionContext,
    subject: TemporaryValue,
    expected: int,
    allow_extra: bool,
    node: ast.AST,
) -> TemporaryValue:
    length = emit_builtin_call(state, ctx, builtins.len, [subject], node)
    wanted = const_value(state, ctx, expected, node)
    matched = fresh_temp(state)
    cmp = (
        ComparisonOperator.GREATER_THAN_OR_EQUAL
        if allow_extra
        else ComparisonOperator.EQUAL
    )
    ctx.builder.emit(
        attach_meta(
            state, CompareOp(dst=matched, cmp=cmp, lhs=length, rhs=wanted), node
        )
    )
    return matched


def lower_match(
    state: CompilerState, ctx: RegionContext, stmt: "ast.Match"
) -> List[Region]:
    nested_regions = []
    subject = lower_expr(state, ctx, stmt.subject)
    end_label = ctx.builder.new_label()
    for index, case in enumerate(stmt.cases):
        body_label = ctx.builder.new_label()
        failure_label = (
            end_label if index == len(stmt.cases) - 1 else ctx.builder.new_label()
        )
        bindings: List[Tuple[Scope, str, TemporaryValue]] = []
        bound_names: List[Tuple[Scope, str]] = []
        lower_pattern(
            state, ctx, case.pattern, subject, body_label, failure_label,
            bindings, bound_names,
        )
        ctx.builder.start_block(body_label)
        # The whole pattern matched; bind its captures now so a guard (if any)
        # can reference them, but the bindings can still be cleaned up if the
        # guard fails.
        emit_pattern_bindings(state, ctx, bindings, case.pattern)
        if case.guard is not None:
            guarded_body_label = ctx.builder.new_label()
            guard_failure_cleanup = ctx.builder.new_label()
            guard = lower_expr(state, ctx, case.guard)
            ctx.builder.terminate(
                attach_meta(
                    state,
                    Branch(
                        cond=guard,
                        true_label=guarded_body_label,
                        false_label=guard_failure_cleanup,
                    ),
                    case.guard,
                )
            )
            ctx.builder.start_block(guard_failure_cleanup)
            # Captures are bound before the guard is evaluated and remain
            # visible when a successful pattern's guard is false.
            ctx.builder.terminate(
                attach_meta(state, Jump(target=failure_label), case.guard)
            )
            ctx.builder.start_block(guarded_body_label)
        nested_regions.extend(lower_stmt_list(state, ctx, case.body))
        if ctx.builder.is_open():
            ctx.builder.terminate(
                attach_meta(state, Jump(target=end_label), case.pattern)
            )
        if failure_label is not end_label:
            ctx.builder.start_block(failure_label)
    ctx.builder.start_block(end_label)
    return nested_regions


def lower_pattern_values(
    state: CompilerState,
    ctx: RegionContext,
    patterns: List["ast.pattern"],
    values: List[TemporaryValue],
    success_label: BasicBlockLabel,
    failure_label: BasicBlockLabel,
    bindings: List[Tuple[Scope, str, TemporaryValue]],
    bound_names: List[Tuple[Scope, str]],
    node: ast.AST,
) -> None:
    if len(patterns) != len(values):
        raise UnsupportedFeature(node, "pattern arity mismatch during AST lowering")
    if not patterns:
        ctx.builder.terminate(attach_meta(state, Jump(target=success_label), node))
        return
    for index, (pattern, value) in enumerate(zip(patterns, values)):
        is_last = index == len(patterns) - 1
        next_label = success_label if is_last else ctx.builder.new_label()
        lower_pattern(
            state, ctx, pattern, value, next_label, failure_label,
            bindings, bound_names,
        )
        if not is_last:
            ctx.builder.start_block(next_label)


def lower_pattern(
    state: CompilerState,
    ctx: RegionContext,
    pattern: "ast.pattern",
    subject: TemporaryValue,
    success_label: BasicBlockLabel,
    failure_label: BasicBlockLabel,
    bindings: List[Tuple[Scope, str, TemporaryValue]],
    bound_names: List[Tuple[Scope, str]],
) -> None:
    if isinstance(pattern, ast.MatchAs):
        if pattern.pattern is None:
            bind_pattern_name(
                state, ctx, bindings, bound_names, pattern.name, subject, pattern
            )
            ctx.builder.terminate(
                attach_meta(state, Jump(target=success_label), pattern)
            )
            return
        matched_label = ctx.builder.new_label()
        lower_pattern(
            state, ctx, pattern.pattern, subject, matched_label, failure_label,
            bindings, bound_names,
        )
        ctx.builder.start_block(matched_label)
        bind_pattern_name(
            state, ctx, bindings, bound_names, pattern.name, subject, pattern
        )
        ctx.builder.terminate(attach_meta(state, Jump(target=success_label), pattern))
        return
    if isinstance(pattern, ast.MatchValue):
        wanted = lower_expr(state, ctx, pattern.value)
        matched = fresh_temp(state)
        ctx.builder.emit(
            attach_meta(
                state,
                CompareOp(
                    dst=matched,
                    cmp=ComparisonOperator.EQUAL,
                    lhs=subject,
                    rhs=wanted,
                ),
                pattern,
            )
        )
        ctx.builder.terminate(
            attach_meta(
                state,
                Branch(
                    cond=matched, true_label=success_label, false_label=failure_label
                ),
                pattern,
            )
        )
        return
    if isinstance(pattern, ast.MatchSingleton):
        wanted = const_value(state, ctx, pattern.value, pattern)
        matched = fresh_temp(state)
        ctx.builder.emit(
            attach_meta(
                state,
                CompareOp(
                    dst=matched,
                    cmp=ComparisonOperator.IS,
                    lhs=subject,
                    rhs=wanted,
                ),
                pattern,
            )
        )
        ctx.builder.terminate(
            attach_meta(
                state,
                Branch(
                    cond=matched, true_label=success_label, false_label=failure_label
                ),
                pattern,
            )
        )
        return
    if isinstance(pattern, ast.MatchOr):
        # Each alternative is a separate match path.  Defer its captures until
        # the whole alternative matches, then store them at its success edge so
        # a failed alternative leaves no bindings behind.
        for index, alternative in enumerate(pattern.patterns):
            next_failure = (
                failure_label
                if index == len(pattern.patterns) - 1
                else ctx.builder.new_label()
            )
            alt_success = ctx.builder.new_label()
            alt_bindings: List[Tuple[Scope, str, TemporaryValue]] = []
            lower_pattern(
                state, ctx, alternative, subject, alt_success, next_failure,
                alt_bindings, bound_names,
            )
            ctx.builder.start_block(alt_success)
            emit_pattern_bindings(state, ctx, alt_bindings, pattern)
            ctx.builder.terminate(
                attach_meta(state, Jump(target=success_label), pattern)
            )
            if next_failure is not failure_label:
                ctx.builder.start_block(next_failure)
        return
    if isinstance(pattern, ast.MatchSequence):
        sequence_ok = fresh_temp(state)
        ctx.builder.emit(
            attach_meta(state, MatchSequence(dst=sequence_ok, value=subject), pattern)
        )
        sequence_label = ctx.builder.new_label()
        ctx.builder.terminate(
            attach_meta(
                state,
                Branch(
                    cond=sequence_ok,
                    true_label=sequence_label,
                    false_label=failure_label,
                ),
                pattern,
            )
        )
        ctx.builder.start_block(sequence_label)
        star_indexes = [
            index
            for index, child in enumerate(pattern.patterns)
            if isinstance(child, ast.MatchStar)
        ]
        if len(star_indexes) > 1:
            raise UnsupportedFeature(
                pattern,
                "multiple starred sequence patterns are not implemented in AST lowering",
            )
        unpack_label = ctx.builder.new_label()
        if not star_indexes:
            length_ok = emit_pattern_length_check(
                state,
                ctx,
                subject,
                len(pattern.patterns),
                allow_extra=False,
                node=pattern,
            )
            ctx.builder.terminate(
                attach_meta(
                    state,
                    Branch(
                        cond=length_ok,
                        true_label=unpack_label,
                        false_label=failure_label,
                    ),
                    pattern,
                )
            )
            ctx.builder.start_block(unpack_label)
            values = [fresh_temp(state) for _ in pattern.patterns]
            ctx.builder.emit(
                attach_meta(state, Unpack(src=subject, dsts=COWList(values)), pattern)
            )
            lower_pattern_values(
                state,
                ctx,
                pattern.patterns,
                values,
                success_label,
                failure_label,
                bindings,
                bound_names,
                pattern,
            )
            return
        star_index = star_indexes[0]
        minimum_size = len(pattern.patterns) - 1
        length_ok = emit_pattern_length_check(
            state, ctx, subject, minimum_size, allow_extra=True, node=pattern
        )
        ctx.builder.terminate(
            attach_meta(
                state,
                Branch(
                    cond=length_ok, true_label=unpack_label, false_label=failure_label
                ),
                pattern,
            )
        )
        ctx.builder.start_block(unpack_label)
        before_values = [fresh_temp(state) for _ in pattern.patterns[:star_index]]
        star_value = fresh_temp(state)
        after_values = [fresh_temp(state) for _ in pattern.patterns[star_index + 1 :]]
        ctx.builder.emit(
            attach_meta(
                state,
                Unpack(
                    src=subject,
                    dsts=COWList(before_values + [star_value] + after_values),
                    star_index=star_index,
                ),
                pattern,
            )
        )
        values = before_values + [star_value] + after_values
        lower_pattern_values(
            state, ctx, pattern.patterns, values, success_label, failure_label,
            bindings, bound_names, pattern,
        )
        return
    if isinstance(pattern, ast.MatchStar):
        bind_pattern_name(
            state, ctx, bindings, bound_names, pattern.name, subject, pattern
        )
        ctx.builder.terminate(attach_meta(state, Jump(target=success_label), pattern))
        return
    if isinstance(pattern, ast.MatchMapping):
        if pattern.rest is not None:
            raise UnsupportedFeature(
                pattern, "mapping rest patterns are not implemented in AST lowering"
            )
        mapping_ok = fresh_temp(state)
        ctx.builder.emit(
            attach_meta(state, MatchMapping(dst=mapping_ok, value=subject), pattern)
        )
        mapping_label = ctx.builder.new_label()
        ctx.builder.terminate(
            attach_meta(
                state,
                Branch(
                    cond=mapping_ok, true_label=mapping_label, false_label=failure_label
                ),
                pattern,
            )
        )
        ctx.builder.start_block(mapping_label)
        if not pattern.keys:
            ctx.builder.terminate(
                attach_meta(state, Jump(target=success_label), pattern)
            )
            return
        key_values = [lower_expr(state, ctx, key) for key in pattern.keys]
        keys_tuple = build_tuple(state, ctx, key_values, pattern)
        matched_items = fresh_temp(state)
        ctx.builder.emit(
            attach_meta(
                state,
                MatchKeys(dst=matched_items, mapping=subject, keys=keys_tuple),
                pattern,
            )
        )
        none_value = const_value(state, ctx, None, pattern)
        found = fresh_temp(state)
        ctx.builder.emit(
            attach_meta(
                state,
                CompareOp(
                    dst=found,
                    cmp=ComparisonOperator.IS_NOT,
                    lhs=matched_items,
                    rhs=none_value,
                ),
                pattern,
            )
        )
        values_label = ctx.builder.new_label()
        ctx.builder.terminate(
            attach_meta(
                state,
                Branch(cond=found, true_label=values_label, false_label=failure_label),
                pattern,
            )
        )
        ctx.builder.start_block(values_label)
        values = [fresh_temp(state) for _ in pattern.patterns]
        ctx.builder.emit(
            attach_meta(state, Unpack(src=matched_items, dsts=COWList(values)), pattern)
        )
        lower_pattern_values(
            state, ctx, pattern.patterns, values, success_label, failure_label,
            bindings, bound_names, pattern,
        )
        return
    if isinstance(pattern, ast.MatchClass):
        cls = lower_expr(state, ctx, pattern.cls)
        matched = fresh_temp(state)
        ctx.builder.emit(
            attach_meta(
                state,
                MatchClass(
                    dst=matched,
                    value=subject,
                    cls=cls,
                    attr_names=COWList(pattern.kwd_attrs),
                    positional_count=len(pattern.patterns),
                ),
                pattern,
            )
        )
        attrs_label = ctx.builder.new_label()
        ctx.builder.terminate(
            attach_meta(
                state,
                Branch(cond=matched, true_label=attrs_label, false_label=failure_label),
                pattern,
            )
        )
        ctx.builder.start_block(attrs_label)
        values = []
        patterns = []
        if pattern.patterns:
            match_args = fresh_temp(state)
            ctx.builder.emit(
                attach_meta(
                    state,
                    LoadAttr(dst=match_args, obj=cls, attr_name="__match_args__"),
                    pattern,
                )
            )
            getattr_fn = builtin_const_value(state, ctx, builtins.getattr, pattern)
            for index, positional_pattern in enumerate(pattern.patterns):
                position = const_value(state, ctx, index, pattern)
                attr_name = fresh_temp(state)
                ctx.builder.emit(
                    attach_meta(
                        state,
                        LoadItem(dst=attr_name, obj=match_args, key=position),
                        pattern,
                    )
                )
                attr_value = emit_call(
                    state, ctx, getattr_fn, [subject, attr_name], pattern
                )
                values.append(attr_value)
                patterns.append(positional_pattern)
        for attr_name, keyword_pattern in zip(pattern.kwd_attrs, pattern.kwd_patterns):
            attr_value = fresh_temp(state)
            ctx.builder.emit(
                attach_meta(
                    state,
                    LoadAttr(dst=attr_value, obj=subject, attr_name=attr_name),
                    pattern,
                )
            )
            values.append(attr_value)
            patterns.append(keyword_pattern)
        if not patterns:
            ctx.builder.terminate(
                attach_meta(state, Jump(target=success_label), pattern)
            )
            return
        lower_pattern_values(
            state, ctx, patterns, values, success_label, failure_label,
            bindings, bound_names, pattern,
        )
        return
    raise UnsupportedFeature(
        pattern,
        "pattern %s is not implemented in AST lowering" % type(pattern).__name__,
    )


def lower_ifexp(
    state: CompilerState, ctx: RegionContext, expr: ast.IfExp
) -> TemporaryValue:
    result_name = synthetic_local_name(fresh_synthetic_local(state, SyntheticLocalPurpose.IFEXP_RESULT))
    then_label = ctx.builder.new_label()
    else_label = ctx.builder.new_label()
    end_label = ctx.builder.new_label()
    cond = lower_expr(state, ctx, expr.test)
    ctx.builder.terminate(
        attach_meta(
            state,
            Branch(cond=cond, true_label=then_label, false_label=else_label),
            expr.test,
        )
    )
    ctx.builder.start_block(then_label)
    then_value = lower_expr(state, ctx, expr.body)
    ctx.builder.emit(
        attach_meta(
            state,
            StoreName(src=then_value, scope=Scope.LOCAL, name=result_name),
            expr.body,
        )
    )
    ctx.builder.terminate(attach_meta(state, Jump(target=end_label), expr.body))
    ctx.builder.start_block(else_label)
    else_value = lower_expr(state, ctx, expr.orelse)
    ctx.builder.emit(
        attach_meta(
            state,
            StoreName(src=else_value, scope=Scope.LOCAL, name=result_name),
            expr.orelse,
        )
    )
    ctx.builder.terminate(attach_meta(state, Jump(target=end_label), expr.orelse))
    ctx.builder.start_block(end_label)
    result = fresh_temp(state)
    ctx.builder.emit(
        attach_meta(
            state, LoadName(dst=result, scope=Scope.LOCAL, name=result_name), expr
        )
    )
    ctx.builder.emit(
        attach_meta(state, DeleteName(scope=Scope.LOCAL, name=result_name), expr)
    )
    return result


def lower_bool_op(
    state: CompilerState, ctx: RegionContext, expr: ast.BoolOp
) -> TemporaryValue:
    if not expr.values:
        raise UnsupportedFeature(expr, "empty boolean operation is not valid")
    if len(expr.values) == 1:
        return lower_expr(state, ctx, expr.values[0])
    result_name = synthetic_local_name(fresh_synthetic_local(state, SyntheticLocalPurpose.BOOL_OP_RESULT))
    end_label = ctx.builder.new_label()
    is_and = isinstance(expr.op, ast.And)
    for value_expr in expr.values[:-1]:
        value = lower_expr(state, ctx, value_expr)
        ctx.builder.emit(
            attach_meta(
                state,
                StoreName(src=value, scope=Scope.LOCAL, name=result_name),
                value_expr,
            )
        )
        next_label = ctx.builder.new_label()
        if is_and:
            ctx.builder.terminate(
                attach_meta(
                    state,
                    Branch(cond=value, true_label=next_label, false_label=end_label),
                    value_expr,
                )
            )
        else:
            ctx.builder.terminate(
                attach_meta(
                    state,
                    Branch(cond=value, true_label=end_label, false_label=next_label),
                    value_expr,
                )
            )
        ctx.builder.start_block(next_label)
    last_value = lower_expr(state, ctx, expr.values[-1])
    ctx.builder.emit(
        attach_meta(
            state,
            StoreName(src=last_value, scope=Scope.LOCAL, name=result_name),
            expr.values[-1],
        )
    )
    ctx.builder.terminate(attach_meta(state, Jump(target=end_label), expr))
    ctx.builder.start_block(end_label)
    result = fresh_temp(state)
    ctx.builder.emit(
        attach_meta(
            state, LoadName(dst=result, scope=Scope.LOCAL, name=result_name), expr
        )
    )
    ctx.builder.emit(
        attach_meta(state, DeleteName(scope=Scope.LOCAL, name=result_name), expr)
    )
    return result


def emit_method_call(
    state: CompilerState,
    ctx: RegionContext,
    obj: TemporaryValue,
    method_name: str,
    args: List[TemporaryValue],
    node: ast.AST,
) -> TemporaryValue:
    method = fresh_temp(state)
    ctx.builder.emit(
        attach_meta(state, LoadAttr(dst=method, obj=obj, attr_name=method_name), node)
    )
    result = fresh_temp(state)
    ctx.builder.emit(
        attach_meta(
            state,
            Call(
                dst=result,
                callee=method,
                args=normal_call_args(args),
                kwargs=normal_call_kwargs(),
            ),
            node,
        )
    )
    return result


def comprehension_iter_name(
    state: CompilerState,
    ctx: RegionContext,
    generator: ast.comprehension,
    index: int,
) -> Tuple[str, bool]:
    # Generator-expression regions receive the outer iterator as an implicit first argument.
    if (
        index == 0
        and isinstance(ctx.node, ast.GeneratorExp)
        and ctx.code_obj.co_argcount
    ):
        return ctx.code_obj.co_varnames[0], False
    iterable = lower_expr(state, ctx, generator.iter)
    iter_temp = fresh_temp(state)
    if generator.is_async:
        ctx.builder.emit(
            attach_meta(
                state, GetAIter(dst=iter_temp, iterable=iterable), generator.iter
            )
        )
    else:
        ctx.builder.emit(
            attach_meta(
                state, GetIter(dst=iter_temp, iterable=iterable), generator.iter
            )
        )
    iter_name = synthetic_local_name(fresh_synthetic_local(state, SyntheticLocalPurpose.COMPREHENSION_ITER))
    ctx.builder.emit(
        attach_meta(
            state,
            StoreName(src=iter_temp, scope=Scope.LOCAL, name=iter_name),
            generator,
        )
    )
    return iter_name, True


def lower_comprehension(
    state: CompilerState,
    ctx: RegionContext,
    expr: ast.AST,
    generators: List[ast.comprehension],
    emit_item: Any,
    result: TemporaryValue,
) -> TemporaryValue:
    # Save any outer bindings shadowed by the comprehension targets, then restore them afterwards.
    if not generators:
        raise UnsupportedFeature(expr, "comprehension without generators is not valid")
    saved_names = snapshot_comprehension_target_names(state, ctx, generators, expr)
    after_label = ctx.builder.new_label()
    lower_comprehension_generator(
        state, ctx, generators, 0, emit_item, after_label, expr
    )
    ctx.builder.start_block(after_label)
    restore_comprehension_target_names(state, ctx, saved_names, expr)
    return result


def lower_comprehension_generator(
    state: CompilerState,
    ctx: RegionContext,
    generators: List[ast.comprehension],
    index: int,
    emit_item: Any,
    exhaustion_label: BasicBlockLabel,
    owner: ast.AST,
) -> None:
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
    ctx.builder.emit(
        attach_meta(
            state,
            LoadName(dst=current_iter, scope=Scope.LOCAL, name=iter_name),
            generator,
        )
    )
    if generator.is_async:
        next_awaitable = fresh_temp(state)
        ctx.builder.emit(
            attach_meta(
                state, GetANext(dst=next_awaitable, aiter=current_iter), generator
            )
        )
        ctx.builder.emit(
            attach_meta(state, PushTry(except_label=stop_label), generator)
        )
        value_dst = await_value(state, ctx, next_awaitable, generator)
        ctx.builder.emit(attach_meta(state, PopTry(), generator))
        ctx.builder.terminate(attach_meta(state, Jump(target=body_label), generator))
        ctx.builder.start_block(stop_label)
        current_exc = current_exception_value(state, ctx, generator)
        stop_type = builtin_const_value(
            state, ctx, builtins.StopAsyncIteration, generator
        )
        matched = fresh_temp(state)
        ctx.builder.emit(
            attach_meta(
                state,
                CheckExcMatch(dst=matched, exc=current_exc, typ=stop_type),
                generator,
            )
        )
        ctx.builder.terminate(
            attach_meta(
                state,
                Branch(
                    cond=matched,
                    true_label=stop_match_label,
                    false_label=stop_nomatch_label,
                ),
                generator,
            )
        )
        ctx.builder.start_block(stop_match_label)
        ctx.builder.emit(attach_meta(state, ClearException(), generator))
        ctx.builder.terminate(attach_meta(state, Jump(target=cleanup_label), generator))
        ctx.builder.start_block(stop_nomatch_label)
        ctx.builder.terminate(attach_meta(state, Reraise(), generator))
        ctx.builder.start_block(body_label)
    else:
        value_dst = fresh_temp(state)
        ctx.builder.terminate(
            attach_meta(
                state,
                ForIter(
                    iter_obj=current_iter,
                    value_dst=value_dst,
                    body_label=body_label,
                    exit_label=cleanup_label,
                ),
                generator,
            )
        )
        ctx.builder.start_block(body_label)
    assign_target(state, ctx, generator.target, value_dst)
    for if_expr in generator.ifs:
        next_label = ctx.builder.new_label()
        cond = lower_expr(state, ctx, if_expr)
        ctx.builder.terminate(
            attach_meta(
                state,
                Branch(cond=cond, true_label=next_label, false_label=header_label),
                if_expr,
            )
        )
        ctx.builder.start_block(next_label)
    if index + 1 < len(generators):
        lower_comprehension_generator(
            state, ctx, generators, index + 1, emit_item, header_label, owner
        )
    else:
        emit_item()
        if ctx.builder.is_open():
            ctx.builder.terminate(attach_meta(state, Jump(target=header_label), owner))
    ctx.builder.start_block(cleanup_label)
    if owns_iter_name:
        ctx.builder.emit(
            attach_meta(state, DeleteName(scope=Scope.LOCAL, name=iter_name), generator)
        )
    ctx.builder.terminate(attach_meta(state, Jump(target=exhaustion_label), generator))


def lower_list_comp(
    state: CompilerState, ctx: RegionContext, expr: ast.ListComp
) -> TemporaryValue:
    result = fresh_temp(state)
    ctx.builder.emit(attach_meta(state, BuildList(dst=result, items=COWList()), expr))

    def emit_item() -> None:
        item = lower_expr(state, ctx, expr.elt)
        emit_method_call(state, ctx, result, "append", [item], expr)

    return lower_comprehension(state, ctx, expr, expr.generators, emit_item, result)


def lower_set_comp(
    state: CompilerState, ctx: RegionContext, expr: ast.SetComp
) -> TemporaryValue:
    result = fresh_temp(state)
    ctx.builder.emit(attach_meta(state, BuildSet(dst=result, items=COWList()), expr))

    def emit_item() -> None:
        item = lower_expr(state, ctx, expr.elt)
        emit_method_call(state, ctx, result, "add", [item], expr)

    return lower_comprehension(state, ctx, expr, expr.generators, emit_item, result)


def lower_dict_comp(
    state: CompilerState, ctx: RegionContext, expr: ast.DictComp
) -> TemporaryValue:
    result = fresh_temp(state)
    ctx.builder.emit(attach_meta(state, BuildMap(dst=result, items=COWList()), expr))

    def emit_item() -> None:
        key = lower_expr(state, ctx, expr.key)
        value = lower_expr(state, ctx, expr.value)
        ctx.builder.emit(
            attach_meta(state, StoreItem(obj=result, key=key, value=value), expr)
        )

    return lower_comprehension(state, ctx, expr, expr.generators, emit_item, result)


def lower_generator_exp(
    state: CompilerState, ctx: RegionContext, expr: ast.GeneratorExp
) -> TemporaryValue:
    # Generator expressions lower to an explicit nested region plus a call that seeds the outer iterator.
    child_table, child_code = take_child_region_inputs(
        state,
        ctx,
        table_type=ChildRegionType.FUNCTION,
        symtable_name="genexpr",
        code_name="<genexpr>",
        owner=expr,
    )
    child_label = fresh_child_region_label(ctx)
    child_name = child_region_name(state, "<genexpr>")
    child_path = child_name_path(state, ctx, "<genexpr>", for_class=False)
    nested_region = compile_region_node(
        state,
        node=expr,
        table=child_table,
        code_obj=child_code,
        name=child_name,
        name_path=child_path,
        is_class=False,
        label=child_label,
    )
    state.region_nested_stacks[-1].append(nested_region)
    outer_iterable = lower_expr(state, ctx, expr.generators[0].iter)
    outer_iter = fresh_temp(state)
    if expr.generators[0].is_async:
        ctx.builder.emit(
            attach_meta(
                state,
                GetAIter(dst=outer_iter, iterable=outer_iterable),
                expr.generators[0].iter,
            )
        )
    else:
        ctx.builder.emit(
            attach_meta(
                state,
                GetIter(dst=outer_iter, iterable=outer_iterable),
                expr.generators[0].iter,
            )
        )
    func = fresh_temp(state)
    ctx.builder.emit(
        attach_meta(
            state,
            MakeFunction(
                dst=func, code=child_label, flags=CodeFlag(child_code.co_flags)
            ),
            expr,
        )
    )
    call = fresh_temp(state)
    ctx.builder.emit(
        attach_meta(
            state,
            Call(
                dst=call,
                callee=func,
                args=normal_call_args([outer_iter]),
                kwargs=normal_call_kwargs(),
            ),
            expr,
        )
    )
    return call


def lower_lambda(
    state: CompilerState, ctx: RegionContext, expr: ast.Lambda
) -> TemporaryValue:
    child_table, child_code = take_child_region_inputs(
        state,
        ctx,
        table_type=ChildRegionType.FUNCTION,
        symtable_name="lambda",
        code_name="<lambda>",
        owner=expr,
    )
    child_label = fresh_child_region_label(ctx)
    child_name = child_region_name(state, "<lambda>")
    child_path = child_name_path(state, ctx, "<lambda>", for_class=False)
    nested_region = compile_region_node(
        state,
        node=expr,
        table=child_table,
        code_obj=child_code,
        name=child_name,
        name_path=child_path,
        is_class=False,
        label=child_label,
    )
    state.region_nested_stacks[-1].append(nested_region)
    default_values = COWList(
        [lower_expr(state, ctx, value) for value in expr.args.defaults]
    )
    kwonly_items = []
    for arg, default in zip(expr.args.kwonlyargs, expr.args.kw_defaults):
        if default is None:
            continue
        kwonly_items.append((arg.arg, lower_expr(state, ctx, default)))
    temp = fresh_temp(state)
    ctx.builder.emit(
        attach_meta(
            state,
            MakeFunction(
                dst=temp,
                code=child_label,
                defaults=default_values,
                kwdefaults=COWList(kwonly_items),
                flags=CodeFlag(child_code.co_flags),
            ),
            expr,
        )
    )
    return temp


def snapshot_comprehension_target_names(
    state: CompilerState,
    ctx: RegionContext,
    generators: List[ast.comprehension],
    owner: ast.AST,
) -> List[Tuple[Scope, str, str, str]]:
    # Preserve any outer names shadowed by comprehension targets so the enclosing scope sees no leak.
    saved = []
    names = sorted(
        {
            name
            for generator in generators
            for name in target_names(state, generator.target)
        }
    )
    for name in names:
        scope = scope_for_store(state, ctx, name)
        present_name = synthetic_local_name(fresh_synthetic_local(state, SyntheticLocalPurpose.SAVED_PRESENT))
        value_name = synthetic_local_name(fresh_synthetic_local(state, SyntheticLocalPurpose.SAVED_VALUE))
        missing_label = ctx.builder.new_label()
        after_label = ctx.builder.new_label()
        ctx.builder.emit(attach_meta(state, PushTry(except_label=missing_label), owner))
        loaded = fresh_temp(state)
        ctx.builder.emit(
            attach_meta(state, LoadName(dst=loaded, scope=scope, name=name), owner)
        )
        ctx.builder.emit(attach_meta(state, PopTry(), owner))
        present_true = const_value(state, ctx, True, owner)
        ctx.builder.emit(
            attach_meta(
                state,
                StoreName(src=present_true, scope=Scope.LOCAL, name=present_name),
                owner,
            )
        )
        ctx.builder.emit(
            attach_meta(
                state, StoreName(src=loaded, scope=Scope.LOCAL, name=value_name), owner
            )
        )
        ctx.builder.terminate(attach_meta(state, Jump(target=after_label), owner))
        ctx.builder.start_block(missing_label)
        ctx.builder.emit(attach_meta(state, ClearException(), owner))
        present_false = const_value(state, ctx, False, owner)
        missing_value = const_value(state, ctx, None, owner)
        ctx.builder.emit(
            attach_meta(
                state,
                StoreName(src=present_false, scope=Scope.LOCAL, name=present_name),
                owner,
            )
        )
        ctx.builder.emit(
            attach_meta(
                state,
                StoreName(src=missing_value, scope=Scope.LOCAL, name=value_name),
                owner,
            )
        )
        ctx.builder.terminate(attach_meta(state, Jump(target=after_label), owner))
        ctx.builder.start_block(after_label)
        saved.append((scope, name, present_name, value_name))
    return saved


def restore_comprehension_target_names(
    state: CompilerState,
    ctx: RegionContext,
    saved_names: List[Tuple[Scope, str, str, str]],
    owner: ast.AST,
) -> None:
    # Restore or delete the saved outer bindings after the comprehension region finishes.
    for scope, name, present_name, value_name in saved_names:
        present_temp = fresh_temp(state)
        value_temp = fresh_temp(state)
        ctx.builder.emit(
            attach_meta(
                state,
                LoadName(dst=present_temp, scope=Scope.LOCAL, name=present_name),
                owner,
            )
        )
        ctx.builder.emit(
            attach_meta(
                state,
                LoadName(dst=value_temp, scope=Scope.LOCAL, name=value_name),
                owner,
            )
        )
        restore_label = ctx.builder.new_label()
        delete_label = ctx.builder.new_label()
        next_label = ctx.builder.new_label()
        delete_missing_label = ctx.builder.new_label()
        ctx.builder.terminate(
            attach_meta(
                state,
                Branch(
                    cond=present_temp,
                    true_label=restore_label,
                    false_label=delete_label,
                ),
                owner,
            )
        )
        ctx.builder.start_block(restore_label)
        ctx.builder.emit(
            attach_meta(state, StoreName(src=value_temp, scope=scope, name=name), owner)
        )
        ctx.builder.terminate(attach_meta(state, Jump(target=next_label), owner))
        ctx.builder.start_block(delete_label)
        ctx.builder.emit(
            attach_meta(state, PushTry(except_label=delete_missing_label), owner)
        )
        ctx.builder.emit(attach_meta(state, DeleteName(scope=scope, name=name), owner))
        ctx.builder.emit(attach_meta(state, PopTry(), owner))
        ctx.builder.terminate(attach_meta(state, Jump(target=next_label), owner))
        ctx.builder.start_block(delete_missing_label)
        ctx.builder.emit(attach_meta(state, ClearException(), owner))
        ctx.builder.terminate(attach_meta(state, Jump(target=next_label), owner))
        ctx.builder.start_block(next_label)
        ctx.builder.emit(
            attach_meta(state, DeleteName(scope=Scope.LOCAL, name=present_name), owner)
        )
        ctx.builder.emit(
            attach_meta(state, DeleteName(scope=Scope.LOCAL, name=value_name), owner)
        )


def target_names(
    state: CompilerState, target: ast.AST
) -> List[str]:
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


def lower_container_items(
    state: CompilerState, ctx: RegionContext, elts: List[ast.AST]
) -> List[Any]:
    items = []
    for elt in elts:
        if isinstance(elt, ast.Starred):
            items.append(UnpackedTemporaryValue(lower_expr(state, ctx, elt.value)))
        else:
            items.append(lower_expr(state, ctx, elt))
    return items


def lower_compare_expr(
    state: CompilerState, ctx: RegionContext, expr: ast.Compare
) -> TemporaryValue:
    if len(expr.ops) == 1 and len(expr.comparators) == 1:
        lhs = lower_expr(state, ctx, expr.left)
        rhs = lower_expr(state, ctx, expr.comparators[0])
        temp = fresh_temp(state)
        ctx.builder.emit(
            attach_meta(
                state,
                CompareOp(
                    dst=temp, cmp=compare_op(state, expr.ops[0]), lhs=lhs, rhs=rhs
                ),
                expr,
            )
        )
        return temp

    current_name = synthetic_local_name(fresh_synthetic_local(state, SyntheticLocalPurpose.COMPARE_CURRENT))
    result_name = synthetic_local_name(fresh_synthetic_local(state, SyntheticLocalPurpose.COMPARE_RESULT))
    false_label = ctx.builder.new_label()
    end_label = ctx.builder.new_label()

    first_value = lower_expr(state, ctx, expr.left)
    ctx.builder.emit(
        attach_meta(
            state,
            StoreName(src=first_value, scope=Scope.LOCAL, name=current_name),
            expr.left,
        )
    )

    for index, (op_node, rhs_expr) in enumerate(zip(expr.ops, expr.comparators)):
        lhs = fresh_temp(state)
        ctx.builder.emit(
            attach_meta(
                state, LoadName(dst=lhs, scope=Scope.LOCAL, name=current_name), rhs_expr
            )
        )
        rhs = lower_expr(state, ctx, rhs_expr)
        cmp_result = fresh_temp(state)
        ctx.builder.emit(
            attach_meta(
                state,
                CompareOp(
                    dst=cmp_result, cmp=compare_op(state, op_node), lhs=lhs, rhs=rhs
                ),
                rhs_expr,
            )
        )
        is_last = index == len(expr.ops) - 1
        if is_last:
            true_label = ctx.builder.new_label()
            ctx.builder.terminate(
                attach_meta(
                    state,
                    Branch(
                        cond=cmp_result, true_label=true_label, false_label=false_label
                    ),
                    rhs_expr,
                )
            )
            ctx.builder.start_block(true_label)
            true_value = const_value(state, ctx, True, rhs_expr)
            ctx.builder.emit(
                attach_meta(
                    state,
                    StoreName(src=true_value, scope=Scope.LOCAL, name=result_name),
                    rhs_expr,
                )
            )
            ctx.builder.terminate(attach_meta(state, Jump(target=end_label), rhs_expr))
        else:
            next_label = ctx.builder.new_label()
            ctx.builder.emit(
                attach_meta(
                    state,
                    StoreName(src=rhs, scope=Scope.LOCAL, name=current_name),
                    rhs_expr,
                )
            )
            ctx.builder.terminate(
                attach_meta(
                    state,
                    Branch(
                        cond=cmp_result, true_label=next_label, false_label=false_label
                    ),
                    rhs_expr,
                )
            )
            ctx.builder.start_block(next_label)

    ctx.builder.start_block(false_label)
    false_value = const_value(state, ctx, False, expr)
    ctx.builder.emit(
        attach_meta(
            state, StoreName(src=false_value, scope=Scope.LOCAL, name=result_name), expr
        )
    )
    ctx.builder.terminate(attach_meta(state, Jump(target=end_label), expr))

    ctx.builder.start_block(end_label)
    result = fresh_temp(state)
    ctx.builder.emit(
        attach_meta(
            state, LoadName(dst=result, scope=Scope.LOCAL, name=result_name), expr
        )
    )
    ctx.builder.emit(
        attach_meta(state, DeleteName(scope=Scope.LOCAL, name=current_name), expr)
    )
    ctx.builder.emit(
        attach_meta(state, DeleteName(scope=Scope.LOCAL, name=result_name), expr)
    )
    return result


def lower_expr(
    state: CompilerState, ctx: RegionContext, expr: ast.AST
) -> TemporaryValue:
    """Lower one expression and return the IR value holding its result."""
    if isinstance(expr, ast.Constant):
        return const_value(state, ctx, expr.value, expr)
    if NamedExprNode is not None and isinstance(expr, NamedExprNode):
        value = lower_expr(state, ctx, expr.value)
        if not isinstance(expr.target, ast.Name):
            raise UnsupportedFeature(
                expr.target,
                "named-expression target %s is not implemented in AST lowering"
                % type(expr.target).__name__,
            )
        scope = scope_for_store(state, ctx, expr.target.id)
        ctx.builder.emit(
            attach_meta(
                state,
                StoreName(src=value, scope=scope, name=expr.target.id),
                expr.target,
            )
        )
        return value
    if isinstance(expr, ast.Lambda):
        return lower_lambda(state, ctx, expr)
    if isinstance(expr, ast.Name):
        temp = fresh_temp(state)
        scope = scope_for_load(state, ctx, expr.id)
        ctx.builder.emit(
            attach_meta(state, LoadName(dst=temp, scope=scope, name=expr.id), expr)
        )
        return temp
    if isinstance(expr, ast.Attribute):
        obj = lower_expr(state, ctx, expr.value)
        temp = fresh_temp(state)
        ctx.builder.emit(
            attach_meta(state, LoadAttr(dst=temp, obj=obj, attr_name=expr.attr), expr)
        )
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
        ctx.builder.emit(
            attach_meta(
                state,
                Call(
                    dst=temp,
                    callee=callee,
                    args=COWList(args),
                    kwargs=COWList(kwargs),
                ),
                expr,
            )
        )
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
        ctx.builder.emit(
            attach_meta(state, BuildList(dst=temp, items=COWList(items)), expr)
        )
        return temp
    if isinstance(expr, ast.Dict):
        items = []
        for key, value in zip(expr.keys, expr.values):
            if key is None:
                items.append((None, lower_expr(state, ctx, value)))
            else:
                items.append(
                    (lower_expr(state, ctx, key), lower_expr(state, ctx, value))
                )
        temp = fresh_temp(state)
        ctx.builder.emit(
            attach_meta(state, BuildMap(dst=temp, items=COWList(items)), expr)
        )
        return temp
    if isinstance(expr, ast.ListComp):
        return lower_list_comp(state, ctx, expr)
    if isinstance(expr, ast.JoinedStr):
        parts = [lower_expr(state, ctx, value) for value in expr.values]
        temp = fresh_temp(state)
        ctx.builder.emit(
            attach_meta(state, BuildString(dst=temp, parts=COWList(parts)), expr)
        )
        return temp
    if isinstance(expr, ast.FormattedValue):
        value = lower_expr(state, ctx, expr.value)
        conversion = None
        if expr.conversion == ord("s"):
            conversion = FormatConversion.STR
        elif expr.conversion == ord("r"):
            conversion = FormatConversion.REPR
        elif expr.conversion == ord("a"):
            conversion = FormatConversion.ASCII
        elif expr.conversion not in (-1, None):
            raise UnsupportedFeature(
                expr,
                "formatted-value conversion %r is not implemented in AST lowering"
                % (expr.conversion,),
            )
        spec = (
            None
            if expr.format_spec is None
            else lower_expr(state, ctx, expr.format_spec)
        )
        temp = fresh_temp(state)
        ctx.builder.emit(
            attach_meta(
                state,
                FormatValue(dst=temp, value=value, conversion=conversion, spec=spec),
                expr,
            )
        )
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
        ctx.builder.emit(
            attach_meta(state, BuildSet(dst=temp, items=COWList(items)), expr)
        )
        return temp
    if isinstance(expr, ast.BinOp):
        lhs = lower_expr(state, ctx, expr.left)
        rhs = lower_expr(state, ctx, expr.right)
        temp = fresh_temp(state)
        ctx.builder.emit(
            attach_meta(
                state,
                BinaryOp(dst=temp, op=binary_op(state, expr.op), lhs=lhs, rhs=rhs),
                expr,
            )
        )
        return temp
    if isinstance(expr, ast.UnaryOp):
        src = lower_expr(state, ctx, expr.operand)
        temp = fresh_temp(state)
        ctx.builder.emit(
            attach_meta(
                state, UnaryOp(dst=temp, op=unary_op(state, expr.op), src=src), expr
            )
        )
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
        ctx.builder.emit(
            attach_meta(state, GetAwaitable(dst=awaitable, value=value), expr)
        )
        temp = fresh_temp(state)
        ctx.builder.emit(
            attach_meta(state, AwaitValue(dst=temp, value=awaitable), expr)
        )
        return temp
    raise UnsupportedFeature(
        expr, "expression %s is not implemented in AST lowering" % type(expr).__name__
    )


def current_exception_value(
    state: CompilerState, ctx: RegionContext, node: ast.AST
) -> TemporaryValue:
    temp = fresh_temp(state)
    ctx.builder.emit(attach_meta(state, CurrentException(dst=temp), node))
    return temp


def lower_optional_expr(
    state: CompilerState,
    ctx: RegionContext,
    expr: Optional[ast.AST],
    owner: ast.AST,
) -> TemporaryValue:
    if expr is None:
        return const_value(state, ctx, None, owner)
    return lower_expr(state, ctx, expr)


def lower_slice_expr(
    state: CompilerState, ctx: RegionContext, slice_node: ast.AST
) -> TemporaryValue:
    if isinstance(slice_node, ast.Slice):
        start = lower_optional_expr(state, ctx, slice_node.lower, slice_node)
        stop = lower_optional_expr(state, ctx, slice_node.upper, slice_node)
        step = (
            None if slice_node.step is None else lower_expr(state, ctx, slice_node.step)
        )
        temp = fresh_temp(state)
        ctx.builder.emit(
            attach_meta(
                state,
                BuildSlice(dst=temp, start=start, stop=stop, step=step),
                slice_node,
            )
        )
        return temp
    return lower_expr(state, ctx, slice_node)


def assign_target(
    state: CompilerState,
    ctx: RegionContext,
    target: ast.AST,
    value: TemporaryValue,
) -> None:
    if isinstance(target, ast.Name):
        scope = scope_for_store(state, ctx, target.id)
        ctx.builder.emit(
            attach_meta(
                state, StoreName(src=value, scope=scope, name=target.id), target
            )
        )
        return
    if isinstance(target, ast.Attribute):
        obj = lower_expr(state, ctx, target.value)
        ctx.builder.emit(
            attach_meta(
                state, StoreAttr(obj=obj, attr_name=target.attr, value=value), target
            )
        )
        return
    if isinstance(target, ast.Subscript):
        obj = lower_expr(state, ctx, target.value)
        key = lower_slice_expr(state, ctx, target.slice)
        ctx.builder.emit(
            attach_meta(state, StoreItem(obj=obj, key=key, value=value), target)
        )
        return
    if isinstance(target, (ast.Tuple, ast.List)):
        assign_sequence_target(state, ctx, target, value)
        return
    raise UnsupportedFeature(
        target,
        "assignment target %s is not implemented in AST lowering"
        % type(target).__name__,
    )


def assign_sequence_target(
    state: CompilerState,
    ctx: RegionContext,
    target: ast.AST,
    value: TemporaryValue,
) -> None:
    starred = [
        index for index, elt in enumerate(target.elts) if isinstance(elt, ast.Starred)
    ]
    if len(starred) > 1:
        raise UnsupportedFeature(
            target, "multiple starred assignment targets are not implemented"
        )
    if not starred:
        dsts = [fresh_temp(state) for _ in target.elts]
        ctx.builder.emit(
            attach_meta(state, Unpack(src=value, dsts=COWList(dsts)), target)
        )
        for child, dst in zip(target.elts, dsts):
            assign_target(state, ctx, child, dst)
        return
    star_index = starred[0]
    before_dsts = [fresh_temp(state) for _ in target.elts[:star_index]]
    star_dst = fresh_temp(state)
    after_dsts = [fresh_temp(state) for _ in target.elts[star_index + 1 :]]
    ctx.builder.emit(
        attach_meta(
            state,
            Unpack(
                src=value,
                dsts=COWList(before_dsts + [star_dst] + after_dsts),
                star_index=star_index,
            ),
            target,
        )
    )
    for child, dst in zip(target.elts[:star_index], before_dsts):
        assign_target(state, ctx, child, dst)
    assign_target(state, ctx, target.elts[star_index].value, star_dst)
    for child, dst in zip(target.elts[star_index + 1 :], after_dsts):
        assign_target(state, ctx, child, dst)


def delete_target(
    state: CompilerState, ctx: RegionContext, target: ast.AST
) -> None:
    if isinstance(target, ast.Name):
        scope = scope_for_store(state, ctx, target.id)
        ctx.builder.emit(
            attach_meta(state, DeleteName(scope=scope, name=target.id), target)
        )
        return
    if isinstance(target, ast.Attribute):
        obj = lower_expr(state, ctx, target.value)
        ctx.builder.emit(
            attach_meta(state, DeleteAttr(obj=obj, attr_name=target.attr), target)
        )
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
    raise UnsupportedFeature(
        target,
        "delete target %s is not implemented in AST lowering" % type(target).__name__,
    )


def symbol_table_region_type(table: Any) -> ChildRegionType:
    """Normalize version-specific ``symtable`` kinds to an internal enum."""
    raw_type = table.get_type()
    raw_value = getattr(raw_type, "value", raw_type)
    if raw_value in ("type parameter", "type parameters"):
        return ChildRegionType.TYPE_PARAMETERS
    return ChildRegionType(raw_value)


def take_child_region_inputs(
    state: CompilerState,
    ctx: RegionContext,
    table_type: ChildRegionType,
    symtable_name: str,
    code_name: str,
    owner: ast.AST,
) -> Tuple[Any, types.CodeType]:
    """Find the next nested symbol-table child and code object for a region.

    AST lowering needs both pieces of metadata for nested executable regions such as
    functions, classes, and comprehension/genexpr bodies:

    - the `symtable` child provides scope information
    - the compiled child code object provides flags, locals, cells, and freevars

    Matching prefers an unused child with the same source line when available,
    and otherwise falls back to the first unused child with the requested kind
    and name. This keeps lowering robust even when CFG emission order differs
    from textual order (e.g. try/except/else lowering).
    """
    # PEP 695: a generic class/function/alias is wrapped in an intermediate
    # "type parameters" scope (symtable) and a "<generic parameters of X>" code
    # object.  Descend through that wrapper to reach the real body scope/code.
    if getattr(owner, "type_params", None):
        return take_generic_child_region_inputs(
            state, ctx, table_type, symtable_name, code_name, owner
        )

    owner_lineno = getattr(owner, "lineno", None)

    table_index = None
    for index, candidate in enumerate(ctx.child_tables):
        if index in ctx.used_child_tables:
            continue
        if (
            symbol_table_region_type(candidate) != table_type
            or candidate.get_name() != symtable_name
        ):
            continue
        candidate_lineno = (
            candidate.get_lineno() if hasattr(candidate, "get_lineno") else None
        )
        if owner_lineno is not None and candidate_lineno == owner_lineno:
            table_index = index
            break
        if table_index is None:
            table_index = index
    if table_index is None:
        raise UnsupportedFeature(owner, "missing nested symbol-table child for region")
    ctx.used_child_tables.add(table_index)
    table = ctx.child_tables[table_index]

    code_index = None
    for index, candidate in enumerate(ctx.child_codes):
        if index in ctx.used_child_codes:
            continue
        if candidate.co_name != code_name:
            continue
        if (
            owner_lineno is not None
            and getattr(candidate, "co_firstlineno", None) == owner_lineno
        ):
            code_index = index
            break
        if code_index is None:
            code_index = index
    if code_index is None:
        raise UnsupportedFeature(owner, "missing nested code object for region")
    ctx.used_child_codes.add(code_index)
    code_obj = ctx.child_codes[code_index]
    return (table, code_obj)


def take_generic_child_region_inputs(
    state: CompilerState,
    ctx: RegionContext,
    table_type: ChildRegionType,
    symtable_name: str,
    code_name: str,
    owner: ast.AST,
) -> Tuple[Any, types.CodeType]:
    """Resolve the body scope/code of a PEP 695 generic definition.

    The enclosing scope's direct child is a "type parameters" wrapper (both in
    the symtable and as a ``<generic parameters of X>`` code object); the real
    class/function/alias body lives one level inside it.
    """
    wrapper_code_name = "<generic parameters of %s>" % symtable_name

    table_index = None
    for index, candidate in enumerate(ctx.child_tables):
        if index in ctx.used_child_tables:
            continue
        if (
            symbol_table_region_type(candidate) == ChildRegionType.TYPE_PARAMETERS
            and candidate.get_name() == symtable_name
        ):
            table_index = index
            break
    if table_index is None:
        raise UnsupportedFeature(owner, "missing type-parameter symbol table")
    ctx.used_child_tables.add(table_index)
    wrapper_table = ctx.child_tables[table_index]
    table = None
    for child in wrapper_table.get_children():
        if (
            symbol_table_region_type(child) == table_type
            and child.get_name() == symtable_name
        ):
            table = child
            break
    if table is None:
        raise UnsupportedFeature(owner, "missing nested symbol-table child for region")

    code_index = None
    for index, candidate in enumerate(ctx.child_codes):
        if index in ctx.used_child_codes:
            continue
        if candidate.co_name == wrapper_code_name:
            code_index = index
            break
    if code_index is None:
        raise UnsupportedFeature(owner, "missing type-parameter code object")
    ctx.used_child_codes.add(code_index)
    wrapper_code = ctx.child_codes[code_index]
    code_obj = None
    for const in wrapper_code.co_consts:
        if isinstance(const, types.CodeType) and const.co_name == code_name:
            code_obj = const
            break
    if code_obj is None:
        raise UnsupportedFeature(owner, "missing nested code object for region")
    return (table, code_obj)


def type_param_kind(type_param: ast.AST) -> TypeParamKind:
    """Map a PEP 695 AST node to its structured IR kind."""
    if TypeVarNode is not None and isinstance(type_param, TypeVarNode):
        return TypeParamKind.TYPE_VAR
    if ParamSpecNode is not None and isinstance(type_param, ParamSpecNode):
        return TypeParamKind.PARAM_SPEC
    if TypeVarTupleNode is not None and isinstance(type_param, TypeVarTupleNode):
        return TypeParamKind.TYPE_VAR_TUPLE
    raise UnsupportedFeature(type_param, "unknown type-parameter kind")


def lower_type_params(
    state: CompilerState, parent_ctx: RegionContext, owner: ast.AST
) -> Tuple[List[TypeParam], List[Region]]:
    """Lower a definition's PEP 695 type parameters.

    Bounds/constraints and defaults are lowered into lazy nested regions in the
    enclosing scope (where the type-parameter scope evaluates them).  Returns
    the ``TypeParam`` descriptors and the regions they reference.
    """
    params = []
    regions = []
    for tp in getattr(owner, "type_params", None) or []:
        bound_label = None
        bound = getattr(tp, "bound", None)
        if bound is not None:
            bound_label, bound_region = lower_expression_region(
                state, parent_ctx, bound, "<type-param-bound>"
            )
            regions.append(bound_region)
        default_label = None
        default = getattr(tp, "default_value", None)
        if default is not None:
            default_label, default_region = lower_expression_region(
                state, parent_ctx, default, "<type-param-default>"
            )
            regions.append(default_region)
        params.append(
            TypeParam(
                name=tp.name,
                kind=type_param_kind(tp),
                bound=bound_label,
                default=default_label,
            )
        )
    return params, regions


def child_region_name(state: CompilerState, base_name: str) -> str:
    counts = state.synthetic_region_name_stacks[-1]
    count = counts.get(base_name, 0) + 1
    counts[base_name] = count
    if count == 1:
        return base_name
    return "%s#%d" % (base_name, count)


def child_name_path(
    state: CompilerState,
    parent_ctx: RegionContext,
    child_name: str,
    for_class: bool,
) -> COWList:
    parent_path = [item for item in parent_ctx.name_path if item != "<module>"]
    if not parent_path:
        return COWList([child_name])
    if for_class or parent_ctx.is_class:
        return COWList(parent_path + [child_name])
    return COWList(parent_path + ["<locals>", child_name])


def scope_for_load(
    state: CompilerState, ctx: RegionContext, name: str
) -> Scope:
    return scope_for_name(state, ctx, name)


def scope_for_store(
    state: CompilerState, ctx: RegionContext, name: str
) -> Scope:
    return scope_for_name(state, ctx, name)


def scope_for_name(
    state: CompilerState, ctx: RegionContext, name: str
) -> Scope:
    # Reconstruct Python's local/global/name/cell addressing mode for this symbol.
    if ctx.is_class:
        symbol = lookup_symbol(state, ctx.table, name)
        if symbol is not None and symbol.is_declared_global():
            return Scope.GLOBAL
        return Scope.NAME
    if ctx.name == "<module>":
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


def lookup_symbol(
    state: CompilerState, table: Any, name: str
) -> Optional[Any]:
    if name not in table.get_identifiers():
        return None
    return table.lookup(name)


def region_variadic_names(
    node: ast.AST,
) -> Tuple[Optional[str], Optional[str]]:
    if not hasattr(node, "args"):
        return (None, None)
    vararg = None if node.args.vararg is None else node.args.vararg.arg
    kwarg = None if node.args.kwarg is None else node.args.kwarg.arg
    return (vararg, kwarg)


def const_value(
    state: CompilerState, ctx: RegionContext, value: Any, node: ast.AST
) -> TemporaryValue:
    temp = fresh_temp(state)
    ctx.builder.emit(attach_meta(state, Const(dst=temp, value=value), node))
    return temp


def builtin_const_value(
    state: CompilerState, ctx: RegionContext, value: Any, node: ast.AST
) -> TemporaryValue:
    return const_value(state, ctx, value, node)


def normal_call_args(
    args: Sequence[Any] = (),
) -> COWList:
    return COWList(args)


def normal_call_kwargs(
    kwargs: Sequence[Tuple[Optional[str], TemporaryValue]] = (),
) -> COWList:
    return COWList(kwargs)


def build_tuple(
    state: CompilerState,
    ctx: RegionContext,
    items: List[Any],
    node: ast.AST,
) -> TemporaryValue:
    temp = fresh_temp(state)
    ctx.builder.emit(
        attach_meta(state, BuildTuple(dst=temp, items=COWList(items)), node)
    )
    return temp


def emit_return_none(
    state: CompilerState, builder: BlockBuilder, node: ast.AST
) -> None:
    temp = fresh_temp(state)
    builder.emit(attach_meta(state, Const(dst=temp, value=None), node))
    builder.emit(attach_meta(state, Return(value=temp), node))


def attach_meta(state: CompilerState, instruction: Any, node: ast.AST) -> Any:
    # Keep source spans as optional metadata so the executable IR stays simple.
    span = SourceSpan(
        lineno=getattr(node, "lineno", None),
        end_lineno=getattr(node, "end_lineno", None),
        col_offset=getattr(node, "col_offset", None),
        end_col_offset=getattr(node, "end_col_offset", None),
    )
    return attrs.evolve(instruction, span=span)


def binary_op(state: CompilerState, op: ast.AST) -> BinaryOperator:
    mapping = {
        ast.Add: BinaryOperator.ADD,
        ast.Sub: BinaryOperator.SUBTRACT,
        ast.Mult: BinaryOperator.MULTIPLY,
        ast.Div: BinaryOperator.TRUE_DIVIDE,
        ast.FloorDiv: BinaryOperator.FLOOR_DIVIDE,
        ast.Mod: BinaryOperator.MODULO,
        ast.Pow: BinaryOperator.POWER,
        ast.LShift: BinaryOperator.LEFT_SHIFT,
        ast.RShift: BinaryOperator.RIGHT_SHIFT,
        ast.BitAnd: BinaryOperator.BITWISE_AND,
        ast.BitOr: BinaryOperator.BITWISE_OR,
        ast.BitXor: BinaryOperator.BITWISE_XOR,
        ast.MatMult: BinaryOperator.MATRIX_MULTIPLY,
    }
    for cls, name in mapping.items():
        if isinstance(op, cls):
            return name
    raise UnsupportedFeature(
        op, "binary operator %s is not implemented in AST lowering" % type(op).__name__
    )


def unary_op(state: CompilerState, op: ast.AST) -> UnaryOperator:
    mapping = {
        ast.UAdd: UnaryOperator.POSITIVE,
        ast.USub: UnaryOperator.NEGATIVE,
        ast.Not: UnaryOperator.NOT,
        ast.Invert: UnaryOperator.INVERT,
    }
    for cls, name in mapping.items():
        if isinstance(op, cls):
            return name
    raise UnsupportedFeature(
        op, "unary operator %s is not implemented in AST lowering" % type(op).__name__
    )


def compare_op(state: CompilerState, op: ast.AST) -> ComparisonOperator:
    mapping = {
        ast.Lt: ComparisonOperator.LESS_THAN,
        ast.LtE: ComparisonOperator.LESS_THAN_OR_EQUAL,
        ast.Eq: ComparisonOperator.EQUAL,
        ast.NotEq: ComparisonOperator.NOT_EQUAL,
        ast.Gt: ComparisonOperator.GREATER_THAN,
        ast.GtE: ComparisonOperator.GREATER_THAN_OR_EQUAL,
        ast.Is: ComparisonOperator.IS,
        ast.IsNot: ComparisonOperator.IS_NOT,
        ast.In: ComparisonOperator.IN,
        ast.NotIn: ComparisonOperator.NOT_IN,
    }
    for cls, name in mapping.items():
        if isinstance(op, cls):
            return name
    raise UnsupportedFeature(
        op, "compare operator %s is not implemented in AST lowering" % type(op).__name__
    )
