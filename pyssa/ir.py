# Copyright (c) 2026 Jifeng Wu
# Licensed under the Apache-2.0 License. See LICENSE file in the project root for full license information.
"""Core IR instruction and operand types, plus pretty-printer.

This module defines all IR instruction classes, operand types, CFG
containers (BasicBlock, Region), and the rendering helpers for
debugging and visualization.
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import attrs
from cowlist import COWList

# ---------------------------------------------------------------------------
# Variable / operand types
# ---------------------------------------------------------------------------


class Scope(str, Enum):
    """Variable addressing modes used by the frontend and interpreter."""

    LOCAL = "local"
    GLOBAL = "global"
    NAME = "name"
    CELL = "cell"


@attrs.define(frozen=True)
class Operand:
    """Base class for all IR operands."""

    pass


@attrs.define(frozen=True)
class TemporaryValue(Operand):
    """An SSA-style temporary produced by a value instruction."""

    index: int


@attrs.define(frozen=True)
class BasicBlockLabel(Operand):
    """Label identifying a basic block within a region."""

    index: int


@attrs.define(frozen=True)
class RegionLabel(Operand):
    """Label identifying a child region."""

    index: int


@attrs.define(frozen=True)
class SyntheticLocal(Operand):
    """Compiler-generated local variable (not user-visible)."""

    index: int
    purpose: str = ""


@attrs.define(frozen=True)
class UnpackedTemporaryValue(Operand):
    """Marker for a splatted value in argument lists."""

    value: TemporaryValue


# ---------------------------------------------------------------------------
# Source provenance
# ---------------------------------------------------------------------------


@attrs.define(frozen=True)
class SourceSpan:
    """Optional source location attached to instructions."""

    lineno: Optional[int] = None
    end_lineno: Optional[int] = None
    col_offset: Optional[int] = None
    end_col_offset: Optional[int] = None


# ---------------------------------------------------------------------------
# Base instruction families
# ---------------------------------------------------------------------------


@attrs.define(frozen=True, kw_only=True)
class Instruction:
    """Base class for all IR instructions."""

    span: Optional[SourceSpan] = None


@attrs.define(frozen=True, kw_only=True)
class EffectInstruction(Instruction):
    """An instruction with no result value (only side effects)."""

    pass


@attrs.define(frozen=True, kw_only=True)
class ValueInstruction(Instruction):
    """An instruction that produces a result in ``dst``."""

    dst: TemporaryValue


# ---------------------------------------------------------------------------
# Variable and constant operations
# ---------------------------------------------------------------------------


@attrs.define(frozen=True)
class Const(ValueInstruction):
    """Load a compile-time constant."""

    value: Any


@attrs.define(frozen=True)
class LoadName(ValueInstruction):
    """Load a named variable from a scope."""

    scope: Scope = Scope.LOCAL
    name: str = ""


@attrs.define(frozen=True)
class StoreName(EffectInstruction):
    """Store a value into a named variable."""

    src: TemporaryValue
    scope: Scope = Scope.LOCAL
    name: str = ""


@attrs.define(frozen=True)
class DeleteName(EffectInstruction):
    """Delete a named variable."""

    scope: Scope = Scope.LOCAL
    name: str = ""


# ---------------------------------------------------------------------------
# Computation and object access
# ---------------------------------------------------------------------------


@attrs.define(frozen=True)
class UnaryOp(ValueInstruction):
    """Apply a unary operator."""

    op: str
    src: TemporaryValue


@attrs.define(frozen=True)
class BinaryOp(ValueInstruction):
    """Apply a binary operator."""

    op: str
    lhs: TemporaryValue
    rhs: TemporaryValue


@attrs.define(frozen=True)
class CompareOp(ValueInstruction):
    """Apply a comparison operator."""

    cmp: str
    lhs: TemporaryValue
    rhs: TemporaryValue


@attrs.define(frozen=True)
class LoadAttr(ValueInstruction):
    """Load an attribute from an object."""

    obj: TemporaryValue
    attr_name: str


@attrs.define(frozen=True)
class StoreAttr(EffectInstruction):
    """Store a value into an object attribute."""

    obj: TemporaryValue
    attr_name: str
    value: TemporaryValue


@attrs.define(frozen=True)
class DeleteAttr(EffectInstruction):
    """Delete an attribute from an object."""

    obj: TemporaryValue
    attr_name: str


@attrs.define(frozen=True)
class LoadItem(ValueInstruction):
    """Load an item via subscript (obj[key])."""

    obj: TemporaryValue
    key: TemporaryValue


@attrs.define(frozen=True)
class StoreItem(EffectInstruction):
    """Store a value via subscript assignment (obj[key] = value)."""

    obj: TemporaryValue
    key: TemporaryValue
    value: TemporaryValue


@attrs.define(frozen=True)
class DeleteItem(EffectInstruction):
    """Delete an item via subscript (del obj[key])."""

    obj: TemporaryValue
    key: TemporaryValue


# ---------------------------------------------------------------------------
# Aggregate builders and destructuring
# ---------------------------------------------------------------------------


@attrs.define(frozen=True)
class BuildTuple(ValueInstruction):
    """Build a tuple from operands."""

    items: Sequence[Operand] = attrs.field(factory=COWList)


@attrs.define(frozen=True)
class BuildList(ValueInstruction):
    """Build a list from operands."""

    items: Sequence[Operand] = attrs.field(factory=COWList)


@attrs.define(frozen=True)
class BuildSet(ValueInstruction):
    """Build a set from operands."""

    items: Sequence[Operand] = attrs.field(factory=COWList)


@attrs.define(frozen=True)
class BuildMap(ValueInstruction):
    """Build a dict from key-value pairs.  ``None`` keys indicate unpacked mappings."""

    items: Sequence[Tuple[Optional[TemporaryValue], TemporaryValue]] = attrs.field(
        factory=COWList
    )


@attrs.define(frozen=True)
class BuildSlice(ValueInstruction):
    """Build a slice object."""

    start: TemporaryValue
    stop: TemporaryValue
    step: Optional[TemporaryValue] = None


@attrs.define(frozen=True)
class BuildString(ValueInstruction):
    """Build a string from parts (f-string concatenation)."""

    parts: Sequence[TemporaryValue] = attrs.field(factory=COWList)


@attrs.define(frozen=True)
class FormatValue(ValueInstruction):
    """Format a single value for an f-string."""

    value: TemporaryValue
    conversion: Optional[str] = None
    spec: Optional[TemporaryValue] = None


@attrs.define(frozen=True)
class Unpack(EffectInstruction):
    """Unpack an iterable into multiple destinations (with optional star)."""

    src: TemporaryValue
    dsts: Sequence[TemporaryValue] = attrs.field(factory=COWList)
    star_index: Optional[int] = None


# ---------------------------------------------------------------------------
# Calls, imports, function / class creation
# ---------------------------------------------------------------------------


@attrs.define(frozen=True)
class Call(ValueInstruction):
    """Call a function or other callable."""

    callee: TemporaryValue
    args: Sequence[Operand] = attrs.field(factory=COWList)
    kwargs: Sequence[Tuple[Optional[str], TemporaryValue]] = attrs.field(
        factory=COWList
    )
    flags: int = 0


@attrs.define(frozen=True)
class ImportName(ValueInstruction):
    """Import a module by name."""

    module: Optional[str]
    fromlist: Sequence[str] = attrs.field(factory=COWList)
    level: int = 0


@attrs.define(frozen=True)
class ImportFrom(ValueInstruction):
    """Import a single name from a module object."""

    module_obj: TemporaryValue
    name: str


@attrs.define(frozen=True)
class ImportStar(EffectInstruction):
    """Perform ``from module import *``."""

    module_obj: TemporaryValue


@attrs.define(frozen=True)
class MakeFunction(ValueInstruction):
    """Create a function object from a child region."""

    code: RegionLabel
    defaults: Sequence[TemporaryValue] = attrs.field(factory=COWList)
    kwdefaults: Sequence[Tuple[str, TemporaryValue]] = attrs.field(factory=COWList)
    annotations: Sequence[Tuple[str, TemporaryValue]] = attrs.field(factory=COWList)
    closure: Sequence[TemporaryValue] = attrs.field(factory=COWList)
    flags: int = 0


@attrs.define(frozen=True)
class BuildClass(ValueInstruction):
    """Build a class from a body function and bases."""

    body_func: TemporaryValue
    name: TemporaryValue
    bases: Sequence[TemporaryValue] = attrs.field(factory=COWList)
    keywords: Sequence[Tuple[str, TemporaryValue]] = attrs.field(factory=COWList)


# ---------------------------------------------------------------------------
# Iteration, generators, and async
# ---------------------------------------------------------------------------


@attrs.define(frozen=True)
class GetIter(ValueInstruction):
    """Get an iterator from an iterable."""

    iterable: TemporaryValue


@attrs.define(frozen=True)
class ForIter(EffectInstruction):
    """Advance an iterator and branch to body or exit."""

    iter_obj: TemporaryValue
    value_dst: TemporaryValue
    body_label: BasicBlockLabel
    exit_label: BasicBlockLabel


@attrs.define(frozen=True)
class GetAIter(ValueInstruction):
    """Get an async iterator from an async iterable."""

    iterable: TemporaryValue


@attrs.define(frozen=True)
class GetANext(ValueInstruction):
    """Get the next awaitable from an async iterator."""

    aiter: TemporaryValue


@attrs.define(frozen=True)
class GetAwaitable(ValueInstruction):
    """Wrap a value as an awaitable."""

    value: TemporaryValue
    where: int = 0


@attrs.define(frozen=True)
class YieldValue(ValueInstruction):
    """Yield a value from a generator."""

    value: TemporaryValue


@attrs.define(frozen=True)
class YieldFrom(ValueInstruction):
    """Delegate to a sub-iterator."""

    value: TemporaryValue


@attrs.define(frozen=True)
class AwaitValue(ValueInstruction):
    """Await an awaitable."""

    value: TemporaryValue


# ---------------------------------------------------------------------------
# Exception and control-flow
# ---------------------------------------------------------------------------


@attrs.define(frozen=True)
class CurrentException(ValueInstruction):
    """Get the currently active exception."""

    pass


@attrs.define(frozen=True)
class Raise(EffectInstruction):
    """Raise an exception."""

    exc: TemporaryValue
    cause: Optional[TemporaryValue] = None


@attrs.define(frozen=True)
class Reraise(EffectInstruction):
    """Re-raise the current exception."""

    pass


@attrs.define(frozen=True)
class CheckExcMatch(ValueInstruction):
    """Check whether an exception matches a type (for except clauses)."""

    exc: TemporaryValue
    typ: TemporaryValue


@attrs.define(frozen=True)
class CheckEGMatch(ValueInstruction):
    """Check whether an exception group matches a type (for except*)."""

    exc: TemporaryValue
    typ: TemporaryValue


@attrs.define(frozen=True)
class PushTry(EffectInstruction):
    """Push a try-block handler entry onto the exception stack."""

    except_label: Optional[BasicBlockLabel] = None
    finally_label: Optional[BasicBlockLabel] = None


@attrs.define(frozen=True)
class PopTry(EffectInstruction):
    """Pop a try-block handler entry."""

    pass


@attrs.define(frozen=True)
class ClearException(EffectInstruction):
    """Clear the currently active exception."""

    pass


@attrs.define(frozen=True)
class EndFinally(EffectInstruction):
    """End a finally block and dispatch based on exception state."""

    pass


@attrs.define(frozen=True)
class Escape(EffectInstruction):
    """Non-local jump (break/continue) to a target block."""

    target: BasicBlockLabel


@attrs.define(frozen=True)
class Jump(EffectInstruction):
    """Unconditional jump to a target block."""

    target: BasicBlockLabel


@attrs.define(frozen=True)
class Branch(EffectInstruction):
    """Conditional branch based on Python truthiness."""

    cond: TemporaryValue
    true_label: BasicBlockLabel
    false_label: BasicBlockLabel


@attrs.define(frozen=True)
class Return(EffectInstruction):
    """Return a value from the current region."""

    value: TemporaryValue


# ---------------------------------------------------------------------------
# Pattern matching
# ---------------------------------------------------------------------------


@attrs.define(frozen=True)
class MatchMapping(ValueInstruction):
    """Check that a value is a mapping for match/case."""

    value: TemporaryValue


@attrs.define(frozen=True)
class MatchSequence(ValueInstruction):
    """Check that a value is a sequence for match/case."""

    value: TemporaryValue


@attrs.define(frozen=True)
class MatchKeys(ValueInstruction):
    """Extract keys from a mapping for match/case."""

    mapping: TemporaryValue
    keys: TemporaryValue


@attrs.define(frozen=True)
class MatchClass(ValueInstruction):
    """Match a class instance for match/case."""

    value: TemporaryValue
    cls: TemporaryValue
    attr_names: Sequence[str] = attrs.field(factory=COWList)
    positional_count: int = 0


# ---------------------------------------------------------------------------
# CFG containers
# ---------------------------------------------------------------------------


@attrs.define(frozen=True)
class BasicBlock:
    """A straight-line sequence of instructions with a label."""

    label: BasicBlockLabel
    instructions: Sequence[Instruction] = attrs.field(factory=COWList)


@attrs.define(frozen=True)
class ExceptionHandler:
    """Metadata describing an exception handler range."""

    start_label: BasicBlockLabel
    end_label: BasicBlockLabel
    target_label: BasicBlockLabel
    stack_depth: Optional[int] = None
    push_lasti: bool = False
    pop_on_entry: int = 0


@attrs.define(frozen=True)
class Region:
    """A named executable unit with its own CFG and nested child regions."""

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


# ===========================================================================
# IR pretty-printer
# ===========================================================================


VALUE_FIELDS = {"dst", "span"}
EFFECT_FIELDS = {"span"}


def format_value(value: Any) -> str:
    """Render a small leaf value inline."""
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


def render_payload(value: Any) -> str:
    """Render nested payload structures (lists of args, block targets, etc.)."""
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


def render_instruction(instr: Any, indent: str = "    ") -> str:
    """Render one instruction using attrs field order."""
    fields = attrs.fields(type(instr))
    names = [field.name for field in fields]
    exclude = VALUE_FIELDS if hasattr(instr, "dst") else EFFECT_FIELDS
    payload_names = [name for name in names if name not in exclude]

    prefix = (
        "%st%s = " % (indent, instr.dst.index)
        if hasattr(instr, "dst")
        else "%s" % indent
    )
    pieces = [
        "%s=%s" % (name, render_payload(getattr(instr, name))) for name in payload_names
    ]
    return "%s%s(%s)" % (prefix, type(instr).__name__, ", ".join(pieces))


def print_instruction(instr: Any, indent: str = "    ") -> None:
    """Print one instruction to stdout."""
    print(render_instruction(instr, indent=indent))


def print_region_ir(region_ir: Region, indent: str = "") -> None:
    """Print one region followed by any nested child regions."""
    label_prefix = "" if region_ir.label is None else "R%s " % (region_ir.label.index,)
    print(
        "%sregion %s%s entry=L%s"
        % (indent, label_prefix, region_ir.name, region_ir.entry_label.index)
    )
    for block in region_ir.basic_blocks:
        print("%s  block L%s:" % (indent, block.label.index))
        for instr in block.instructions:
            print_instruction(instr, indent=indent + "    ")
        if not block.instructions:
            print("%s    <empty>" % indent)
    for child_region in region_ir.child_regions:
        print()
        print_region_ir(child_region, indent=indent + "  ")
