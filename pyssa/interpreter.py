import builtins
from typing import (
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
)

import attrs

from pyssa.ir import (
    AwaitValue,
    BasicBlock,
    BasicBlockLabel,
    BinaryOp,
    Branch,
    BuildClass,
    BuildList,
    BuildMap,
    BuildSet,
    BuildSlice,
    BuildString,
    BuildTuple,
    Call,
    CheckEGMatch,
    CheckExcMatch,
    ClearException,
    CompareOp,
    Const,
    CurrentException,
    DeleteAttr,
    DeleteItem,
    DeleteName,
    EndFinally,
    Escape,
    ForIter,
    FormatValue,
    GetAIter,
    GetANext,
    GetAwaitable,
    GetIter,
    ImportFrom,
    ImportName,
    ImportStar,
    Instruction,
    Jump,
    LoadAttr,
    LoadItem,
    LoadName,
    MakeFunction,
    MatchClass,
    MatchKeys,
    MatchMapping,
    MatchSequence,
    PopTry,
    PushTry,
    Raise,
    Region,
    RegionLabel,
    Reraise,
    Return,
    Scope,
    StoreAttr,
    StoreItem,
    StoreName,
    TemporaryValue,
    UnaryOp,
    Unpack,
    UnpackedTemporaryValue,
    YieldFrom,
    YieldValue,
)
from pyssa.compiler import new_compiler_state

_UNSET = object()


@attrs.define
class Cell:
    """Mutable box used to model Python closure cells."""

    value: Any = _UNSET


class BaseEvent:
    """Base type for all dispatch events."""


@attrs.define(frozen=True)
class JumpEvent(BaseEvent):
    """Transfer control to a new basic block."""

    target: Any  # BasicBlockLabel


@attrs.define(frozen=True)
class YieldEvent(BaseEvent):
    """Yield a value from a generator or coroutine."""

    value: Any


@attrs.define(frozen=True)
class ReturnEvent(BaseEvent):
    """Return a value from the current frame."""

    value: Any


@attrs.define(frozen=True)
class NextInstructionEvent(BaseEvent):
    """Advance to the next instruction."""


# ---------------------------------------------------------------------------
# BaseFrame — abstract frame with stub dispatchers
# ---------------------------------------------------------------------------


class BaseFrame:
    """Abstract interpreter frame with stub dispatchers.

    Fields:
        region_ir       — the ``Region`` being executed
        locals          — local variable namespace
        globals         — global variable namespace
        block_label     — current ``BasicBlockLabel``
        instr_index     — instruction offset within the current block
        cells           — closure cell dict (name → ``Cell``)
        finished        — whether execution has completed
        return_value    — value returned after completion

    Subclasses override ``dispatch_*`` methods to define instruction
    semantics.  Name resolution, block navigation, and the dispatch
    loop are provided by the base class.
    """

    def __init__(
        self,
        region_ir: Region,
        globals: Mapping[str, Any],
        locals: Mapping[str, Any],
        cells: Dict[str, Cell],
        block_label: BasicBlockLabel,
        instr_index: int,
        finished: bool,
        return_value: Any,
    ) -> None:
        self.region_ir = region_ir
        self.locals = locals
        self.globals = globals
        self.block_label = block_label
        self.instr_index = instr_index
        self.cells = dict(cells or {})
        self.finished = finished
        self.return_value = return_value

    # ------------------------------------------------------------------
    # Name resolution
    # ------------------------------------------------------------------

    def load_name(self, scope: Scope, name: str) -> Any:
        if scope == Scope.LOCAL:
            if name in self.locals:
                return self.locals[name]
            if name in self.cells and self.cells[name].value is not _UNSET:
                return self.cells[name].value
            raise NameError(name)
        if scope == Scope.CELL:
            if name in self.cells and self.cells[name].value is not _UNSET:
                return self.cells[name].value
            raise NameError(name)
        if scope == Scope.GLOBAL:
            if name in self.globals:
                return self.globals[name]
            return self.load_builtin(name)
        if scope == Scope.NAME:
            if name in self.locals:
                return self.locals[name]
            if name in self.globals:
                return self.globals[name]
            return self.load_builtin(name)
        raise NotImplementedError("unknown scope %r" % (scope,))

    def store_name(self, scope: Scope, name: str, value: Any) -> None:
        if scope == Scope.GLOBAL:
            self.globals[name] = value
            return
        if scope == Scope.CELL:
            self.cells.setdefault(name, Cell())
            self.cells[name].value = value
            return
        self.locals[name] = value

    def delete_name(self, scope: Scope, name: str) -> None:
        if scope == Scope.GLOBAL:
            del self.globals[name]
            return
        if scope == Scope.CELL:
            if name in self.cells:
                self.cells[name].value = _UNSET
                return
            raise NameError(name)
        del self.locals[name]

    def has_name(self, name: str) -> Optional[Scope]:
        if name in self.locals:
            return Scope.LOCAL
        if name in self.cells and self.cells[name].value is not _UNSET:
            return Scope.CELL
        if name in self.globals:
            return Scope.GLOBAL
        return None

    def load_builtin(self, name: str) -> Any:
        builtins_obj = self.globals.get("__builtins__", builtins.__dict__)
        if isinstance(builtins_obj, dict):
            if name in builtins_obj:
                return builtins_obj[name]
        else:
            if hasattr(builtins_obj, name):
                return getattr(builtins_obj, name)
        raise NameError(name)

    # ===================================================================
    # Instruction handler stubs  —  override in subclasses
    # ===================================================================

    def dispatch_const(self, instr: Const) -> BaseEvent:
        raise NotImplementedError

    def dispatch_load_name(self, instr: LoadName) -> BaseEvent:
        raise NotImplementedError

    def dispatch_store_name(self, instr: StoreName) -> BaseEvent:
        raise NotImplementedError

    def dispatch_delete_name(self, instr: DeleteName) -> BaseEvent:
        raise NotImplementedError

    def dispatch_unary_op(self, instr: UnaryOp) -> BaseEvent:
        raise NotImplementedError

    def dispatch_binary_op(self, instr: BinaryOp) -> BaseEvent:
        raise NotImplementedError

    def dispatch_compare_op(self, instr: CompareOp) -> BaseEvent:
        raise NotImplementedError

    def dispatch_load_attr(self, instr: LoadAttr) -> BaseEvent:
        raise NotImplementedError

    def dispatch_store_attr(self, instr: StoreAttr) -> BaseEvent:
        raise NotImplementedError

    def dispatch_delete_attr(self, instr: DeleteAttr) -> BaseEvent:
        raise NotImplementedError

    def dispatch_load_item(self, instr: LoadItem) -> BaseEvent:
        raise NotImplementedError

    def dispatch_store_item(self, instr: StoreItem) -> BaseEvent:
        raise NotImplementedError

    def dispatch_delete_item(self, instr: DeleteItem) -> BaseEvent:
        raise NotImplementedError

    def dispatch_build_tuple(self, instr: BuildTuple) -> BaseEvent:
        raise NotImplementedError

    def dispatch_build_list(self, instr: BuildList) -> BaseEvent:
        raise NotImplementedError

    def dispatch_build_set(self, instr: BuildSet) -> BaseEvent:
        raise NotImplementedError

    def dispatch_build_map(self, instr: BuildMap) -> BaseEvent:
        raise NotImplementedError

    def dispatch_build_slice(self, instr: BuildSlice) -> BaseEvent:
        raise NotImplementedError

    def dispatch_build_string(self, instr: BuildString) -> BaseEvent:
        raise NotImplementedError

    def dispatch_format_value(self, instr: FormatValue) -> BaseEvent:
        raise NotImplementedError

    def dispatch_unpack(self, instr: Unpack) -> BaseEvent:
        raise NotImplementedError

    def dispatch_call(self, instr: Call) -> BaseEvent:
        raise NotImplementedError

    def dispatch_import_name(self, instr: ImportName) -> BaseEvent:
        raise NotImplementedError

    def dispatch_import_from(self, instr: ImportFrom) -> BaseEvent:
        raise NotImplementedError

    def dispatch_import_star(self, instr: ImportStar) -> BaseEvent:
        raise NotImplementedError

    def dispatch_make_function(self, instr: MakeFunction) -> BaseEvent:
        raise NotImplementedError

    def dispatch_build_class(self, instr: BuildClass) -> BaseEvent:
        raise NotImplementedError

    def dispatch_get_iter(self, instr: GetIter) -> BaseEvent:
        raise NotImplementedError

    def dispatch_for_iter(self, instr: ForIter) -> BaseEvent:
        raise NotImplementedError

    def dispatch_get_aiter(self, instr: GetAIter) -> BaseEvent:
        raise NotImplementedError

    def dispatch_get_anext(self, instr: GetANext) -> BaseEvent:
        raise NotImplementedError

    def dispatch_get_awaitable(self, instr: GetAwaitable) -> BaseEvent:
        raise NotImplementedError

    def dispatch_yield_value(self, instr: YieldValue) -> BaseEvent:
        raise NotImplementedError

    def dispatch_yield_from(self, instr: YieldFrom) -> BaseEvent:
        raise NotImplementedError

    def dispatch_await_value(self, instr: AwaitValue) -> BaseEvent:
        raise NotImplementedError

    def dispatch_current_exception(self, instr: CurrentException) -> BaseEvent:
        raise NotImplementedError

    def dispatch_raise(self, instr: Raise) -> BaseEvent:
        raise NotImplementedError

    def dispatch_reraise(self, instr: Reraise) -> BaseEvent:
        raise NotImplementedError

    def dispatch_check_exc_match(self, instr: CheckExcMatch) -> BaseEvent:
        raise NotImplementedError

    def dispatch_check_eg_match(self, instr: CheckEGMatch) -> BaseEvent:
        raise NotImplementedError

    def dispatch_push_try(self, instr: PushTry) -> BaseEvent:
        raise NotImplementedError

    def dispatch_pop_try(self, instr: PopTry) -> BaseEvent:
        raise NotImplementedError

    def dispatch_clear_exception(self, instr: ClearException) -> BaseEvent:
        raise NotImplementedError

    def dispatch_end_finally(self, instr: EndFinally) -> BaseEvent:
        raise NotImplementedError

    def dispatch_escape(self, instr: Escape) -> BaseEvent:
        raise NotImplementedError

    def dispatch_jump(self, instr: Jump) -> BaseEvent:
        raise NotImplementedError

    def dispatch_branch(self, instr: Branch) -> BaseEvent:
        raise NotImplementedError

    def dispatch_return(self, instr: Return) -> BaseEvent:
        raise NotImplementedError

    def dispatch_match_mapping(self, instr: MatchMapping) -> BaseEvent:
        raise NotImplementedError

    def dispatch_match_sequence(self, instr: MatchSequence) -> BaseEvent:
        raise NotImplementedError

    def dispatch_match_keys(self, instr: MatchKeys) -> BaseEvent:
        raise NotImplementedError

    def dispatch_match_class(self, instr: MatchClass) -> BaseEvent:
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Block / instruction navigation
    # ------------------------------------------------------------------

    def get_block(self, label: Optional[BasicBlockLabel] = None) -> BasicBlock:
        if label is None:
            label = self.block_label
        for block in self.region_ir.basic_blocks:
            if block.label == label:
                return block
        raise KeyError(
            "unknown block label %r in %s" % (label, self.region_ir.name)
        )

    def fallthrough_label(
        self, label: Optional[BasicBlockLabel] = None
    ) -> Optional[BasicBlockLabel]:
        if label is None:
            label = self.block_label
        basic_blocks = list(self.region_ir.basic_blocks)
        for index, block in enumerate(basic_blocks):
            if block.label == label and index + 1 < len(basic_blocks):
                return basic_blocks[index + 1].label
        return None

    def get_current_instruction(self) -> Optional[Instruction]:
        block = self.get_block()
        if self.instr_index < len(block.instructions):
            return block.instructions[self.instr_index]
        return None

    # ------------------------------------------------------------------
    # Instruction dispatch
    # ------------------------------------------------------------------

    def dispatch_current_instruction(self) -> BaseEvent:
        """Dispatch the current instruction and return its event."""
        if self.finished:
            return ReturnEvent(None)

        block = self.get_block()
        if self.instr_index >= len(block.instructions):
            next_label = self.fallthrough_label()
            if next_label is None:
                self.finished = True
                return ReturnEvent(None)
            self.block_label = next_label
            self.instr_index = 0
            block = self.get_block()

        instr = block.instructions[self.instr_index]

        if isinstance(instr, Const):
            event = self.dispatch_const(instr)
        elif isinstance(instr, LoadName):
            event = self.dispatch_load_name(instr)
        elif isinstance(instr, StoreName):
            event = self.dispatch_store_name(instr)
        elif isinstance(instr, DeleteName):
            event = self.dispatch_delete_name(instr)
        elif isinstance(instr, UnaryOp):
            event = self.dispatch_unary_op(instr)
        elif isinstance(instr, BinaryOp):
            event = self.dispatch_binary_op(instr)
        elif isinstance(instr, CompareOp):
            event = self.dispatch_compare_op(instr)
        elif isinstance(instr, LoadAttr):
            event = self.dispatch_load_attr(instr)
        elif isinstance(instr, StoreAttr):
            event = self.dispatch_store_attr(instr)
        elif isinstance(instr, DeleteAttr):
            event = self.dispatch_delete_attr(instr)
        elif isinstance(instr, LoadItem):
            event = self.dispatch_load_item(instr)
        elif isinstance(instr, StoreItem):
            event = self.dispatch_store_item(instr)
        elif isinstance(instr, DeleteItem):
            event = self.dispatch_delete_item(instr)
        elif isinstance(instr, BuildTuple):
            event = self.dispatch_build_tuple(instr)
        elif isinstance(instr, BuildList):
            event = self.dispatch_build_list(instr)
        elif isinstance(instr, BuildSet):
            event = self.dispatch_build_set(instr)
        elif isinstance(instr, BuildMap):
            event = self.dispatch_build_map(instr)
        elif isinstance(instr, BuildSlice):
            event = self.dispatch_build_slice(instr)
        elif isinstance(instr, BuildString):
            event = self.dispatch_build_string(instr)
        elif isinstance(instr, FormatValue):
            event = self.dispatch_format_value(instr)
        elif isinstance(instr, Unpack):
            event = self.dispatch_unpack(instr)
        elif isinstance(instr, Call):
            event = self.dispatch_call(instr)
        elif isinstance(instr, ImportName):
            event = self.dispatch_import_name(instr)
        elif isinstance(instr, ImportFrom):
            event = self.dispatch_import_from(instr)
        elif isinstance(instr, ImportStar):
            event = self.dispatch_import_star(instr)
        elif isinstance(instr, MakeFunction):
            event = self.dispatch_make_function(instr)
        elif isinstance(instr, BuildClass):
            event = self.dispatch_build_class(instr)
        elif isinstance(instr, GetIter):
            event = self.dispatch_get_iter(instr)
        elif isinstance(instr, ForIter):
            event = self.dispatch_for_iter(instr)
        elif isinstance(instr, GetAIter):
            event = self.dispatch_get_aiter(instr)
        elif isinstance(instr, GetANext):
            event = self.dispatch_get_anext(instr)
        elif isinstance(instr, GetAwaitable):
            event = self.dispatch_get_awaitable(instr)
        elif isinstance(instr, YieldValue):
            event = self.dispatch_yield_value(instr)
        elif isinstance(instr, YieldFrom):
            event = self.dispatch_yield_from(instr)
        elif isinstance(instr, AwaitValue):
            event = self.dispatch_await_value(instr)
        elif isinstance(instr, CurrentException):
            event = self.dispatch_current_exception(instr)
        elif isinstance(instr, Raise):
            event = self.dispatch_raise(instr)
        elif isinstance(instr, Reraise):
            event = self.dispatch_reraise(instr)
        elif isinstance(instr, CheckExcMatch):
            event = self.dispatch_check_exc_match(instr)
        elif isinstance(instr, CheckEGMatch):
            event = self.dispatch_check_eg_match(instr)
        elif isinstance(instr, PushTry):
            event = self.dispatch_push_try(instr)
        elif isinstance(instr, PopTry):
            event = self.dispatch_pop_try(instr)
        elif isinstance(instr, ClearException):
            event = self.dispatch_clear_exception(instr)
        elif isinstance(instr, EndFinally):
            event = self.dispatch_end_finally(instr)
        elif isinstance(instr, Escape):
            event = self.dispatch_escape(instr)
        elif isinstance(instr, Jump):
            event = self.dispatch_jump(instr)
        elif isinstance(instr, Branch):
            event = self.dispatch_branch(instr)
        elif isinstance(instr, Return):
            event = self.dispatch_return(instr)
        elif isinstance(instr, MatchMapping):
            event = self.dispatch_match_mapping(instr)
        elif isinstance(instr, MatchSequence):
            event = self.dispatch_match_sequence(instr)
        elif isinstance(instr, MatchKeys):
            event = self.dispatch_match_keys(instr)
        elif isinstance(instr, MatchClass):
            event = self.dispatch_match_class(instr)
        else:
            raise NotImplementedError(
                "unsupported IR instruction: %r" % (instr,)
            )

        if isinstance(event, NextInstructionEvent):
            self.instr_index += 1
            return event
        if isinstance(event, JumpEvent):
            self.block_label = event.target
            self.instr_index = 0
            return event
        elif isinstance(event, YieldEvent):
            self.instr_index += 1
            return event
        elif isinstance(event, ReturnEvent):
            self.finished = True
            return event
        else:
            raise RuntimeError("unknown execution event %r" % (event,))


# ---------------------------------------------------------------------------
# make_frame — standalone frame constructor
# ---------------------------------------------------------------------------


def make_frame(
    frame_class: Type[BaseFrame],
    region_ir: Region,
    globals: Mapping[str, Any],
    locals: Optional[Mapping[str, Any]] = None,
    cells: Optional[Dict[str, Cell]] = None,
) -> BaseFrame:
    """Materialize a frame for one region invocation.

    When *locals* is not given, *globals* is used as the locals
    namespace (module entry).  *cells* provides pre-seeded closure cells.
    """
    if locals is None:
        locals = globals
    cells = dict(cells or {})
    for name in region_ir.cells:
        if name not in cells:
            cells[name] = Cell(locals.get(name, _UNSET))
    return frame_class(
        region_ir=region_ir,
        globals=globals,
        locals=locals,
        cells=cells,
        block_label=region_ir.entry_label,
        instr_index=0,
        finished=False,
        return_value=None,
    )
