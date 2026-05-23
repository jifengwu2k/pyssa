import builtins
from collections import abc
import contextlib
import importlib
import importlib.machinery
import importlib.util
import inspect
import operator
import os
import sys
import types
from typing import (
    Any,
    AsyncGenerator,
    Coroutine,
    Dict,
    Generator,
    Iterable,
    ItemsView,
    KeysView,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
    ValuesView,
)

import attrs
from typing_extensions import Protocol

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
from pyssa.compiler import compile_file, new_compiler_state

_UNSET = object()


@attrs.define
class Cell:
    """Mutable box used to model Python closure cells."""

    value: Any = _UNSET


@attrs.define(frozen=True)
class JumpEvent:
    """Transfer control to a new basic block."""

    target: Any  # BasicBlockLabel


@attrs.define(frozen=True)
class YieldEvent:
    """Yield a value from a generator or coroutine."""

    value: Any


@attrs.define(frozen=True)
class ReturnEvent:
    """Return a value from the current frame."""

    value: Any


class Namespace(Protocol):
    def __contains__(self, key: str) -> bool: ...

    def __getitem__(self, key: str) -> Any: ...

    def __setitem__(self, key: str, value: Any) -> None: ...

    def __delitem__(self, key: str) -> None: ...

    def get(self, key: str, default: Any = None) -> Any: ...

    def setdefault(self, key: str, default: Any = None) -> Any: ...

    def items(self) -> Iterable[Tuple[str, Any]]: ...


ModuleSpec = importlib.machinery.ModuleSpec
ControlEvent = Union[JumpEvent, ReturnEvent]
DispatchEvent = Union[JumpEvent, YieldEvent, ReturnEvent]
StepEvent = Union[YieldEvent, ReturnEvent]
TryStackEntry = Dict[str, Optional[BasicBlockLabel]]


# ---------------------------------------------------------------------------
# Module loader utilities
# ---------------------------------------------------------------------------


def bind_submodule(fullname: str, module: Any) -> None:
    parent_name, _, child_name = fullname.rpartition(".")
    if not parent_name:
        return
    parent_module = sys.modules.get(parent_name)
    if parent_module is not None:
        setattr(parent_module, child_name, module)


def is_ir_source_spec(spec: Optional[Any]) -> bool:
    if spec is None or spec.origin in (None, "built-in", "frozen"):
        return False
    return isinstance(
        spec.loader, importlib.machinery.SourceFileLoader
    ) and spec.origin.endswith(".py")


# ---------------------------------------------------------------------------
# Operation semantics
# ---------------------------------------------------------------------------


def apply_unary(op: str, value: Any) -> Any:
    if op == "+":
        return +value
    if op == "-":
        return -value
    if op == "not":
        return not value
    if op == "~":
        return ~value
    raise NotImplementedError("unsupported unary op %r" % (op,))


def apply_binary(op: str, lhs: Any, rhs: Any) -> Any:
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


def apply_compare(cmp: str, lhs: Any, rhs: Any) -> Any:
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


# ---------------------------------------------------------------------------
# Async / exception helpers
# ---------------------------------------------------------------------------


def await_sync(value: Any) -> Any:
    """Synchronous await trampoline."""
    if inspect.isawaitable(value):
        iterator = value.__await__()
        send_value = None
        while True:
            try:
                yielded = iterator.send(send_value)
            except StopIteration as stop:
                return stop.value
            if inspect.isawaitable(yielded):
                send_value = await_sync(yielded)
            else:
                send_value = yielded
    return value


def normalize_exception_for_raise(
    value: Any, allow_none: bool = False
) -> Optional[BaseException]:
    if allow_none and value is None:
        return None
    if isinstance(value, BaseException):
        return value
    if isinstance(value, type) and issubclass(value, BaseException):
        return value()
    raise TypeError("exceptions must derive from BaseException")


def is_valid_exception_match_type(typ: Any) -> bool:
    if isinstance(typ, type):
        return issubclass(typ, BaseException)
    if isinstance(typ, tuple):
        return all(is_valid_exception_match_type(item) for item in typ)
    return False


def check_exception_match(exc: BaseException, typ: Any) -> bool:
    if not is_valid_exception_match_type(typ):
        raise TypeError(
            "catching classes that do not inherit from BaseException " "is not allowed"
        )
    return isinstance(exc, typ)


# ---------------------------------------------------------------------------
# Module wrappers — keep IR-loaded modules out of sys.modules
# ---------------------------------------------------------------------------


class Module:
    """Module namespace backed by a plain dict, usable as both dict and object."""

    def __init__(
        self,
        name: str,
        file: Optional[str],
        package: Optional[str],
        path: Optional[List[str]] = None,
    ) -> None:
        object.__setattr__(
            self,
            "_dict",
            {
                "__name__": name,
                "__file__": file,
                "__package__": package,
                "__path__": path,
            },
        )

    def __getattr__(self, name: str) -> Any:
        try:
            return self._dict[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        self._dict[name] = value

    def __delattr__(self, name: str) -> None:
        del self._dict[name]

    def __getitem__(self, key: str) -> Any:
        return self._dict[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._dict[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self._dict

    def __delitem__(self, key: str) -> None:
        del self._dict[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self._dict.get(key, default)

    def setdefault(self, key: str, default: Any = None) -> Any:
        return self._dict.setdefault(key, default)

    def keys(self) -> KeysView[str]:
        return self._dict.keys()

    def values(self) -> ValuesView[Any]:
        return self._dict.values()

    def items(self) -> ItemsView[str, Any]:
        return self._dict.items()


# ---------------------------------------------------------------------------
# Function — pure data, no execution logic
# ---------------------------------------------------------------------------


ModuleValue = Union[Module, types.ModuleType]


class Function:
    """A compiled pyssa IR function — pure data, no execution logic."""

    def __init__(
        self,
        region_ir: Region,
        globals_dict: Namespace,
        closure_cells: Optional[Dict[str, Cell]] = None,
        preloaded_locals: Optional[Dict[str, Any]] = None,
        __defaults__: Optional[Tuple[Any, ...]] = None,
        __kwdefaults__: Optional[Dict[str, Any]] = None,
        __annotations__: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.region_ir = region_ir
        self.globals_dict = globals_dict
        self.closure_cells = dict(closure_cells or {})
        self.preloaded_locals = dict(preloaded_locals or {})
        self.__defaults__ = __defaults__
        self.__kwdefaults__ = __kwdefaults__
        self.__annotations__ = __annotations__ if __annotations__ is not None else {}

        self.__name__ = region_ir.name.split("#", 1)[0]
        self.__module__ = globals_dict.get("__name__", "__main__")


# ---------------------------------------------------------------------------
# Interpreter — module loading, frame construction, execution entry
# ---------------------------------------------------------------------------


class Interpreter:
    """Manages module loading and execution of ``Function`` objects."""

    def __init__(
        self,
        search_path: Optional[List[str]] = None,
        module_ir_cache: Optional[Dict[str, Region]] = None,
        module_cache: Optional[Dict[str, ModuleValue]] = None,
    ) -> None:
        self.search_path = search_path or []
        self.module_ir_cache = module_ir_cache or {}
        self.module_cache = module_cache or {}
        self._frame_class_stack: List[Type["BaseFrame"]] = []

    def current_frame_class(self) -> Type["BaseFrame"]:
        if self._frame_class_stack:
            return self._frame_class_stack[-1]
        return Frame

    @contextlib.contextmanager
    def use_frame_class(
        self, frame_class: Type["BaseFrame"]
    ) -> Generator[None, None, None]:
        self._frame_class_stack.append(frame_class)
        try:
            yield
        finally:
            self._frame_class_stack.pop()

    # ------------------------------------------------------------------
    # Module import machinery
    # ------------------------------------------------------------------

    def resolve_absolute_import_name(
        self, globals: Namespace, module_name: Optional[str], level: int
    ) -> str:
        module_name = "" if module_name is None else module_name
        if level == 0:
            return module_name
        package_name = globals.get("__package__")
        if package_name is None:
            current_name = globals.get("__name__", "")
            if "__path__" in globals:
                package_name = current_name
            else:
                package_name = current_name.rpartition(".")[0]
        if not package_name:
            raise ImportError(
                "attempted relative import with no known parent package"
            )
        return importlib.util.resolve_name("." * level + module_name, package_name)

    def import_module(
        self,
        globals: Namespace,
        module_name: Optional[str],
        fromlist: Sequence[str],
        level: int,
    ) -> ModuleValue:
        """Full filesystem module resolution (absolute + relative)."""
        absolute_name = self.resolve_absolute_import_name(
            globals, module_name, level
        )
        module = self.import_absolute_module(absolute_name)
        self.ensure_fromlist(module, fromlist)
        if fromlist:
            return module
        top_level_name = absolute_name.split(".", 1)[0]
        return self.module_cache.get(top_level_name, module)

    def find_module_spec(self, fullname: str) -> Optional[ModuleSpec]:
        parent_name, _, _ = fullname.rpartition(".")
        search_path = list(self.search_path)
        if parent_name:
            parent_module = self.import_absolute_module(parent_name)
            search_path = getattr(parent_module, "__path__", None)
            if search_path is None:
                raise ModuleNotFoundError(
                    "No module named %s; %s is not a package"
                    % (fullname, parent_name),
                    name=fullname,
                )
        return importlib.machinery.PathFinder.find_spec(fullname, search_path)

    def load_module_ir(self, path: str) -> Region:
        path = os.path.abspath(path)
        cached = self.module_ir_cache.get(path)
        if cached is not None:
            return cached
        module_ir = compile_file(new_compiler_state(), path)
        self.module_ir_cache[path] = module_ir
        return module_ir

    def bind_submodule(self, fullname: str, module: ModuleValue) -> None:
        parent_name, _, child_name = fullname.rpartition(".")
        if not parent_name:
            return
        parent = self.module_cache.get(parent_name)
        if parent is not None:
            setattr(parent, child_name, module)

    def load_ir_module_from_spec(self, fullname: str, spec: ModuleSpec) -> Module:
        existing = self.module_cache.get(fullname)
        if existing is not None:
            return existing
        origin = spec.origin
        file = os.path.abspath(origin) if origin else None
        pkg = spec.parent or fullname.rpartition(".")[0] or None
        pkg_path = (
            [os.path.abspath(e) for e in spec.submodule_search_locations]
            if spec.submodule_search_locations
            else None
        )
        module = Module(fullname, file, pkg, pkg_path)
        self.module_cache[fullname] = module
        self.bind_submodule(fullname, module)
        try:
            module.setdefault("__builtins__", builtins.__dict__)
            module["__build_class__"] = self.build_class
            module_ir = self.load_module_ir(origin)
            module_function = Function(region_ir=module_ir, globals_dict=module)
            module.setdefault("__name__", module_function.__name__)
            frame = self.make_frame(module_function, (), {}, locals=module)
            while not frame.finished:
                event = frame.dispatch_current_instruction()
                if event is not None:
                    if isinstance(event, ReturnEvent):
                        break
                    raise RuntimeError("frame yielded unexpectedly: %r" % (event,))
        except BaseException:
            del self.module_cache[fullname]
            raise
        return module

    def load_python_module_from_spec(
        self, fullname: str, spec: ModuleSpec
    ) -> types.ModuleType:
        existing = self.module_cache.get(fullname)
        if existing is not None:
            return existing
        module = importlib.util.module_from_spec(spec)
        if getattr(module, "__file__", None) is not None:
            module.__file__ = os.path.abspath(module.__file__)
        if getattr(module, "__path__", None) is not None:
            module.__path__ = [os.path.abspath(e) for e in module.__path__]
        sys.modules[fullname] = module
        bind_submodule(fullname, module)
        self.module_cache[fullname] = module
        try:
            if spec.loader is not None:
                spec.loader.exec_module(module)
        except BaseException:
            del self.module_cache[fullname]
            if sys.modules.get(fullname) is module:
                del sys.modules[fullname]
            raise
        return module

    def import_absolute_module(self, fullname: str) -> ModuleValue:
        existing = self.module_cache.get(fullname)
        if existing is not None:
            return existing
        spec = self.find_module_spec(fullname)
        if spec is None:
            module = importlib.import_module(fullname)
            self.module_cache[fullname] = module
            return module
        if is_ir_source_spec(spec):
            return self.load_ir_module_from_spec(fullname, spec)
        return self.load_python_module_from_spec(fullname, spec)

    def ensure_fromlist(self, module: ModuleValue, fromlist: Sequence[str]) -> None:
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

    # ------------------------------------------------------------------
    # Frame construction and execution
    # ------------------------------------------------------------------

    def make_frame(
        self,
        function: Function,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
        *,
        locals: Optional[Namespace] = None,
        frame_class: Optional[Type["BaseFrame"]] = None,
    ) -> "BaseFrame":
        """Materialize a frame for one function invocation.

        When *locals* is given it is used directly as the frame's
        locals namespace (class bodies).  Otherwise arguments are
        bound normally via ``bind_arguments``.
        """
        region_ir = function.region_ir
        if locals is None:
            if region_ir.name == "<module>":
                locals = function.globals_dict
            else:
                bound_locals = bind_arguments(function, args, kwargs)
                bound_locals.update(function.preloaded_locals)
                locals = bound_locals
        cells = dict(function.closure_cells)
        for name in region_ir.cells:
            if name not in cells:
                cells[name] = Cell(locals.get(name, _UNSET))
        if frame_class is None:
            frame_class = self.current_frame_class()
        return frame_class(
            interpreter=self,
            function=function,
            globals=function.globals_dict,
            locals=locals,
            cells=cells,
            block_label=region_ir.entry_label,
            instr_index=0,
        )

    def build_class(
        self,
        body_function: Function,
        name: str,
        *bases: Any,
        **kwargs: Any,
    ) -> Any:
        """Execute a lowered class body and return the resulting class."""
        explicit_metaclass = kwargs.pop("metaclass", None)
        if explicit_metaclass is None:
            metaclass = type(bases[0]) if bases else type
            for base in bases[1:]:
                base_metaclass = type(base)
                if issubclass(base_metaclass, metaclass):
                    metaclass = base_metaclass
                elif not issubclass(metaclass, base_metaclass):
                    raise TypeError("metaclass conflict")
        else:
            metaclass = explicit_metaclass
        prepare = getattr(metaclass, "__prepare__", None)
        if prepare is None:
            namespace = {}
        else:
            namespace = prepare(name, bases, **kwargs)
        namespace["__module__"] = body_function.globals_dict.get(
            "__name__", "__main__"
        )
        namespace["__qualname__"] = name
        frame = self.make_frame(body_function, (), {}, locals=namespace)
        frame.run_to_completion()
        for special_name in ("__init_subclass__", "__class_getitem__"):
            value = namespace.get(special_name)
            if isinstance(value, Function):
                namespace[special_name] = classmethod(value)
        cls = metaclass(name, bases, namespace, **kwargs)
        class_cell = frame.cells.get("__class__")
        if class_cell is not None:
            class_cell.value = cls
        return cls

    def copy(self) -> "Interpreter":
        """Return a forked interpreter sharing module caches."""
        return Interpreter(
            search_path=self.search_path,
            module_ir_cache=self.module_ir_cache,
            module_cache=self.module_cache,
        )


# ---------------------------------------------------------------------------
# BaseFrame — abstract interpreter frame with pluggable instruction handlers
# ---------------------------------------------------------------------------


class BaseFrame:
    """Abstract base for interpreter frames.

    Fields (set by ``__init__``):
        interpreter     — the owning ``Interpreter`` instance
        function        — the ``Function`` being executed
        locals          — local variable namespace
        globals         — global variable namespace
        block_label     — current ``BasicBlockLabel``
        instr_index     — instruction offset within the current block
        cells           — closure cell dict (name → ``Cell``)
        try_stack       — exception handler stack
        finished        — whether execution has completed
        return_value    — value returned after completion
        current_exception — active exception (or ``None``)

    Methods:
        Name resolution: ``load_name``, ``store_name``, ``delete_name``,
        ``has_name``, ``load_builtin``.

        Exception handling: ``handle_exception``.

        Block navigation: ``get_block``, ``fallthrough_label``,
        ``get_current_instruction``.

        Dispatch loop: ``dispatch_current_instruction``.

        Instruction stubs: all ``dispatch_*`` methods raise
        ``NotImplementedError``; subclasses override them.
    """

    def __init__(
        self,
        interpreter: Interpreter,
        function: Function,
        locals: Namespace,
        globals: Namespace,
        block_label: Optional[BasicBlockLabel] = None,
        instr_index: int = 0,
        cells: Optional[Dict[str, Cell]] = None,
        try_stack: Optional[List[TryStackEntry]] = None,
    ) -> None:
        self.interpreter = interpreter
        self.function = function
        self.locals = locals
        self.globals = globals
        self.block_label = block_label or self.function.region_ir.entry_label
        self.instr_index = instr_index
        self.cells = cells or {}
        self.try_stack = list(try_stack or [])
        self.finished = False
        self.return_value: Any = None
        self.current_exception: Optional[BaseException] = None

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

    # ------------------------------------------------------------------
    # Exception handling
    # ------------------------------------------------------------------

    def handle_exception(self, exception: BaseException) -> Optional[JumpEvent]:
        self.current_exception = exception
        for index in range(len(self.try_stack) - 1, -1, -1):
            entry = self.try_stack[index]
            target = entry.get("except_label") or entry.get("finally_label")
            if target is None:
                continue
            del self.try_stack[index:]
            return JumpEvent(target)
        return None

    # ===================================================================
    # Instruction handler stubs  (self, instr) -> event | None
    #
    # Each raises NotImplementedError.  Subclasses must provide concrete
    # implementations for the instructions they wish to support.
    # ===================================================================

    # --- value / variable ---

    def dispatch_const(self, instr: Const) -> None:
        raise NotImplementedError

    def dispatch_load_name(self, instr: LoadName) -> None:
        raise NotImplementedError

    def dispatch_store_name(self, instr: StoreName) -> None:
        raise NotImplementedError

    def dispatch_delete_name(self, instr: DeleteName) -> None:
        raise NotImplementedError

    # --- computation ---

    def dispatch_unary_op(self, instr: UnaryOp) -> None:
        raise NotImplementedError

    def dispatch_binary_op(self, instr: BinaryOp) -> None:
        raise NotImplementedError

    def dispatch_compare_op(self, instr: CompareOp) -> None:
        raise NotImplementedError

    # --- attribute / item access ---

    def dispatch_load_attr(self, instr: LoadAttr) -> None:
        raise NotImplementedError

    def dispatch_store_attr(self, instr: StoreAttr) -> None:
        raise NotImplementedError

    def dispatch_delete_attr(self, instr: DeleteAttr) -> None:
        raise NotImplementedError

    def dispatch_load_item(self, instr: LoadItem) -> None:
        raise NotImplementedError

    def dispatch_store_item(self, instr: StoreItem) -> None:
        raise NotImplementedError

    def dispatch_delete_item(self, instr: DeleteItem) -> None:
        raise NotImplementedError

    # --- aggregates ---

    def dispatch_build_tuple(self, instr: BuildTuple) -> None:
        raise NotImplementedError

    def dispatch_build_list(self, instr: BuildList) -> None:
        raise NotImplementedError

    def dispatch_build_set(self, instr: BuildSet) -> None:
        raise NotImplementedError

    def dispatch_build_map(self, instr: BuildMap) -> None:
        raise NotImplementedError

    def dispatch_build_slice(self, instr: BuildSlice) -> None:
        raise NotImplementedError

    def dispatch_build_string(self, instr: BuildString) -> None:
        raise NotImplementedError

    def dispatch_format_value(self, instr: FormatValue) -> None:
        raise NotImplementedError

    def dispatch_unpack(self, instr: Unpack) -> None:
        raise NotImplementedError

    # --- calls / imports / functions / classes ---

    def dispatch_call(self, instr: Call) -> None:
        raise NotImplementedError

    def dispatch_import_name(self, instr: ImportName) -> None:
        raise NotImplementedError

    def dispatch_import_from(self, instr: ImportFrom) -> None:
        raise NotImplementedError

    def dispatch_import_star(self, instr: ImportStar) -> None:
        raise NotImplementedError

    def dispatch_make_function(self, instr: MakeFunction) -> None:
        raise NotImplementedError

    def dispatch_build_class(self, instr: BuildClass) -> None:
        raise NotImplementedError

    # --- iteration / async ---

    def dispatch_get_iter(self, instr: GetIter) -> None:
        raise NotImplementedError

    def dispatch_for_iter(self, instr: ForIter) -> JumpEvent:
        raise NotImplementedError

    def dispatch_get_aiter(self, instr: GetAIter) -> None:
        raise NotImplementedError

    def dispatch_get_anext(self, instr: GetANext) -> None:
        raise NotImplementedError

    def dispatch_get_awaitable(self, instr: GetAwaitable) -> None:
        raise NotImplementedError

    def dispatch_yield_value(self, instr: YieldValue) -> YieldEvent:
        raise NotImplementedError

    def dispatch_yield_from(self, instr: YieldFrom) -> YieldEvent:
        raise NotImplementedError

    def dispatch_await_value(self, instr: AwaitValue) -> None:
        raise NotImplementedError

    # --- exceptions ---

    def dispatch_current_exception(self, instr: CurrentException) -> None:
        raise NotImplementedError

    def dispatch_raise(self, instr: Raise) -> None:
        raise NotImplementedError

    def dispatch_reraise(self, instr: Reraise) -> None:
        raise NotImplementedError

    def dispatch_check_exc_match(self, instr: CheckExcMatch) -> None:
        raise NotImplementedError

    def dispatch_check_eg_match(self, instr: CheckEGMatch) -> None:
        raise NotImplementedError

    def dispatch_push_try(self, instr: PushTry) -> None:
        raise NotImplementedError

    def dispatch_pop_try(self, instr: PopTry) -> None:
        raise NotImplementedError

    def dispatch_clear_exception(self, instr: ClearException) -> None:
        raise NotImplementedError

    # --- control flow ---

    def dispatch_end_finally(self, instr: EndFinally) -> Optional[ControlEvent]:
        raise NotImplementedError

    def dispatch_escape(self, instr: Escape) -> JumpEvent:
        raise NotImplementedError

    def dispatch_jump(self, instr: Jump) -> JumpEvent:
        raise NotImplementedError

    def dispatch_branch(self, instr: Branch) -> JumpEvent:
        raise NotImplementedError

    def dispatch_return(self, instr: Return) -> ControlEvent:
        raise NotImplementedError

    # --- pattern matching ---

    def dispatch_match_mapping(self, instr: MatchMapping) -> None:
        raise NotImplementedError

    def dispatch_match_sequence(self, instr: MatchSequence) -> None:
        raise NotImplementedError

    def dispatch_match_keys(self, instr: MatchKeys) -> None:
        raise NotImplementedError

    def dispatch_match_class(self, instr: MatchClass) -> None:
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Block / instruction navigation
    # ------------------------------------------------------------------

    def get_block(self, label: Optional[BasicBlockLabel] = None) -> BasicBlock:
        if label is None:
            label = self.block_label
        for block in self.function.region_ir.basic_blocks:
            if block.label == label:
                return block
        raise KeyError(
            "unknown block label %r in %s" % (label, self.function.region_ir.name)
        )

    def fallthrough_label(
        self, label: Optional[BasicBlockLabel] = None
    ) -> Optional[BasicBlockLabel]:
        if label is None:
            label = self.block_label
        basic_blocks = list(self.function.region_ir.basic_blocks)
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

    def dispatch_current_instruction(self) -> Optional[StepEvent]:
        """Dispatch the current instruction and return its event.

        Handles block fallthrough, PC advancement, jump application,
        and exception dispatch internally:
        - Effect instructions: PC advanced, returns ``None``.
        - Jumps: applied (PC updated), returns ``None``.
        - Exceptions: dispatched through try_stack, jumps applied.
        - Yield / Return: returned to caller for ``resume`` to stop.
        """
        with self.interpreter.use_frame_class(type(self)):
            if self.finished:
                return ReturnEvent(None)
            if self.block_label is None:
                self.block_label = self.function.region_ir.entry_label

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

            # --- dispatch the instruction, catching exceptions ---
            try:
                event: Optional[DispatchEvent] = None
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
            except BaseException as exc:
                handled = self.handle_exception(exc)
                if handled is None:
                    self.current_exception = exc
                    raise
                event = handled

            # --- apply the event ---
            if event is None:
                self.instr_index += 1
                return None
            if isinstance(event, JumpEvent):
                self.block_label = event.target
                self.instr_index = 0
                return None
            elif isinstance(event, YieldEvent):
                self.instr_index += 1
                return event
            elif isinstance(event, ReturnEvent):
                self.finished = True
                return event
            else:
                raise RuntimeError("unknown execution event %r" % (event,))


# ---------------------------------------------------------------------------
# Frame — full concrete interpreter frame
# ---------------------------------------------------------------------------


class Frame(BaseFrame):
    """Full concrete interpreter frame.

    Provides all state fields, lifecycle methods, helper utilities, and
    concrete ``dispatch_*`` implementations.
    """

    def __init__(
        self,
        interpreter: Interpreter,
        function: Function,
        globals: Namespace,
        locals: Namespace,
        cells: Dict[str, Cell],
        temps: Optional[Dict[TemporaryValue, Any]] = None,
        block_label: Optional[BasicBlockLabel] = None,
        instr_index: int = 0,
        finished: bool = False,
        return_value: Any = None,
        current_exception: Optional[BaseException] = None,
        exc_stack: Optional[List[BaseException]] = None,
        pending_send_value: Any = _UNSET,
        try_stack: Optional[List[TryStackEntry]] = None,
        pending_return_value: Any = _UNSET,
        pending_jump_label: Any = _UNSET,
    ) -> None:
        super().__init__(
            interpreter=interpreter,
            function=function,
            locals=locals,
            globals=globals,
            block_label=block_label,
            instr_index=instr_index,
            cells=cells,
            try_stack=try_stack,
        )
        self.temps = dict(temps or {})
        self.finished = finished
        self.return_value = return_value
        self.current_exception = current_exception
        self.exc_stack = list(exc_stack or [])
        self.pending_send_value = pending_send_value
        self.pending_return_value = pending_return_value
        self.pending_jump_label = pending_jump_label

    # ------------------------------------------------------------------
    # Execution loop
    # ------------------------------------------------------------------

    def resume(self, send_value: Any = None) -> StepEvent:
        if self.finished:
            return ReturnEvent(None)
        if self.block_label is None:
            self.block_label = self.function.region_ir.entry_label
        self.pending_send_value = send_value

        while not self.finished:
            event = self.dispatch_current_instruction()
            if event is not None:
                return event

        return ReturnEvent(self.return_value)

    def run_to_completion(self) -> Any:
        event = self.resume(send_value=None)
        if not isinstance(event, ReturnEvent):
            raise RuntimeError("frame yielded unexpectedly: %r" % (event,))
        return event.value

    # ------------------------------------------------------------------
    # Generator / coroutine wrappers
    # ------------------------------------------------------------------

    def make_generator_object(self) -> Generator[Any, Any, Any]:
        def generator() -> Generator[Any, Any, Any]:
            send_value = None
            while True:
                event = self.resume(send_value=send_value)
                if isinstance(event, YieldEvent):
                    try:
                        send_value = yield event.value
                    except GeneratorExit:
                        self.finished = True
                        raise
                elif isinstance(event, ReturnEvent):
                    return event.value
                else:
                    raise RuntimeError(
                        "unexpected generator event %r" % (event,)
                    )

        return generator()

    def make_coroutine_object(self) -> Coroutine[Any, Any, Any]:
        async def coroutine() -> Any:
            return self.run_to_completion()

        return coroutine()

    def make_async_generator_object(self) -> AsyncGenerator[Any, Any]:
        async def async_generator() -> AsyncGenerator[Any, Any]:
            send_value = None
            while True:
                event = self.resume(send_value=send_value)
                if isinstance(event, YieldEvent):
                    send_value = yield event.value
                elif isinstance(event, ReturnEvent):
                    return
                else:
                    raise RuntimeError(
                        "unexpected async generator event %r" % (event,)
                    )

        return async_generator()

    # ------------------------------------------------------------------
    # Calls
    # ------------------------------------------------------------------

    def call_callee(
        self, callee: Any, args: List[Any], kwargs: Dict[str, Any]
    ) -> Any:
        if callee is builtins.super and not args and not kwargs:
            class_cell = self.cells.get("__class__")
            if class_cell is None:
                raise RuntimeError("super(): __class__ cell not found")
            if class_cell.value is _UNSET:
                raise RuntimeError("super(): empty __class__ cell")
            if self.function.region_ir.argcount <= 0:
                raise RuntimeError("super(): no arguments")
            first_arg_name = self.function.region_ir.locals[0]
            if first_arg_name in self.locals:
                first_arg = self.locals[first_arg_name]
            elif (
                first_arg_name in self.cells
                and self.cells[first_arg_name].value is not _UNSET
            ):
                first_arg = self.cells[first_arg_name].value
            else:
                raise RuntimeError("super(): arg[0] deleted")
            return builtins.super(class_cell.value, first_arg)
        if callee is builtins.globals and not args and not kwargs:
            return self.globals
        if callee is builtins.locals and not args and not kwargs:
            return self.locals
        if callee is builtins.vars and not kwargs:
            if not args:
                return self.locals
            if len(args) == 1:
                return vars(args[0])
        if isinstance(callee, Function):
            frame = self.interpreter.make_frame(callee, tuple(args), kwargs)
            flags = callee.region_ir.flags
            if flags & inspect.CO_ASYNC_GENERATOR:
                return frame.make_async_generator_object()
            if flags & inspect.CO_COROUTINE:
                return frame.make_coroutine_object()
            if flags & inspect.CO_GENERATOR:
                return frame.make_generator_object()
            return frame.run_to_completion()
        return callee(*args, **kwargs)

    # ------------------------------------------------------------------
    # Exception handling
    # ------------------------------------------------------------------

    def handle_return(self, value: Any) -> ControlEvent:
        for index in range(len(self.try_stack) - 1, -1, -1):
            entry = self.try_stack[index]
            target = entry.get("finally_label")
            if target is None:
                continue
            self.pending_return_value = value
            del self.try_stack[index:]
            return JumpEvent(target)
        self.pending_return_value = _UNSET
        self.pending_jump_label = _UNSET
        return ReturnEvent(value)

    def handle_escape(self, target_label: BasicBlockLabel) -> JumpEvent:
        for index in range(len(self.try_stack) - 1, -1, -1):
            entry = self.try_stack[index]
            finally_label = entry.get("finally_label")
            if finally_label is None:
                continue
            self.pending_jump_label = target_label
            del self.try_stack[index:]
            return JumpEvent(finally_label)
        self.pending_jump_label = _UNSET
        self.pending_return_value = _UNSET
        return JumpEvent(target_label)

    def end_finally(self) -> Optional[ControlEvent]:
        if self.current_exception is not None:
            exc = self.current_exception
            handled = self.handle_exception(exc)
            if handled is not None:
                return handled
            raise exc
        if self.pending_return_value is not _UNSET:
            value = self.pending_return_value
            self.pending_return_value = _UNSET
            return self.handle_return(value)
        if self.pending_jump_label is not _UNSET:
            label = self.pending_jump_label
            self.pending_jump_label = _UNSET
            return self.handle_escape(label)
        return None

    # ===================================================================
    # Instruction handler methods  (self, instr) -> event | None
    # ===================================================================

    # --- value / variable ---

    def dispatch_const(self, instr: Const) -> None:
        self.temps[instr.dst] = instr.value
        return None

    def dispatch_load_name(self, instr: LoadName) -> None:
        self.temps[instr.dst] = self.load_name(instr.scope, instr.name)
        return None

    def dispatch_store_name(self, instr: StoreName) -> None:
        self.store_name(instr.scope, instr.name, self.temps[instr.src])
        return None

    def dispatch_delete_name(self, instr: DeleteName) -> None:
        self.delete_name(instr.scope, instr.name)
        return None

    # --- computation ---

    def dispatch_unary_op(self, instr: UnaryOp) -> None:
        self.temps[instr.dst] = apply_unary(
            instr.op, self.temps[instr.src]
        )
        return None

    def dispatch_binary_op(self, instr: BinaryOp) -> None:
        lhs = self.temps[instr.lhs]
        rhs = self.temps[instr.rhs]
        self.temps[instr.dst] = apply_binary(instr.op, lhs, rhs)
        return None

    def dispatch_compare_op(self, instr: CompareOp) -> None:
        lhs = self.temps[instr.lhs]
        rhs = self.temps[instr.rhs]
        self.temps[instr.dst] = apply_compare(instr.cmp, lhs, rhs)
        return None

    # --- attribute / item access ---

    def dispatch_load_attr(self, instr: LoadAttr) -> None:
        obj = self.temps[instr.obj]
        self.temps[instr.dst] = getattr(obj, instr.attr_name)
        return None

    def dispatch_store_attr(self, instr: StoreAttr) -> None:
        setattr(
            self.temps[instr.obj],
            instr.attr_name,
            self.temps[instr.value],
        )
        return None

    def dispatch_delete_attr(self, instr: DeleteAttr) -> None:
        delattr(self.temps[instr.obj], instr.attr_name)
        return None

    def dispatch_load_item(self, instr: LoadItem) -> None:
        obj = self.temps[instr.obj]
        key = self.temps[instr.key]
        self.temps[instr.dst] = obj[key]
        return None

    def dispatch_store_item(self, instr: StoreItem) -> None:
        obj = self.temps[instr.obj]
        key = self.temps[instr.key]
        value = self.temps[instr.value]
        obj[key] = value
        return None

    def dispatch_delete_item(self, instr: DeleteItem) -> None:
        del self.temps[instr.obj][self.temps[instr.key]]
        return None

    # --- aggregates ---

    def dispatch_build_tuple(self, instr: BuildTuple) -> None:
        built = []
        for item in instr.items:
            if isinstance(item, UnpackedTemporaryValue):
                built.extend(self.temps[item.value])
            else:
                built.append(self.temps[item])
        self.temps[instr.dst] = tuple(built)
        return None

    def dispatch_build_list(self, instr: BuildList) -> None:
        built = []
        for item in instr.items:
            if isinstance(item, UnpackedTemporaryValue):
                built.extend(self.temps[item.value])
            else:
                built.append(self.temps[item])
        self.temps[instr.dst] = built
        return None

    def dispatch_build_set(self, instr: BuildSet) -> None:
        built = set()
        for item in instr.items:
            if isinstance(item, UnpackedTemporaryValue):
                built.update(self.temps[item.value])
            else:
                built.add(self.temps[item])
        self.temps[instr.dst] = built
        return None

    def dispatch_build_map(self, instr: BuildMap) -> None:
        built = {}
        for key, value in instr.items:
            if key is None:
                built.update(dict(self.temps[value]))
            else:
                built[self.temps[key]] = self.temps[value]
        self.temps[instr.dst] = built
        return None

    def dispatch_build_slice(self, instr: BuildSlice) -> None:
        self.temps[instr.dst] = slice(
            self.temps[instr.start],
            self.temps[instr.stop],
            None if instr.step is None else self.temps[instr.step],
        )
        return None

    def dispatch_build_string(self, instr: BuildString) -> None:
        self.temps[instr.dst] = "".join(
            str(self.temps[part]) for part in instr.parts
        )
        return None

    def dispatch_format_value(self, instr: FormatValue) -> None:
        value = self.temps[instr.value]
        if instr.conversion == "repr":
            value = repr(value)
        elif instr.conversion == "ascii":
            value = ascii(value)
        else:
            value = str(value)
        if instr.spec is not None:
            value = format(value, self.temps[instr.spec])
        self.temps[instr.dst] = value
        return None

    def dispatch_unpack(self, instr: Unpack) -> None:
        values = list(self.temps[instr.src])
        if instr.star_index is None:
            if len(values) != len(instr.dsts):
                raise ValueError("unpack mismatch")
            for dst, value in zip(instr.dsts, values):
                self.temps[dst] = value
            return None
        if instr.star_index < 0 or instr.star_index >= len(instr.dsts):
            raise ValueError("invalid unpack star index")
        before_count = instr.star_index
        after_count = len(instr.dsts) - before_count - 1
        if len(values) < before_count + after_count:
            raise ValueError("unpack mismatch")
        for dst, value in zip(instr.dsts[:before_count], values[:before_count]):
            self.temps[dst] = value
        self.temps[instr.dsts[instr.star_index]] = values[
            before_count : len(values) - after_count
        ]
        for dst, value in zip(
            instr.dsts[before_count + 1 :], values[len(values) - after_count :]
        ):
            self.temps[dst] = value
        return None

    # --- calls / imports / functions / classes ---

    def dispatch_call(self, instr: Call) -> None:
        callee = self.temps[instr.callee]
        args = []
        for arg in instr.args:
            if isinstance(arg, UnpackedTemporaryValue):
                args.extend(self.temps[arg.value])
            else:
                args.append(self.temps[arg])
        kwargs = {}
        for name, value in instr.kwargs:
            resolved = self.temps[value]
            if name is None:
                for key, item in dict(resolved).items():
                    if key in kwargs:
                        raise TypeError(
                            "multiple values for keyword argument %r" % (key,)
                        )
                    kwargs[key] = item
            else:
                if name in kwargs:
                    raise TypeError(
                        "multiple values for keyword argument %r" % (name,)
                    )
                kwargs[name] = resolved
        self.temps[instr.dst] = self.call_callee(callee, args, kwargs)
        return None

    def dispatch_import_name(self, instr: ImportName) -> None:
        self.temps[instr.dst] = self.interpreter.import_module(
            self.globals, instr.module, list(instr.fromlist), instr.level
        )
        return None

    def dispatch_import_from(self, instr: ImportFrom) -> None:
        module_obj = self.temps[instr.module_obj]
        self.temps[instr.dst] = getattr(module_obj, instr.name)
        return None

    def dispatch_import_star(self, instr: ImportStar) -> None:
        module_obj = self.temps[instr.module_obj]
        export_names = getattr(module_obj, "__all__", None)
        if export_names is None:
            export_names = [
                name for name in vars(module_obj) if not name.startswith("_")
            ]
        for name in export_names:
            self.locals[name] = getattr(module_obj, name)
        return None

    def dispatch_make_function(self, instr: MakeFunction) -> None:
        region = next(
            r for r in self.function.region_ir.child_regions if r.label == instr.code
        )
        closure = {}
        for name in region.freevars:
            if name in self.cells:
                closure[name] = self.cells[name]
        fn = Function(
            region_ir=region,
            globals_dict=self.globals,
            closure_cells=closure,
            __defaults__=(
                tuple(self.temps[value] for value in instr.defaults)
                if instr.defaults
                else None
            ),
            __kwdefaults__=(
                {
                    name: self.temps[value]
                    for name, value in instr.kwdefaults
                }
                if instr.kwdefaults
                else None
            ),
            __annotations__=(
                {
                    name: self.temps[value]
                    for name, value in instr.annotations
                }
                if instr.annotations
                else {}
            ),
        )
        self.temps[instr.dst] = fn
        return None

    def dispatch_build_class(self, instr: BuildClass) -> None:
        body = self.temps[instr.body_func]
        name = self.temps[instr.name]
        bases = [self.temps[base] for base in instr.bases]
        keywords = {n: self.temps[v] for n, v in instr.keywords}
        self.temps[instr.dst] = self.interpreter.build_class(
            body, name, *bases, **keywords
        )
        return None

    # --- iteration / async ---

    def dispatch_get_iter(self, instr: GetIter) -> None:
        self.temps[instr.dst] = iter(self.temps[instr.iterable])
        return None

    def dispatch_for_iter(self, instr: ForIter) -> JumpEvent:
        iterator = self.temps[instr.iter_obj]
        try:
            value = next(iterator)
        except StopIteration:
            return JumpEvent(instr.exit_label)
        self.temps[instr.value_dst] = value
        return JumpEvent(instr.body_label)

    def dispatch_get_aiter(self, instr: GetAIter) -> None:
        self.temps[instr.dst] = self.temps[instr.iterable].__aiter__()
        return None

    def dispatch_get_anext(self, instr: GetANext) -> None:
        self.temps[instr.dst] = self.temps[instr.aiter].__anext__()
        return None

    def dispatch_get_awaitable(self, instr: GetAwaitable) -> None:
        value = self.temps[instr.value]
        self.temps[instr.dst] = value if inspect.isawaitable(value) else value
        return None

    def dispatch_yield_value(self, instr: YieldValue) -> YieldEvent:
        yielded = self.temps[instr.value]
        sent_value = (
            None if self.pending_send_value is _UNSET else self.pending_send_value
        )
        self.temps[instr.dst] = sent_value
        self.pending_send_value = _UNSET
        return YieldEvent(yielded)

    def dispatch_yield_from(self, instr: YieldFrom) -> YieldEvent:
        yielded = self.temps[instr.value]
        self.temps[instr.dst] = None
        return YieldEvent(yielded)

    def dispatch_await_value(self, instr: AwaitValue) -> None:
        self.temps[instr.dst] = await_sync(self.temps[instr.value])
        return None

    # --- exceptions ---

    def dispatch_current_exception(self, instr: CurrentException) -> None:
        self.temps[instr.dst] = self.current_exception
        return None

    def dispatch_raise(self, instr: Raise) -> None:
        exc = normalize_exception_for_raise(self.temps[instr.exc])
        if instr.cause is not None:
            cause = normalize_exception_for_raise(
                self.temps[instr.cause], allow_none=True
            )
            exc.__cause__ = cause
            exc.__suppress_context__ = True
        raise exc

    def dispatch_reraise(self, instr: Reraise) -> None:
        if self.current_exception is None:
            raise RuntimeError("no current exception to reraise")
        raise self.current_exception

    def dispatch_check_exc_match(self, instr: CheckExcMatch) -> None:
        exc = self.temps[instr.exc]
        typ = self.temps[instr.typ]
        self.temps[instr.dst] = check_exception_match(exc, typ)
        return None

    def dispatch_check_eg_match(self, instr: CheckEGMatch) -> None:
        self.temps[instr.dst] = False
        return None

    def dispatch_push_try(self, instr: PushTry) -> None:
        self.try_stack.append(
            {
                "except_label": instr.except_label,
                "finally_label": instr.finally_label,
            }
        )
        return None

    def dispatch_pop_try(self, instr: PopTry) -> None:
        if self.try_stack:
            self.try_stack.pop()
        return None

    def dispatch_clear_exception(self, instr: ClearException) -> None:
        self.current_exception = None
        return None

    # --- control flow ---

    def dispatch_end_finally(
        self, instr: EndFinally
    ) -> Optional[ControlEvent]:
        return self.end_finally()

    def dispatch_escape(self, instr: Escape) -> JumpEvent:
        return self.handle_escape(instr.target)

    def dispatch_jump(self, instr: Jump) -> JumpEvent:
        return JumpEvent(instr.target)

    def dispatch_branch(self, instr: Branch) -> JumpEvent:
        cond = self.temps[instr.cond]
        return JumpEvent(instr.true_label if cond else instr.false_label)

    def dispatch_return(self, instr: Return) -> ControlEvent:
        value = self.temps[instr.value]
        self.current_exception = None
        return self.handle_return(value)

    # --- pattern matching ---

    def dispatch_match_mapping(self, instr: MatchMapping) -> None:
        value = self.temps[instr.value]
        self.temps[instr.dst] = isinstance(value, abc.Mapping)
        return None

    def dispatch_match_sequence(self, instr: MatchSequence) -> None:
        value = self.temps[instr.value]
        self.temps[instr.dst] = (
            isinstance(value, abc.Sequence)
            and not isinstance(value, (str, bytes, bytearray))
        )
        return None

    def dispatch_match_keys(self, instr: MatchKeys) -> None:
        mapping = self.temps[instr.mapping]
        keys = self.temps[instr.keys]
        try:
            result = tuple(mapping[key] for key in keys)
        except Exception:
            result = None
        self.temps[instr.dst] = result
        return None

    def dispatch_match_class(self, instr: MatchClass) -> None:
        value = self.temps[instr.value]
        cls = self.temps[instr.cls]
        self.temps[instr.dst] = isinstance(value, cls)
        return None


# ---------------------------------------------------------------------------
# Argument binding (module-level helper, no interpreter / frame dependency)
# ---------------------------------------------------------------------------


def bind_arguments(
    function: Function, args: Tuple[Any, ...], kwargs: Dict[str, Any]
) -> Dict[str, Any]:
    """Bind Python call arguments to local names. Returns a dict."""
    region = function.region_ir
    positional = region.argcount
    posonly = region.posonlyargcount
    kwonly = region.kwonlyargcount
    names = list(region.locals[: positional + kwonly])
    bound = {}
    kwargs = dict(kwargs)

    positional_names = names[:positional]
    posonly_names = positional_names[:posonly]
    kwonly_names = names[positional : positional + kwonly]

    if len(args) > len(positional_names) and region.vararg_name is None:
        raise TypeError(
            "too many positional arguments for %s" % (function.__name__,)
        )

    consumed_positional = positional_names[
        : min(len(args), len(positional_names))
    ]
    for name, value in zip(consumed_positional, args):
        bound[name] = value

    posonly_keyword_names = [name for name in posonly_names if name in kwargs]
    if posonly_keyword_names:
        raise TypeError(
            "positional-only arguments passed by keyword for %s: %s"
            % (function.__name__, sorted(posonly_keyword_names))
        )

    duplicate_names = [name for name in consumed_positional if name in kwargs]
    if duplicate_names:
        raise TypeError(
            "multiple values for arguments for %s: %s"
            % (function.__name__, sorted(duplicate_names))
        )

    for name in positional_names[len(consumed_positional) :]:
        if name in kwargs:
            bound[name] = kwargs.pop(name)

    if function.__defaults__:
        default_names = positional_names[-len(function.__defaults__) :]
        for name, value in zip(default_names, function.__defaults__):
            bound.setdefault(name, value)

    for name in positional_names:
        if name not in bound:
            raise TypeError(
                "missing argument %r for %s" % (name, function.__name__)
            )

    extra_args = args[len(positional_names) :]
    if region.vararg_name is not None:
        bound[region.vararg_name] = tuple(extra_args)

    for name in kwonly_names:
        if name in kwargs:
            bound[name] = kwargs.pop(name)
        elif function.__kwdefaults__ and name in function.__kwdefaults__:
            bound[name] = function.__kwdefaults__[name]
        else:
            raise TypeError(
                "missing keyword-only argument %r for %s"
                % (name, function.__name__)
            )

    if region.kwarg_name is not None:
        bound[region.kwarg_name] = dict(kwargs)
    elif kwargs:
        raise TypeError(
            "unexpected keyword arguments for %s: %s"
            % (function.__name__, sorted(kwargs))
        )

    return bound
