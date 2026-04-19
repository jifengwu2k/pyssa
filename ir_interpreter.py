"""IR interpreter and convenience entrypoints.

This module owns execution of the project IR. The lowering pipeline and IR data
model remain in `pyssa.py`.
"""

from __future__ import annotations

import argparse
import ast
import builtins
from collections.abc import Mapping, Sequence
import importlib
import importlib.machinery
import importlib.util
import inspect
import operator
import os
import sys
import traceback
from typing import Any, Dict, Optional

import attrs

from multiline_input import enable_readline_if_available, multiline_input
from pyssa import (
    AwaitValue,
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
    compile_file,
    compile_source,
    new_compiler_state,
    render_instruction,
)


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
    function: Any
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


_RUNTIME_REGION_ATTR = "__pyssa_region_ir__"
_RUNTIME_GLOBALS_ATTR = "__pyssa_globals__"
_RUNTIME_CLOSURE_ATTR = "__pyssa_closure_cells__"
_RUNTIME_PRELOADED_LOCALS_ATTR = "__pyssa_preloaded_locals__"


def is_runtime_function(value):
    return inspect.isfunction(value) and hasattr(value, _RUNTIME_REGION_ATTR)


def runtime_function_region(function):
    return getattr(function, _RUNTIME_REGION_ATTR)


def runtime_function_globals(function):
    return getattr(function, _RUNTIME_GLOBALS_ATTR)


def runtime_function_closure_cells(function):
    return getattr(function, _RUNTIME_CLOSURE_ATTR)


def runtime_function_preloaded_locals(function):
    return getattr(function, _RUNTIME_PRELOADED_LOCALS_ATTR)


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
        module_function = self.make_runtime_function(module_ir, globals_dict, qualname=qualname)
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

    def make_runtime_function(self, region_ir, globals_dict, closure_cells=None, qualname=None, preloaded_locals=None):
        closure_cells = dict(closure_cells or {})
        preloaded_locals = dict(preloaded_locals or {})
        name = region_ir.name.split("#", 1)[0]
        interpreter = self

        def runtime_function(*args, **kwargs):
            return interpreter.call_function(runtime_function, args, kwargs)

        runtime_function.__name__ = name
        runtime_function.__qualname__ = qualname or name
        runtime_function.__module__ = globals_dict.get("__name__", "__main__")
        runtime_function.__defaults__ = None
        runtime_function.__kwdefaults__ = None
        runtime_function.__annotations__ = {}
        setattr(runtime_function, _RUNTIME_REGION_ATTR, region_ir)
        setattr(runtime_function, _RUNTIME_GLOBALS_ATTR, globals_dict)
        setattr(runtime_function, _RUNTIME_CLOSURE_ATTR, closure_cells)
        setattr(runtime_function, _RUNTIME_PRELOADED_LOCALS_ATTR, preloaded_locals)
        return runtime_function

    def make_generator_object(self, frame):
        interpreter = self

        def generator():
            send_value = None
            while True:
                kind, value = interpreter.resume_frame(frame, send_value=send_value)
                if kind == "yield":
                    try:
                        send_value = (yield value)
                    except GeneratorExit:
                        frame.finished = True
                        raise
                elif kind == "return":
                    return value
                else:
                    raise RuntimeError("unexpected generator event %r" % ((kind, value),))

        return generator()

    def make_coroutine_object(self, frame):
        interpreter = self

        async def coroutine():
            return interpreter.run_to_completion(frame)

        return coroutine()

    def make_async_generator_object(self, frame):
        interpreter = self

        async def async_generator():
            send_value = None
            while True:
                kind, value = interpreter.resume_frame(frame, send_value=send_value)
                if kind == "yield":
                    send_value = (yield value)
                elif kind == "return":
                    return
                else:
                    raise RuntimeError("unexpected async generator event %r" % ((kind, value),))

        return async_generator()

    def build_class(self, body_function, name, *bases, **kwargs):
        # Execute the lowered class body against the metaclass-prepared namespace.
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
        namespace["__module__"] = runtime_function_globals(body_function).get("__name__", "__main__")
        namespace["__qualname__"] = name
        body_function.__qualname__ = name
        region_ir = runtime_function_region(body_function)
        cells = dict(runtime_function_closure_cells(body_function))
        for cell_name in region_ir.cells:
            if cell_name not in cells:
                cells[cell_name] = Cell(namespace.get(cell_name, _UNSET))
        frame = Frame(
            interpreter=self,
            function=body_function,
            function_ir=region_ir,
            globals=runtime_function_globals(body_function),
            locals=namespace,
            cells=cells,
            block_label=region_ir.entry_label,
            instr_index=0,
        )
        self.run_to_completion(frame)
        for special_name in ("__init_subclass__", "__class_getitem__"):
            value = namespace.get(special_name)
            if is_runtime_function(value):
                namespace[special_name] = classmethod(value)
        cls = metaclass(name, bases, namespace, **kwargs)
        class_cell = frame.cells.get("__class__")
        if class_cell is not None:
            class_cell.value = cls
        return cls

    def call_function(self, function, args, kwargs):
        # Pick the runtime wrapper that matches the code object's generator/coroutine flags.
        region_ir = runtime_function_region(function)
        flags = region_ir.flags
        frame = self.make_frame(function, args, kwargs)

        if flags & inspect.CO_ASYNC_GENERATOR:
            return self.make_async_generator_object(frame)
        if flags & inspect.CO_COROUTINE:
            return self.make_coroutine_object(frame)
        if flags & inspect.CO_GENERATOR:
            return self.make_generator_object(frame)
        return self.run_to_completion(frame)

    def call_zero_arg_super(self, frame):
        class_cell = frame.cells.get("__class__")
        if class_cell is None:
            raise RuntimeError("super(): __class__ cell not found")
        if class_cell.value is _UNSET:
            raise RuntimeError("super(): empty __class__ cell")
        if frame.function_ir.argcount <= 0:
            raise RuntimeError("super(): no arguments")
        first_arg_name = frame.function_ir.locals[0]
        if first_arg_name in frame.locals:
            first_arg = frame.locals[first_arg_name]
        elif first_arg_name in frame.cells and frame.cells[first_arg_name].value is not _UNSET:
            first_arg = frame.cells[first_arg_name].value
        else:
            raise RuntimeError("super(): arg[0] deleted")
        return builtins.super(class_cell.value, first_arg)

    def call_runtime(self, frame, callee, args, kwargs):
        if callee is builtins.super and not args and not kwargs:
            return self.call_zero_arg_super(frame)
        if callee is builtins.globals and not args and not kwargs:
            return frame.globals
        if callee is builtins.locals and not args and not kwargs:
            return frame.locals
        if callee is builtins.vars and not kwargs:
            if not args:
                return frame.locals
            if len(args) == 1:
                return vars(args[0])
        return callee(*args, **kwargs)

    def make_frame(self, function, args, kwargs):
        # Materialize locals and closure cells for one function invocation.
        region_ir = runtime_function_region(function)
        locals_dict = self.bind_arguments(function, args, kwargs)
        locals_dict.update(runtime_function_preloaded_locals(function))
        cells = dict(runtime_function_closure_cells(function))
        for name in region_ir.cells:
            if name not in cells:
                cells[name] = Cell(locals_dict.get(name, _UNSET))
        return Frame(
            interpreter=self,
            function=function,
            function_ir=region_ir,
            globals=runtime_function_globals(function),
            locals=locals_dict,
            cells=cells,
            block_label=region_ir.entry_label,
            instr_index=0,
        )

    def bind_arguments(self, function, args, kwargs):
        # Bind Python arguments using the structural signature recorded on the Region.
        region = runtime_function_region(function)
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

        if function.__defaults__:
            default_names = positional_names[-len(function.__defaults__):]
            for name, value in zip(default_names, function.__defaults__):
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
            elif function.__kwdefaults__ and name in function.__kwdefaults__:
                bound[name] = function.__kwdefaults__[name]
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
            return self.exec_value_instr(frame, instr, self.call_runtime(frame, callee, args, kwargs))

        if isinstance(instr, ImportName):
            module = self.import_module(frame, instr.module, list(instr.fromlist), instr.level)
            return self.exec_value_instr(frame, instr, module)

        if isinstance(instr, ImportFrom):
            module_obj = self.resolve_value(frame, instr.module_obj)
            return self.exec_value_instr(frame, instr, getattr(module_obj, instr.name))

        if isinstance(instr, ImportStar):
            module_obj = self.resolve_value(frame, instr.module_obj)
            export_names = getattr(module_obj, "__all__", None)
            if export_names is None:
                export_names = [name for name in vars(module_obj) if not name.startswith("_")]
            for name in export_names:
                frame.locals[name] = getattr(module_obj, name)
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
            fn = self.make_runtime_function(region, frame.globals, closure_cells=closure, qualname=qualname)
            if instr.defaults:
                fn.__defaults__ = tuple(self.resolve_value(frame, value) for value in instr.defaults)
            if instr.kwdefaults:
                fn.__kwdefaults__ = {name: self.resolve_value(frame, value) for name, value in instr.kwdefaults}
            if instr.annotations:
                fn.__annotations__ = {name: self.resolve_value(frame, value) for name, value in instr.annotations}
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
            matched = self.check_exception_match(exc, typ)
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
        if inspect.isawaitable(value):
            iterator = value.__await__()
            send_value = None
            while True:
                try:
                    yielded = iterator.send(send_value)
                except StopIteration as stop:
                    return stop.value
                if inspect.isawaitable(yielded):
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

    def is_valid_exception_match_type(self, typ):
        if isinstance(typ, type):
            return issubclass(typ, BaseException)
        if isinstance(typ, tuple):
            return all(self.is_valid_exception_match_type(item) for item in typ)
        return False

    def check_exception_match(self, exc, typ):
        if not self.is_valid_exception_match_type(typ):
            raise TypeError("catching classes that do not inherit from BaseException is not allowed")
        return isinstance(exc, typ)

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


def exec_ir(module_ir, globals=None, locals=None, hooks=(), module_name="__main__", module_path=None, search_path=None):
    # Convenience entry point used by tests and comparison tools.
    interpreter = IRInterpreter(
        module_ir,
        hooks=hooks,
        module_name=module_name,
        module_path=module_path,
        search_path=search_path,
    )
    return interpreter.exec(globals=globals, locals=locals)


def run_file(path, hooks=(), argv=None):
    abs_path = os.path.abspath(path)
    module_ir = compile_file(new_compiler_state(), abs_path)
    globals_dict = {"__name__": "__main__", "__file__": abs_path}
    old_argv = sys.argv
    sys.argv = [path] if argv is None else list(argv)
    try:
        return exec_ir(module_ir, globals_dict, globals_dict, hooks=hooks, module_path=abs_path)
    finally:
        sys.argv = old_argv


def interact(hooks=()):
    namespace = {"__name__": "__main__", "__package__": None}
    result_name = "__ir_interpreter_repl_value__"

    def execute_source(source):
        try:
            try:
                ast.parse(source, mode="eval")
            except SyntaxError:
                module_ir = compile_source(new_compiler_state(), source, path="<stdin>")
                exec_ir(module_ir, namespace, namespace, hooks=hooks)
            else:
                wrapped_source = "%s = (\n%s\n)\n" % (result_name, source)
                module_ir = compile_source(new_compiler_state(), wrapped_source, path="<stdin>")
                exec_ir(module_ir, namespace, namespace, hooks=hooks)
                value = namespace.pop(result_name)
                namespace["_"] = value
                print(repr(value))
        except SystemExit:
            raise
        except BaseException:
            traceback.print_exc()
            namespace.pop(result_name, None)

    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        source = sys.stdin.read()
        if source:
            execute_source(source)
        return

    enable_readline_if_available()
    while True:
        print("Enter code, then press Ctrl-D to run it. Press Ctrl-D on an empty prompt to exit.")
        try:
            source = multiline_input()
        except EOFError:
            break
        except KeyboardInterrupt:
            print()
            continue
        execute_source(source)


def main(argv=None):
    if argv is None:
        argv = sys.argv

    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true", help="print executed IR instructions")
    parser.add_argument("python_file", nargs="?")
    parser.add_argument("file_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(list(argv[1:]))

    hooks = (VerboseTraceHook(),) if args.verbose else ()
    if args.python_file is not None:
        run_file(args.python_file, hooks=hooks, argv=[args.python_file, *args.file_args])
        return
    interact(hooks=hooks)


if __name__ == "__main__":
    main()
