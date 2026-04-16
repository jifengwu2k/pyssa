"""Compare CPython execution with direct IR execution for one source file.

This is a debugging tool: it runs the same source through Python and through the IR
frontend/interpreter, then compares a normalized snapshot of the resulting namespaces.
"""

import argparse
import inspect
import os
import sys
import types
from pprint import pformat

from pyssa import compile_source, new_compiler_state
from ir_interpreter import exec_ir


IGNORE_KEYS = {
    "__builtins__",
    "__build_class__",
}


def normalize(value, seen=None):
    # Convert runtime objects into a deterministic, comparable representation.
    if seen is None:
        seen = set()

    obj_id = id(value)
    if obj_id in seen:
        return {"kind": "cycle"}

    if value is None or isinstance(value, (bool, int, float, str, bytes)):
        return value

    if isinstance(value, tuple):
        seen.add(obj_id)
        return tuple(normalize(item, seen) for item in value)

    if isinstance(value, list):
        seen.add(obj_id)
        return [normalize(item, seen) for item in value]

    if isinstance(value, set):
        seen.add(obj_id)
        return {
            "kind": "set",
            "items": sorted(normalize(item, seen) for item in value),
        }

    if isinstance(value, dict):
        seen.add(obj_id)
        items = []
        for key in sorted(value.keys(), key=repr):
            if key in IGNORE_KEYS:
                continue
            items.append((normalize(key, seen), normalize(value[key], seen)))
        return {"kind": "dict", "items": items}

    if inspect.isfunction(value) or inspect.ismethod(value) or callable(value) and type(value).__name__ in ("IRFunction", "BoundIRMethod"):
        return {
            "kind": "function",
            "name": getattr(value, "__name__", type(value).__name__),
            "qualname": getattr(value, "__qualname__", getattr(value, "__name__", type(value).__name__)),
            "defaults": normalize(getattr(value, "__defaults__", None), seen),
            "kwdefaults": normalize(getattr(value, "__kwdefaults__", None), seen),
        }

    if inspect.isclass(value):
        attrs = {}
        for key, attr in sorted(value.__dict__.items()):
            if key.startswith("__") and key.endswith("__"):
                continue
            attrs[key] = normalize(attr, seen)
        return {
            "kind": "class",
            "name": value.__name__,
            "qualname": value.__qualname__,
            "bases": [base.__name__ for base in value.__bases__],
            "attrs": attrs,
        }

    if isinstance(value, types.CodeType):
        return {
            "kind": "code",
            "name": value.co_name,
            "qualname": getattr(value, "co_qualname", value.co_name),
        }

    if hasattr(value, "__dict__"):
        seen.add(obj_id)
        return {
            "kind": "object",
            "type": type(value).__name__,
            "attrs": normalize(vars(value), seen),
        }

    return {
        "kind": "repr",
        "type": type(value).__name__,
        "repr": repr(value),
    }


def snapshot_namespace(namespace):
    # Snapshot a namespace after filtering helper entries injected by the runtimes.
    result = {}
    for key in sorted(namespace.keys()):
        if key in IGNORE_KEYS:
            continue
        result[key] = normalize(namespace[key])
    return result


def compare_namespaces(expected, actual):
    # Produce a readable list of semantic mismatches between two normalized namespaces.
    mismatches = []
    keys = sorted(set(expected) | set(actual))
    for key in keys:
        if key not in expected:
            mismatches.append("extra key in IR namespace: %s" % key)
            continue
        if key not in actual:
            mismatches.append("missing key in IR namespace: %s" % key)
            continue
        if expected[key] != actual[key]:
            mismatches.append(
                "value mismatch for %s\nEXPECTED:\n%s\nACTUAL:\n%s" % (
                    key,
                    pformat(expected[key]),
                    pformat(actual[key]),
                )
            )
    return mismatches


def main():
    # Run both execution paths and report any observable differences.
    parser = argparse.ArgumentParser()
    parser.add_argument("python_file")
    parser.add_argument("--separate-locals", action="store_true")
    args = parser.parse_args()

    with open(args.python_file, "r") as f:
        source = f.read()

    code = compile(source, args.python_file, "exec")
    module_ir = compile_source(new_compiler_state(), source, path=args.python_file)

    module_dir = os.path.dirname(os.path.abspath(args.python_file))
    sys.path.insert(0, module_dir)
    try:
        py_globals = {"__name__": "__main__", "__file__": args.python_file}
        py_locals = {} if args.separate_locals else py_globals
        exec(code, py_globals, py_locals)

        ir_globals = {"__name__": "__main__", "__file__": args.python_file}
        ir_locals = {} if args.separate_locals else ir_globals
        exec_ir(module_ir, ir_globals, ir_locals, module_path=args.python_file)
    finally:
        del sys.path[0]

    expected_globals = snapshot_namespace(py_globals)
    actual_globals = snapshot_namespace(ir_globals)
    global_mismatches = compare_namespaces(expected_globals, actual_globals)

    expected_locals = snapshot_namespace(py_locals)
    actual_locals = snapshot_namespace(ir_locals)
    local_mismatches = compare_namespaces(expected_locals, actual_locals)

    if not global_mismatches and not local_mismatches:
        print("IR execution matches exec() for globals and locals.")
        return

    if global_mismatches:
        print("Global mismatches:")
        for mismatch in global_mismatches:
            print("- %s" % mismatch)

    if local_mismatches:
        print("Local mismatches:")
        for mismatch in local_mismatches:
            print("- %s" % mismatch)

    raise SystemExit(1)


if __name__ == "__main__":
    main()
