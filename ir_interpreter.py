"""Compatibility entrypoint and convenience API for IR execution.

`pyssa.py` contains the core IR/interpreter implementation, while this file
remains the public interpreter-facing module.
"""

import argparse
import ast
import os
import sys
import traceback

from multiline_input import enable_readline_if_available, multiline_input
from pyssa import IRInterpreter, VerboseTraceHook, compile_file, compile_source, new_compiler_state


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


def run_file(path, hooks=()):
    abs_path = os.path.abspath(path)
    module_ir = compile_file(new_compiler_state(), abs_path)
    globals_dict = {"__name__": "__main__", "__file__": abs_path}
    return exec_ir(module_ir, globals_dict, globals_dict, hooks=hooks, module_path=abs_path)


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
    parser.add_argument("python_file", nargs="?")
    parser.add_argument("--verbose", action="store_true", help="print executed IR instructions")
    args = parser.parse_args(list(argv[1:]))

    hooks = (VerboseTraceHook(),) if args.verbose else ()
    if args.python_file is not None:
        run_file(args.python_file, hooks=hooks)
        return
    interact(hooks=hooks)


if __name__ == "__main__":
    main()
