"""Compatibility entrypoint for AST -> IR compilation.

`pyssa.py` now contains the implementation, but this file remains as the
CLI/module entrypoint.
"""

import sys

from pyssa import compile_file, compile_source, new_compiler_state, print_region_ir


def main(argv=None):
    if argv is None:
        argv = sys.argv
    if len(argv) != 2:
        raise SystemExit("usage: python compile_to_ir.py <python-file>")

    path = argv[1]
    state = new_compiler_state()
    module_ir = compile_file(state, path)
    print_region_ir(module_ir)


if __name__ == "__main__":
    main()
