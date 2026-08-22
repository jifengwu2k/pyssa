"""Regression tests for the pyssa IR/compiler/interpreter library.

These tests encode the semantics the compiler should produce and preserve
regressions fixed in the AST lowering and interpreter support code.
"""

import sys
import unittest

from pyssa.compiler import compile_source, new_compiler_state
from pyssa.ir import (
    BinaryOp,
    BinaryOperator,
    Call,
    CodeFlag,
    ComparisonOperator,
    CompareOp,
    Const,
    DeleteName,
    Escape,
    FormatConversion,
    FormatValue,
    ImportName,
    LoadItem,
    LoadName,
    MakeFunction,
    PopTry,
    PushTry,
    Return,
    StoreItem,
    StoreName,
    UnaryOp,
    UnaryOperator,
)


def compile_region(source: str):
    return compile_source(new_compiler_state(), source)


def all_instructions(region):
    out = []
    for block in region.basic_blocks:
        out.extend(block.instructions)
    return out


def all_blocks(region):
    return list(region.basic_blocks)


def find_single(region, cls):
    found = [i for i in all_instructions(region) if isinstance(i, cls)]
    if len(found) != 1:
        raise AssertionError("expected exactly one %s, found %d" % (cls.__name__, len(found)))
    return found[0]


class CompilerSmokeTests(unittest.TestCase):
    def test_module_implicit_return_none(self):
        region = compile_region("x = 1\n")
        instrs = all_instructions(region)
        self.assertIsInstance(instrs[-1], Return)
        self.assertTrue(
            any(isinstance(i, Const) and i.value is None for i in instrs)
        )

    def test_structured_ir_enums(self):
        region = compile_region(
            "negative = -x\n"
            "total = x + y\n"
            "same = x is y\n"
            "rendered = f'{x!r}'\n"
        )
        unary = find_single(region, UnaryOp)
        binary = find_single(region, BinaryOp)
        comparison = find_single(region, CompareOp)
        formatted = find_single(region, FormatValue)
        self.assertEqual(unary.op, UnaryOperator.NEGATIVE)
        self.assertEqual(binary.op, BinaryOperator.ADD)
        self.assertEqual(comparison.cmp, ComparisonOperator.IS)
        self.assertEqual(formatted.conversion, FormatConversion.REPR)

    def test_augassign_evaluation_order(self):
        # Python evaluates the target object/key and reads the current value
        # before evaluating the RHS: a(), b(), LoadItem, c(), then store.
        region = compile_region("a()[b()] += c()\n")
        events = []
        for instr in all_instructions(region):
            if isinstance(instr, LoadName) and isinstance(instr.name, str):
                events.append(instr.name)
            elif isinstance(instr, Call):
                events.append("call")
            elif isinstance(instr, LoadItem):
                events.append("loaditem")
            elif isinstance(instr, BinaryOp):
                events.append("binop")
            elif isinstance(instr, StoreItem):
                events.append("storeitem")
                break
        self.assertEqual(events, ["a", "call", "b", "call", "loaditem", "c", "call", "binop", "storeitem"])

    def test_decorator_evaluation_order(self):
        # Decorator expressions are evaluated in source order, before defaults.
        region = compile_region("@d1\n@d2\ndef f(x=a()):\n    pass\n")
        names = [
            i.name
            for i in all_instructions(region)
            if isinstance(i, LoadName) and isinstance(i.name, str)
        ]
        make = find_single(region, MakeFunction)
        # d1 and d2 are evaluated first, then a() for the default.
        self.assertEqual(names[:3], ["d1", "d2", "a"])
        self.assertTrue(all_instructions(region).index(make) > 2)

    def test_chained_comparison_evaluation_order(self):
        region = compile_region("r = a() < b() < c()\n")
        names = [
            i.name
            for i in all_instructions(region)
            if isinstance(i, LoadName)
            and isinstance(i.name, str)
            and not i.name.startswith("<synthetic:")
        ]
        self.assertEqual(names, ["a", "b", "c"])


class CompilerRegressionTests(unittest.TestCase):
    def test_import_dotted_loads_submodule_and_binds_top_level_package(self):
        region = compile_region("import os.path\n")
        imports = [
            i for i in all_instructions(region) if isinstance(i, ImportName)
        ]
        self.assertEqual([i.module for i in imports], ["os.path", "os"])
        store = find_single(region, StoreName)
        self.assertEqual(store.name, "os")
        self.assertEqual(store.src, imports[-1].dst)

    def test_return_in_try_finally_reaches_finally(self):
        # A return inside try/finally must execute the finally block first.
        region = compile_region(
            "def f():\n    try:\n        return 1\n    finally:\n        y = 2\n"
        )
        fn = region.child_regions[0]
        push = find_single(fn, PushTry)
        self.assertIsNotNone(push.finally_label)
        value_block = next(
            b
            for b in all_blocks(fn)
            if any(isinstance(i, Const) and i.value == 1 for i in b.instructions)
        )
        # Correct lowering must not terminate the try body in Return before
        # running the finally block.
        self.assertNotIsInstance(value_block.instructions[-1], Return)

    def test_break_in_try_finally_reaches_finally(self):
        # A break inside try/finally must execute the finally block first.
        region = compile_region(
            "def f():\n"
            "    for i in range(3):\n"
            "        try:\n"
            "            break\n"
            "        finally:\n"
            "            y = 1\n"
        )
        fn = region.child_regions[0]
        push = find_single(fn, PushTry)
        self.assertIsNotNone(push.finally_label)
        escape_block = next(
            b for b in all_blocks(fn) if any(isinstance(i, Escape) for i in b.instructions)
        )
        # The block that performs the break must run the finally cleanup
        # (``y = 1``) before the break escape.
        self.assertTrue(
            any(
                isinstance(i, StoreName) and i.name == "y"
                for i in escape_block.instructions
            )
        )

    def test_return_in_with_reaches_exit(self):
        # A return inside a with statement must call __exit__ first.
        region = compile_region(
            "def f():\n    with mgr():\n        return 1\n"
        )
        fn = region.child_regions[0]
        push = find_single(fn, PushTry)
        self.assertIsNotNone(push.finally_label)
        value_block = next(
            b
            for b in all_blocks(fn)
            if any(isinstance(i, Const) and i.value == 1 for i in b.instructions)
        )
        self.assertNotIsInstance(value_block.instructions[-1], Return)

    def test_pattern_match_guard_failure_keeps_bindings(self):
        # Captures are made before guard evaluation and remain visible if the
        # guard is false, matching Python's match statement semantics.
        region = compile_region(
            "match x:\n"
            "    case (a, b) if y:\n"
            "        pass\n"
            "    case _:\n"
            "        pass\n"
        )
        deleted = {
            i.name
            for i in all_instructions(region)
            if isinstance(i, DeleteName) and isinstance(i.name, str)
        }
        self.assertNotIn("a", deleted)
        self.assertNotIn("b", deleted)

    def test_annotation_containing_lambda_is_supported(self):
        # Annotations are ordinary expressions in their own region; a lambda
        # inside one should lower instead of failing on missing child tables.
        try:
            compile_region("def f(x: (lambda: int)):\n    pass\n")
        except Exception as exc:  # pragma: no cover - current failure path
            self.fail("annotation lambdas should be supported, got %r" % (exc,))

    def test_synthetic_locals_use_string_names(self):
        # LoadName/StoreName/DeleteName declare ``name: str``; compiler
        # synthetic temporaries currently violate that contract.
        region = compile_region("for x in y:\n    pass\n")
        for instr in all_instructions(region):
            if isinstance(instr, (LoadName, StoreName, DeleteName)):
                self.assertIsInstance(instr.name, str)

    def test_make_function_flags_carry_async(self):
        region = compile_region("async def f():\n    return 1\n")
        make = find_single(region, MakeFunction)
        self.assertTrue(make.flags & CodeFlag.COROUTINE)

    def test_nested_function_return_does_not_use_parent_finally(self):
        region = compile_region(
            "try:\n"
            "    def f():\n"
            "        return 1\n"
            "finally:\n"
            "    x = 2\n"
        )
        fn = region.child_regions[0]
        self.assertIsInstance(fn.basic_blocks[0].instructions[-1], Return)
        labels = [block.label for block in region.basic_blocks]
        self.assertEqual(len(labels), len(set(labels)))

    def test_finally_with_nested_function_and_return_compiles(self):
        region = compile_region(
            "def f():\n"
            "    try:\n"
            "        return 1\n"
            "    finally:\n"
            "        def g():\n"
            "            pass\n"
        )
        self.assertTrue(any(child.name.startswith("g") for child in region.child_regions[0].child_regions))

    def test_return_from_except_pops_only_finally_handler(self):
        region = compile_region(
            "def f():\n"
            "    try:\n"
            "        1 / 0\n"
            "    except Exception:\n"
            "        return 1\n"
            "    finally:\n"
            "        x = 2\n"
        )
        fn = region.child_regions[0]
        cleanup_return = next(
            block
            for block in fn.basic_blocks
            if isinstance(block.instructions[-1], Return)
            and any(
                isinstance(i, StoreName) and i.name == "x"
                for i in block.instructions
            )
        )
        self.assertEqual(
            sum(isinstance(i, PopTry) for i in cleanup_return.instructions), 1
        )

    @unittest.skipUnless(sys.version_info >= (3, 12), "PEP 695 requires Python 3.12")
    def test_pep695_generic_function_and_class_compile(self):
        region = compile_region(
            "def identity[T](value: T) -> T:\n"
            "    return value\n"
            "class Box[T]:\n"
            "    pass\n"
        )
        child_names = [child.name for child in region.child_regions]
        self.assertIn("identity", child_names)
        self.assertIn("Box", child_names)


if __name__ == "__main__":
    unittest.main()
