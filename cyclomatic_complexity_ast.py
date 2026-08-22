"""Print cyclomatic complexity per nested Python scope using the AST.

This is a structural AST-based comparison tool next to the IR-based
`cyclomatic_complexity.py` tool. It emits a recursive JSON tree with the shape:

    {
      "name": str,
      "cyclomatic_complexity": int,
      "child_regions": [...]
    }
"""

import argparse
import ast
import json
import sys
from typing import Dict, List, Optional, Sequence

# ``ast.MatchAs`` only exists on Python 3.10+; guard it for older interpreters.
if sys.version_info >= (3, 10):
    MatchAsNode = ast.MatchAs
else:
    MatchAsNode = None


class ScopeNode:
    def __init__(self, name: str) -> None:
        self.name = name
        self.cyclomatic_complexity = 1
        self.child_regions: List["ScopeNode"] = []

    def to_json(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "cyclomatic_complexity": self.cyclomatic_complexity,
            "child_regions": [child.to_json() for child in self.child_regions],
        }


class ComplexityBuilder:
    def build(self, tree: ast.AST) -> ScopeNode:
        root = ScopeNode("<module>")
        self.visit_statements(getattr(tree, "body", []), root)
        return root

    def add(self, scope: ScopeNode, amount: int = 1) -> None:
        scope.cyclomatic_complexity += amount

    def visit_statements(self, statements: Sequence[ast.stmt], scope: ScopeNode) -> None:
        for stmt in statements:
            self.visit_stmt(stmt, scope)

    def visit_stmt(self, stmt: ast.stmt, scope: ScopeNode) -> None:
        method = getattr(self, "visit_" + type(stmt).__name__, None)
        if method is not None:
            method(stmt, scope)
            return
        self.visit_generic(stmt, scope)

    def visit_expr(self, expr: ast.AST, scope: ScopeNode) -> None:
        method = getattr(self, "visit_" + type(expr).__name__, None)
        if method is not None:
            method(expr, scope)
            return
        self.visit_generic(expr, scope)

    def visit_generic(self, node: ast.AST, scope: ScopeNode) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.stmt):
                self.visit_stmt(child, scope)
            else:
                self.visit_expr(child, scope)

    def visit_FunctionDef(self, node: ast.FunctionDef, scope: ScopeNode) -> None:
        self.visit_function_like(node, scope)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef, scope: ScopeNode) -> None:
        self.visit_function_like(node, scope)

    def visit_function_like(self, node: ast.AST, scope: ScopeNode) -> None:
        for decorator in node.decorator_list:
            self.visit_expr(decorator, scope)
        self.visit_arguments(node.args, scope)
        if getattr(node, "returns", None) is not None:
            self.visit_expr(node.returns, scope)
        child = ScopeNode(node.name)
        scope.child_regions.append(child)
        self.visit_statements(node.body, child)

    def visit_ClassDef(self, node: ast.ClassDef, scope: ScopeNode) -> None:
        for decorator in node.decorator_list:
            self.visit_expr(decorator, scope)
        for base in node.bases:
            self.visit_expr(base, scope)
        for keyword in node.keywords:
            self.visit_expr(keyword.value, scope)
        child = ScopeNode(node.name)
        scope.child_regions.append(child)
        self.visit_statements(node.body, child)

    def visit_Lambda(self, node: ast.Lambda, scope: ScopeNode) -> None:
        child = ScopeNode("<lambda>")
        scope.child_regions.append(child)
        self.visit_arguments(node.args, child)
        self.visit_expr(node.body, child)

    def visit_ListComp(self, node: ast.ListComp, scope: ScopeNode) -> None:
        self.visit_comprehension_scope("<listcomp>", node.generators, node.elt, scope)

    def visit_SetComp(self, node: ast.SetComp, scope: ScopeNode) -> None:
        self.visit_comprehension_scope("<setcomp>", node.generators, node.elt, scope)

    def visit_GeneratorExp(self, node: ast.GeneratorExp, scope: ScopeNode) -> None:
        self.visit_comprehension_scope("<genexpr>", node.generators, node.elt, scope)

    def visit_DictComp(self, node: ast.DictComp, scope: ScopeNode) -> None:
        child = ScopeNode("<dictcomp>")
        scope.child_regions.append(child)
        for generator in node.generators:
            self.add(child)
            self.visit_expr(generator.iter, child)
            self.visit_expr(generator.target, child)
            for condition in generator.ifs:
                self.add(child)
                self.visit_expr(condition, child)
        self.visit_expr(node.key, child)
        self.visit_expr(node.value, child)

    def visit_comprehension_scope(self, name: str, generators: Sequence[ast.comprehension], elt: ast.AST, scope: ScopeNode) -> None:
        child = ScopeNode(name)
        scope.child_regions.append(child)
        for generator in generators:
            self.add(child)
            self.visit_expr(generator.iter, child)
            self.visit_expr(generator.target, child)
            for condition in generator.ifs:
                self.add(child)
                self.visit_expr(condition, child)
        self.visit_expr(elt, child)

    def visit_arguments(self, node: ast.arguments, scope: ScopeNode) -> None:
        positional = list(node.posonlyargs) + list(node.args)
        for arg in positional + list(node.kwonlyargs):
            if arg.annotation is not None:
                self.visit_expr(arg.annotation, scope)
        if node.vararg is not None and node.vararg.annotation is not None:
            self.visit_expr(node.vararg.annotation, scope)
        if node.kwarg is not None and node.kwarg.annotation is not None:
            self.visit_expr(node.kwarg.annotation, scope)
        for default in node.defaults:
            self.visit_expr(default, scope)
        for default in node.kw_defaults:
            if default is not None:
                self.visit_expr(default, scope)

    def visit_If(self, node: ast.If, scope: ScopeNode) -> None:
        self.add(scope)
        self.visit_expr(node.test, scope)
        self.visit_statements(node.body, scope)
        self.visit_statements(node.orelse, scope)

    def visit_IfExp(self, node: ast.IfExp, scope: ScopeNode) -> None:
        self.add(scope)
        self.visit_expr(node.test, scope)
        self.visit_expr(node.body, scope)
        self.visit_expr(node.orelse, scope)

    def visit_For(self, node: ast.For, scope: ScopeNode) -> None:
        self.visit_loop(node, scope)

    def visit_AsyncFor(self, node: ast.AsyncFor, scope: ScopeNode) -> None:
        self.visit_loop(node, scope)

    def visit_While(self, node: ast.While, scope: ScopeNode) -> None:
        self.add(scope)
        self.visit_expr(node.test, scope)
        self.visit_statements(node.body, scope)
        self.visit_statements(node.orelse, scope)

    def visit_loop(self, node: ast.AST, scope: ScopeNode) -> None:
        self.add(scope)
        self.visit_expr(node.target, scope)
        self.visit_expr(node.iter, scope)
        self.visit_statements(node.body, scope)
        self.visit_statements(node.orelse, scope)

    def visit_Try(self, node: ast.Try, scope: ScopeNode) -> None:
        self.visit_try_like(node, scope)

    def visit_TryStar(self, node: "ast.TryStar", scope: ScopeNode) -> None:
        self.visit_try_like(node, scope)

    def visit_try_like(self, node: ast.AST, scope: ScopeNode) -> None:
        self.visit_statements(node.body, scope)
        for handler in node.handlers:
            self.add(scope)
            if handler.type is not None:
                self.visit_expr(handler.type, scope)
            self.visit_statements(handler.body, scope)
        self.visit_statements(node.orelse, scope)
        self.visit_statements(node.finalbody, scope)

    def visit_BoolOp(self, node: ast.BoolOp, scope: ScopeNode) -> None:
        self.add(scope, max(0, len(node.values) - 1))
        for value in node.values:
            self.visit_expr(value, scope)

    def visit_Match(self, node: "ast.Match", scope: ScopeNode) -> None:
        self.visit_expr(node.subject, scope)
        for case in node.cases:
            if not (
                isinstance(case.pattern, MatchAsNode)
                and case.pattern.pattern is None
                and case.pattern.name is None
            ):
                self.add(scope)
            self.visit_match_case(case, scope)

    def visit_match_case(self, node: "ast.match_case", scope: ScopeNode) -> None:
        self.visit_pattern(node.pattern, scope)
        if node.guard is not None:
            self.add(scope)
            self.visit_expr(node.guard, scope)
        self.visit_statements(node.body, scope)

    def visit_pattern(self, node: ast.AST, scope: ScopeNode) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.AST):
                self.visit_expr(child, scope)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("python_file")
    args = parser.parse_args(argv)

    with open(args.python_file, "r") as f:
        source = f.read()

    tree = ast.parse(source, filename=args.python_file, mode="exec")
    result = ComplexityBuilder().build(tree)
    json.dump(result.to_json(), sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
