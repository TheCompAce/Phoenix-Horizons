import ast
import json
import math
import os
from ctags.tag import Tag
from radon.raw import analyze
from radon.metrics import h_visit

class HalsteadMetrics:
    def __init__(self):
        self.reset()

    def reset(self):
        self.operators = set()
        self.operands = set()
        self.total_operators = 0
        self.total_operands = 0

    def visit_operator(self, operator):
        self.operators.add(operator)
        self.total_operators += 1

    def visit_operand(self, operand):
        self.operands.add(operand)
        self.total_operands += 1

    def calculate_metrics(self):
        N = self.total_operators + self.total_operands
        n = len(self.operators) + len(self.operands)
        V = N * math.log2(n) if n != 0 else 0
        D = (len(self.operators) / 2) * (self.total_operands / len(self.operands)) if len(self.operands) != 0 else 0
        E = D * V
        return {
            'length': N,
            'vocabulary': n,
            'volume': V,
            'difficulty': D,
            'effort': E
        }


class CyclomaticComplexity:
    def __init__(self):
        self.reset()

    def reset(self):
        self.complexity = 1  # Cyclomatic complexity starts at 1

    def visit_decision_point(self):
        self.complexity += 1

    def calculate_complexity(self):
        return self.complexity

class PythonTagGenerator(ast.NodeVisitor):
    BUILT_IN_FUNCTIONS = set(dir(__builtins__))
    STANDARD_LIBRARIES = set(['os', 'sys', 'math', 'datetime', 'json', 're', 'random', 'collections', 'itertools', 'functools', 'pickle'])
    DATA_STRUCTURES = set(['list', 'dict', 'set', 'tuple', 'str', 'int', 'float', 'bool'])

    def __init__(self):
        if os.path.exists('ctags\metrics.json'):
            os.remove('ctags\metrics.json')

        self.file_path = None
        self.tags = []
        self.metrics = []
        self.halstead = HalsteadMetrics()
        self.cyclomatic = CyclomaticComplexity()  # Create an instance of CyclomaticComplexity
        self.loc = 0
        self.comments = 0
        self.function_count = 0
        self.class_count = 0
        self.total_function_length = 0
        self.code_lines = 0
        self.comment_lines = 0

    def visit_FunctionDef(self, node):
        self.function_count += 1
        self.total_function_length += len(node.body)
        self.add_tag(Tag(node.name, self.file_path, node.lineno, "Python"))
        self.halstead.reset()
        self.cyclomatic.reset()  # Reset the cyclomatic complexity for the new function
        self.generic_visit(node)
        self.metrics.append({
        'name': node.name,
        'halstead_metrics': self.halstead.calculate_metrics(),
        'cyclomatic_complexity': self.cyclomatic.calculate_complexity(),  # Use the CyclomaticComplexity instance to calculate the complexity
        'function_count': self.function_count,
        'class_count': self.class_count,
        'lines_of_code': self.loc,
        'number_of_comments': self.comments,
        'average_function_length': self.total_function_length / self.function_count if self.function_count != 0 else 0,
        'comment_to_code_ratio': self.comments / self.loc if self.loc != 0 else 0,
        'code_lines': self.code_lines,
        'comment_lines': self.comment_lines,
    })

        self.halstead.reset()
        self.cyclomatic.reset()

    def visit(self, node):
        if isinstance(node, (ast.BinOp, ast.UnaryOp, ast.Compare, ast.BoolOp, ast.Assign, ast.AugAssign)):
            self.halstead.visit_operator(node)
        elif isinstance(node, (ast.Name, ast.Constant)):
            self.halstead.visit_operand(node)
        self.loc += 1
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Str):
            self.comments += 1
            self.comment_lines += 1
        else:
            self.code_lines += 1
        super().visit(node)

    def handle_decorators(self, node):
        for decorator in node.decorator_list:
            self.add_tag(Tag(f"@{self.get_function_name(decorator)}", self.file_path, decorator.lineno, "Python"))

    def handle_docstring(self, node):
        if ast.get_docstring(node):
            self.add_tag(Tag(f"{node.name} (doc)", self.file_path, node.lineno, "Python"))

    def handle_magic_method(self, node):
        if node.name.startswith('__') and node.name.endswith('__'):
            self.add_tag(Tag(f"{node.name} magic method in {self.file_path}", self.file_path, node.lineno, "Python"))

    def handle_complex_assignments(self, node):
        if isinstance(node, ast.Name):
            self.add_tag(Tag(node.id, self.file_path, node.lineno, "Python"))
        elif isinstance(node, (ast.Tuple, ast.List)):
            for el in node.elts:
                self.handle_complex_assignments(el)
        elif isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values):
                self.handle_complex_assignments(key)
                self.handle_complex_assignments(value)

    def handle_imports(self, node, alias):
        if alias.name in self.STANDARD_LIBRARIES:
            self.add_tag(Tag(f"standard library {alias.name}", self.file_path, node.lineno, "Python"))
        else:
            self.add_tag(Tag(alias.name, self.file_path, node.lineno, "Python"))

    def handle_function_call(self, node, function_name):
        if function_name in self.BUILT_IN_FUNCTIONS:
            self.add_tag(Tag(f"builtin {function_name}", self.file_path, node.lineno, "Python"))
        elif function_name in self.DATA_STRUCTURES:
            self.add_tag(Tag(f"data structure {function_name}", self.file_path, node.lineno, "Python"))
        else:
            self.add_tag(Tag(f"call {function_name}", self.file_path, node.lineno, "Python"))


    def visit_AsyncFunctionDef(self, node):
        self.add_tag(Tag(node.name, self.file_path, node.lineno, "Python"))
        self.generic_visit(node)

    def visit_ClassDef(self, node):
        self.class_count += 1
        self.add_tag(Tag(node.name, self.file_path, node.lineno, "Python"))
        self.handle_decorators(node)
        self.handle_docstring(node)
        self.handle_magic_method(node)
        self.generic_visit(node)

    def visit_Assign(self, node):
        # Add tags for variable type if type hinting is used
        if isinstance(node, ast.AnnAssign) and isinstance(node.annotation, ast.Name):
            self.add_tag(Tag(f"{node.target.id}: {node.annotation.id}", self.file_path, node.lineno, "Python"))
        else:
            for target in node.targets:
                self.handle_complex_assignments(target)
        self.generic_visit(node)

    def visit_Call(self, node):
        function_name = self.get_function_name(node.func)
        self.handle_function_call(node, function_name)
        self.generic_visit(node)

    def visit_Import(self, node):
        for alias in node.names:
            self.handle_imports(node, alias)
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        for alias in node.names:
            self.handle_imports(node, alias)
        self.generic_visit(node)

    def visit_For(self, node):
        self.cyclomatic.visit_decision_point()
        self.add_tag(Tag(f"for loop at line {node.lineno}", self.file_path, node.lineno, "Python"))
        self.generic_visit(node)

    def visit_While(self, node):
        self.cyclomatic.visit_decision_point()
        self.add_tag(Tag(f"while loop at line {node.lineno}", self.file_path, node.lineno, "Python"))
        self.generic_visit(node)

    def visit_If(self, node):
        self.cyclomatic.visit_decision_point()
        self.add_tag(Tag(f"if statement at line {node.lineno}", self.file_path, node.lineno, "Python"))
        self.generic_visit(node)

    def visit_Try(self, node):
        self.add_tag(Tag(f"try block at line {node.lineno}", self.file_path, node.lineno, "Python"))
        self.generic_visit(node)

    def visit_ExceptHandler(self, node):
        self.add_tag(Tag(f"except clause at line {node.lineno}", self.file_path, node.lineno, "Python"))
        self.generic_visit(node)

    def visit_With(self, node):
        self.add_tag(Tag(f"with statement at line {node.lineno}", self.file_path, node.lineno, "Python"))
        self.generic_visit(node)

    def visit_Lambda(self, node):
        self.add_tag(Tag(f"lambda function at line {node.lineno}", self.file_path, node.lineno, "Python"))
        self.generic_visit(node)

    def visit_ListComp(self, node):
        self.add_tag(Tag(f"list comprehension at line {node.lineno}", self.file_path, node.lineno, "Python"))
        self.generic_visit(node)

    def visit_SetComp(self, node):
        self.add_tag(Tag(f"set comprehension at line {node.lineno}", self.file_path, node.lineno, "Python"))
        self.generic_visit(node)

    def visit_DictComp(self, node):
        self.add_tag(Tag(f"dict comprehension at line {node.lineno}", self.file_path, node.lineno, "Python"))
        self.generic_visit(node)

    def visit_GeneratorExp(self, node):
        self.add_tag(Tag(f"generator expression at line {node.lineno}", self.file_path, node.lineno, "Python"))
        self.generic_visit(node)

    def visit_Yield(self, node):
        self.add_tag(Tag(f"yield statement at line {node.lineno}", self.file_path, node.lineno, "Python"))
        self.generic_visit(node)

    def visit_YieldFrom(self, node):
        self.add_tag(Tag(f"yield from statement at line {node.lineno}", self.file_path, node.lineno, "Python"))
        self.generic_visit(node)

    def visit_Await(self, node):
        self.add_tag(Tag(f"await expression at line {node.lineno}", self.file_path, node.lineno, "Python"))
        self.generic_visit(node)

    def visit_AsyncFor(self, node):
        self.add_tag(Tag(f"async for loop at line {node.lineno}", self.file_path, node.lineno, "Python"))
        self.generic_visit(node)

    def visit_AsyncWith(self, node):
        self.add_tag(Tag(f"async with statement at line {node.lineno}", self.file_path, node.lineno, "Python"))
        self.generic_visit(node)

    def visit_Assert(self, node):
        self.add_tag(Tag(f"assert at line {node.lineno}", self.file_path, node.lineno, "Python"))
        self.generic_visit(node)

    def visit_Global(self, node):
        for name in node.names:
            self.add_tag(Tag(f"global {name}", self.file_path, node.lineno, "Python"))
        self.generic_visit(node)

    def visit_Nonlocal(self, node):
        for name in node.names:
            self.add_tag(Tag(f"nonlocal {name}", self.file_path, node.lineno, "Python"))
        self.generic_visit(node)

    def visit_arguments(self, node):
        for arg in node.args:
            self.add_tag(Tag(f"argument {arg.arg}", self.file_path, arg.lineno, "Python"))
        self.generic_visit(node)

    def visit_Return(self, node):
        self.add_tag(Tag(f"return at line {node.lineno}", self.file_path, node.lineno, "Python"))
        self.generic_visit(node)

    def visit_Expr(self, node):
        self.add_tag(Tag(f"expression at line {node.lineno}", self.file_path, node.lineno, "Python"))
        self.generic_visit(node)

    def visit_Starred(self, node):
        self.add_tag(Tag(f"starred expression at line {node.lineno}", self.file_path, node.lineno, "Python"))
        self.generic_visit(node)

    def visit_Slice(self, node):
        self.add_tag(Tag(f"slice at line {node.lineno}", self.file_path, node.lineno, "Python"))
        self.generic_visit(node)

    def visit_comprehension(self, node):
        if hasattr(node, 'lineno'):
            self.add_tag(Tag(f"comprehension at line {node.lineno}", self.file_path, node.lineno, "Python"))
        else:
            self.add_tag(Tag("comprehension", self.file_path, 0, "Python"))
        self.generic_visit(node)


    def visit_Try(self, node):
        self.add_tag(Tag("try", self.file_path, node.lineno, "Python"))
        for handler in node.handlers:
            self.add_tag(Tag("except", self.file_path, handler.lineno, "Python"))
        if node.orelse:
            self.add_tag(Tag("else", self.file_path, node.orelse[0].lineno, "Python"))
        if node.finalbody:
            self.add_tag(Tag("finally", self.file_path, node.finalbody[0].lineno, "Python"))
        self.generic_visit(node)

    def visit_Raise(self, node):
        self.add_tag(Tag("raise", self.file_path, node.lineno, "Python"))
        self.generic_visit(node)

    def visit_Yield(self, node):
        self.add_tag(Tag("yield", self.file_path, node.lineno, "Python"))
        self.generic_visit(node)

    def visit_Lambda(self, node):
        self.add_tag(Tag("lambda", self.file_path, node.lineno, "Python"))
        self.generic_visit(node)

    def visit_Attribute(self, node):
        self.add_tag(Tag(f"{self.get_function_name(node.value)}.{node.attr}", self.file_path, node.lineno, "Python"))
        self.generic_visit(node)

    def visit_Dict(self, node):
        self.add_tag(Tag("dict", self.file_path, node.lineno, "Python"))
        self.generic_visit(node)

    def visit_List(self, node):
        self.add_tag(Tag("list", self.file_path, node.lineno, "Python"))
        self.generic_visit(node)

    def visit_Set(self, node):
        self.add_tag(Tag("set", self.file_path, node.lineno, "Python"))
        self.generic_visit(node)

    def visit_Tuple(self, node):
        self.add_tag(Tag("tuple", self.file_path, node.lineno, "Python"))
        self.generic_visit(node)

    def visit_Str(self, node):
        self.add_tag(Tag("str", self.file_path, node.lineno, "Python"))
        self.generic_visit(node)

    def visit_Num(self, node):
        self.add_tag(Tag("num", self.file_path, node.lineno, "Python"))
        self.generic_visit(node)

    def visit_NameConstant(self, node):
        self.add_tag(Tag(str(node.value), self.file_path, node.lineno, "Python"))
        self.generic_visit(node)
        

    def get_type_hint(self, node):
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Subscript):
            return f"{node.value.id}[{self.get_type_hint(node.slice.value)}]"
        elif isinstance(node, ast.Tuple):
            return f"({', '.join(self.get_type_hint(e) for e in node.elts)})"
        else:
            return "<unknown>"

    def get_function_name(self, node):
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self.get_function_name(node.value)}.{node.attr}"
        else:
            return "<unknown>"

    def generate_tags(self, source_code):
        try:
            tree = ast.parse(source_code)
            self.visit(tree)
        except Exception as e:
            self.add_tag(Tag(f"error: {str(e)}", self.file_path, 0, "Python"))

    def generate_metrics(self, source_code):
        try:
            tree = ast.parse(source_code)
            self.visit(tree)
        except Exception as e:
            self.metrics.append({'error': str(e)})

    def add_tag(self, tag):
        self.tags.append(tag.to_dict())

    def process_source_code(self, file_path, source_code):
        self.file_path = file_path
        self.generate_tags(source_code)
        self.generate_metrics(source_code)

        raw_metrics = analyze(source_code)
        halstead_metrics = h_visit(source_code)

        self.code_lines += raw_metrics.loc
        self.comment_lines += raw_metrics.comments

    def write_metrics_to_file(self, metrics):
        with open('ctags\metrics.json', 'w') as f:
            json.dump(metrics, f)