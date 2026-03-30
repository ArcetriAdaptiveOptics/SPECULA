import ast
import pkgutil
import textwrap
from pathlib import Path


def expr_to_string(node):
    """Best-effort conversion of AST expression to readable string."""
    if isinstance(node, ast.Constant):
        if isinstance(node.value, str):
            return node.value
        if node.value is None:
            return 'None'
        return str(node.value)

    if isinstance(node, getattr(ast, 'Str', type(None))):
        return node.s

    if isinstance(node, ast.Name):
        return '{' + node.id + '}'

    if isinstance(node, ast.Attribute):
        base = expr_to_string(node.value)
        return '{' + f"{base}.{node.attr}" + '}'

    if isinstance(node, ast.JoinedStr):
        parts = []
        for value in node.values:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                parts.append(value.value)
            elif isinstance(value, ast.FormattedValue):
                parts.append('{'+ expr_to_string(value.value) + '}')
        return ''.join(parts)

    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = expr_to_string(node.left)
        right = expr_to_string(node.right)
        return f"{left}{right}"

    if hasattr(ast, 'unparse'):
        return '{' + ast.unparse(node) + '}'

    return '{expr}'


def get_slice_value(slice_node):
    """Helper to safely extract string values from AST slices across Python versions."""
    if isinstance(slice_node, ast.Constant):
        return slice_node.value
    elif isinstance(slice_node, getattr(ast, 'Str', type(None))):
        return slice_node.s
    elif isinstance(slice_node, getattr(ast, 'Index', type(None))):
        if isinstance(slice_node.value, ast.Constant):
            return slice_node.value.value
        elif isinstance(slice_node.value, getattr(ast, 'Str', type(None))):
            return getattr(slice_node.value, 's', None)
    return None


def get_subscript_key(slice_node):
    """Return a readable key for subscript slices, including dynamic expressions."""
    value = get_slice_value(slice_node)
    if value is not None:
        return value
    return expr_to_string(slice_node)


def format_port_name(name):
    """Render dynamic placeholders in a user-friendly style for docs."""
    return str(name).replace('{', '[').replace('}', ']')


def parse_optional_flag(value_node):
    """Extract optional=True/False from InputValue/InputList constructor calls."""
    if not isinstance(value_node, ast.Call):
        return False

    for kw in value_node.keywords:
        if kw.arg == 'optional':
            if isinstance(kw.value, ast.Constant):
                return bool(kw.value.value)
            return False

    return False


def record_add(mutations, attr_name, key_name, optional=False):
    """Record an add/update operation for inputs or outputs."""
    if attr_name == 'inputs':
        mutations.append(('set_input', key_name, optional))
    elif attr_name == 'outputs':
        mutations.append(('set_output', key_name, False))


def record_delete(mutations, attr_name, key_name):
    """Record a delete operation for inputs or outputs."""
    if attr_name == 'inputs':
        mutations.append(('del_input', key_name, False))
    elif attr_name == 'outputs':
        mutations.append(('del_output', key_name, False))


def collect_io_mutations(stmts, mutations):
    """Collect I/O mutations in source order from a list of statements."""
    for stmt in stmts:
        if isinstance(stmt, ast.Assign):
            optional = parse_optional_flag(stmt.value)
            for target in stmt.targets:
                if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Attribute):
                    if isinstance(target.value.value, ast.Name) and target.value.value.id == 'self':
                        attr_name = target.value.attr
                        key_name = get_subscript_key(target.slice)
                        if key_name:
                            record_add(mutations, attr_name, key_name, optional)

        elif isinstance(stmt, ast.AnnAssign):
            target = stmt.target
            optional = parse_optional_flag(stmt.value)
            if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Attribute):
                if isinstance(target.value.value, ast.Name) and target.value.value.id == 'self':
                    attr_name = target.value.attr
                    key_name = get_subscript_key(target.slice)
                    if key_name:
                        record_add(mutations, attr_name, key_name, optional)

        elif isinstance(stmt, ast.Delete):
            for target in stmt.targets:
                if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Attribute):
                    if isinstance(target.value.value, ast.Name) and target.value.value.id == 'self':
                        attr_name = target.value.attr
                        key_name = get_subscript_key(target.slice)
                        if key_name:
                            record_delete(mutations, attr_name, key_name)

        elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            call = stmt.value
            if isinstance(call.func, ast.Attribute) and isinstance(call.func.value, ast.Attribute):
                parent = call.func.value
                if isinstance(parent.value, ast.Name) and parent.value.id == 'self':
                    attr_name = parent.attr
                    method = call.func.attr

                    if method == 'pop' and call.args:
                        key_name = expr_to_string(call.args[0])
                        if key_name:
                            record_delete(mutations, attr_name, key_name)

                    elif method == 'setdefault' and call.args:
                        key_name = expr_to_string(call.args[0])
                        optional = parse_optional_flag(call.args[1]) if len(call.args) > 1 else False
                        if key_name:
                            record_add(mutations, attr_name, key_name, optional)

                    elif method == 'update' and call.args:
                        first = call.args[0]
                        if isinstance(first, ast.Dict):
                            for key_node, value_node in zip(first.keys, first.values):
                                if key_node is None:
                                    continue
                                key_name = expr_to_string(key_node)
                                optional = parse_optional_flag(value_node)
                                if key_name:
                                    record_add(mutations, attr_name, key_name, optional)

        if isinstance(stmt, ast.If):
            collect_io_mutations(stmt.body, mutations)
            collect_io_mutations(stmt.orelse, mutations)
        elif isinstance(stmt, (ast.For, ast.AsyncFor, ast.While)):
            collect_io_mutations(stmt.body, mutations)
            collect_io_mutations(stmt.orelse, mutations)
        elif isinstance(stmt, (ast.With, ast.AsyncWith)):
            collect_io_mutations(stmt.body, mutations)
        elif isinstance(stmt, ast.Try):
            collect_io_mutations(stmt.body, mutations)
            for handler in stmt.handlers:
                collect_io_mutations(handler.body, mutations)
            collect_io_mutations(stmt.orelse, mutations)
            collect_io_mutations(stmt.finalbody, mutations)


def extract_classes_from_file(filepath):
    """Return a dict with class info (doc, bases, inputs, outputs) for a file."""
    results = {}
    try:
        source = filepath.read_text(encoding='utf-8')
        tree = ast.parse(source)
    except (SyntaxError, UnicodeDecodeError, FileNotFoundError):
        return results

    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue

        # 1. Extract Docstring
        docstring = ast.get_docstring(node)
        if not docstring:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == '__init__':
                    docstring = ast.get_docstring(item)
                    break
        short = ''
        if docstring:
            lines = docstring.splitlines()
            short_lines = []
            for line in lines:
                if line.strip() == '':
                    break
                short_lines.append(line.strip())
            short = ' '.join(short_lines)
            
        # 2. Extract Base Classes
        bases = []
        for b in node.bases:
            if isinstance(b, ast.Name):
                bases.append(b.id)
            elif isinstance(b, ast.Attribute):
                bases.append(b.attr)
                
        # 3. Extract Inputs & Outputs
        mutations = []
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                collect_io_mutations(item.body, mutations)

        results[node.name] = {
            'doc': short,
            'bases': bases,
            'mutations': mutations,
            'module': str(filepath),
        }
    return results


def build_global_registry(specula_path):
    """Scan the entire specula codebase to build a global class registry for inheritance."""
    registry = {}
    for filepath in specula_path.rglob('*.py'):
        classes = extract_classes_from_file(filepath)
        registry.update(classes)
    return registry


def get_inherited_io(classname, registry, resolved=None):
    """Resolve final inputs/outputs by applying inherited and local mutations."""
    if resolved is None:
        resolved = set()
    if classname in resolved or classname not in registry:
        return {}, []
        
    resolved.add(classname)
    info = registry[classname]

    all_inputs = {}
    all_outputs = []

    for base in info['bases']:
        base_in, base_out = get_inherited_io(base, registry, resolved)
        for k, v in base_in.items():
            if k not in all_inputs:
                all_inputs[k] = v
        for o in base_out:
            if o not in all_outputs:
                all_outputs.append(o)

    for op, key_name, opt in info['mutations']:
        if op == 'set_input':
            all_inputs[key_name] = bool(opt)
        elif op == 'set_output':
            if key_name not in all_outputs:
                all_outputs.append(key_name)
        elif op == 'del_input':
            all_inputs.pop(key_name, None)
        elif op == 'del_output':
            if key_name in all_outputs:
                all_outputs.remove(key_name)
  
    return all_inputs, all_outputs


def scan_package(package_path, package_name):
    """Scan a package directory and return list of (module_name, filepath)."""
    modules = []
    if not Path(package_path).exists():
        return modules
    for _, modname, ispkg in pkgutil.iter_modules([str(package_path)]):
        if not ispkg:
            modules.append((
                f"{package_name}.{modname}",
                Path(package_path) / f"{modname}.py"
            ))
    return sorted(modules)


def generate_rst_table(category_name, modules, registry, description='', include_io=False):
    """Generate RST content with a table listing class names, descriptions, and I/O."""
    valid_classes = []
    for module_name, filepath in modules:
        classes_in_file = extract_classes_from_file(filepath)
        for classname, info in classes_in_file.items():
            if not classname.startswith('_'):
                valid_classes.append((module_name, classname, info))

    title = f"{category_name} Summary"
    lines = [
        title,
        '=' * len(title),
        '',
        description,
        f'Total: **{len(valid_classes)}** classes.',
        '',
        '.. list-table::',
        '   :header-rows: 1',
    ]

    has_io = include_io

    if has_io:
        lines.extend([
            '   :widths: 20 40 20 20',
            '',
            '   * - Class',
            '     - Description',
            '     - Inputs',
            '     - Outputs',
        ])
    else:
        lines.extend([
            '   :widths: 30 70',
            '',
            '   * - Class',
            '     - Description',
        ])

    for module_name, classname, info in valid_classes:
        full_name = f"{module_name}.{classname}"
        lines.append(f'   * - :class:`~{full_name}`')

        # Description Column
        desc = info['doc'] if info['doc'] else '*No description available.*'
        wrapped_lines = textwrap.wrap(desc, width=50)
        cell_content = '\n       | '.join(wrapped_lines)
        if len(wrapped_lines) > 1:
            cell_content = '| ' + cell_content
        lines.append(f'     - {cell_content}')

        # I/O Columns (Only for Processing Objects)
        if has_io:
            inputs, outputs = get_inherited_io(classname, registry)

            in_list = [
                f"{format_port_name(k)} *(opt)*" if opt else format_port_name(k)
                for k, opt in inputs.items()
            ]
            out_list = [format_port_name(o) for o in outputs]
            in_str = ', '.join(in_list) if in_list else '-'
            out_str = ', '.join(out_list) if out_list else '-'

            # Textwrap handles long comma-separated lists gracefully in RST
            in_lines = textwrap.wrap(in_str, width=30)
            in_wrapped = '\n       | '.join(in_lines)
            if len(in_lines) > 1:
                in_wrapped = '| ' + in_wrapped

            out_lines = textwrap.wrap(out_str, width=30)
            out_wrapped = '\n       | '.join(out_lines)
            if len(out_lines) > 1:
                out_wrapped = '| ' + out_wrapped

            lines.append(f'     - {in_wrapped}')
            lines.append(f'     - {out_wrapped}')

    lines.append('')
    return '\n'.join(lines)


def main():
    specula_path = Path(__file__).parent.parent.parent / 'specula'
    api_docs_path = Path(__file__).parent.parent / 'api'
    api_docs_path.mkdir(exist_ok=True)

    print("Building global class registry for inheritance resolution...")
    registry = build_global_registry(specula_path)

    categories = [
        {
            'name': 'Processing Objects',
            'path': specula_path / 'processing_objects',
            'package': 'specula.processing_objects',
            'description': 'Processing objects for simulating AO system components.',
            'filename': 'processing_objects_summary',
            'include_io': True,
        },
        {
            'name': 'Data Objects',
            'path': specula_path / 'data_objects',
            'package': 'specula.data_objects',
            'description': 'Data objects for representing simulation data.',
            'filename': 'data_objects_summary',
            'include_io': False,
        },
    ]

    for cat in categories:
        print(f"Scanning {cat['path']}...")
        modules = scan_package(cat['path'], cat['package'])
        if not modules:
            print("  No modules found.")
            continue

        content = generate_rst_table(
            cat['name'],
            modules,
            registry,
            cat['description'],
            include_io=cat.get('include_io', False),
        )
        out_file = api_docs_path / f"{cat['filename']}.rst"
        out_file.write_text(content, encoding='utf-8')
        print(f"  -> Generated {out_file}")

    print('\nDone.')


if __name__ == '__main__':
    main()
