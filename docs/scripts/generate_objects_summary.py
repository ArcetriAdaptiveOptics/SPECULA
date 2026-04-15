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


def is_super_method_call(node, method_name):
    """Return True when node is super().<method_name>() call."""
    if not isinstance(node, ast.Call):
        return False
    if not isinstance(node.func, ast.Attribute):
        return False
    if node.func.attr != method_name:
        return False
    base = node.func.value
    return isinstance(base, ast.Call) and isinstance(base.func, ast.Name) and base.func.id == 'super'


def parse_optional_from_input_desc(value_node):
    """Extract optional flag from InputDesc(..., '...optional...')."""
    if not isinstance(value_node, ast.Call):
        return False

    if not isinstance(value_node.func, ast.Name) or value_node.func.id != 'InputDesc':
        return False

    desc_node = None
    if len(value_node.args) >= 2:
        desc_node = value_node.args[1]
    else:
        for kw in value_node.keywords:
            if kw.arg == 'desc':
                desc_node = kw.value
                break

    if desc_node is None:
        return False

    desc_str = expr_to_string(desc_node)
    return '(optional)' in str(desc_str).lower()


def parse_port_dict(dict_node, is_input):
    """Parse static port dict AST node and return parsed entries."""
    if not isinstance(dict_node, ast.Dict):
        return None

    if is_input:
        parsed = {}
        for key_node, value_node in zip(dict_node.keys, dict_node.values):
            if key_node is None:
                continue
            key_name = get_subscript_key(key_node)
            if key_name:
                parsed[key_name] = parse_optional_from_input_desc(value_node)
        return parsed

    parsed = []
    for key_node in dict_node.keys:
        if key_node is None:
            continue
        key_name = get_subscript_key(key_node)
        if key_name and key_name not in parsed:
            parsed.append(key_name)
    return parsed


def merge_ports(current_ports, new_ports, is_input):
    """Merge parsed ports preserving order for outputs."""
    if is_input:
        merged = dict(current_ports)
        merged.update(new_ports)
        return merged

    merged = list(current_ports)
    for name in new_ports:
        if name not in merged:
            merged.append(name)
    return merged


def extract_named_ports(class_node, method_name, is_input):
    """Extract static ports from input_names/output_names methods.

    Returns
    -------
    tuple
        (mode, ports) where mode is one of:
        - None: method not found or not statically parseable
        - 'replace': method returns full local dictionary
        - 'extend': method starts from super().<method_name>() and extends it
    """
    method_node = None
    for item in class_node.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == method_name:
            method_node = item
            break

    if method_node is None:
        return None, None

    var_values = {}
    super_seeded_vars = set()

    empty_ports = {} if is_input else []

    for stmt in method_node.body:
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
            target_name = stmt.targets[0].id

            if is_super_method_call(stmt.value, method_name):
                var_values[target_name] = dict(empty_ports) if is_input else list(empty_ports)
                super_seeded_vars.add(target_name)
                continue

            parsed = parse_port_dict(stmt.value, is_input=is_input)
            if parsed is not None:
                var_values[target_name] = parsed
                continue

        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            call = stmt.value
            if isinstance(call.func, ast.Attribute) and call.func.attr == 'update':
                if isinstance(call.func.value, ast.Name) and call.args:
                    target_name = call.func.value.id
                    parsed = parse_port_dict(call.args[0], is_input=is_input)
                    if target_name in var_values and parsed is not None:
                        var_values[target_name] = merge_ports(var_values[target_name], parsed, is_input=is_input)
                        continue

        if isinstance(stmt, ast.Return):
            parsed = parse_port_dict(stmt.value, is_input=is_input)
            if parsed is not None:
                return 'replace', parsed

            if isinstance(stmt.value, ast.Name) and stmt.value.id in var_values:
                mode = 'extend' if stmt.value.id in super_seeded_vars else 'replace'
                return mode, var_values[stmt.value.id]

            if is_super_method_call(stmt.value, method_name):
                return 'extend', dict(empty_ports) if is_input else list(empty_ports)

    return None, None


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
                
        input_names_mode, named_inputs = extract_named_ports(node, 'input_names', is_input=True)
        output_names_mode, named_outputs = extract_named_ports(node, 'output_names', is_input=False)

        results[node.name] = {
            'doc': short,
            'bases': bases,
            'input_names_mode': input_names_mode,
            'output_names_mode': output_names_mode,
            'named_inputs': named_inputs,
            'named_outputs': named_outputs,
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

    # Resolve only from explicit input_names/output_names declarations.
    if info.get('input_names_mode') == 'replace':
        all_inputs = dict(info.get('named_inputs') or {})
    elif info.get('input_names_mode') == 'extend':
        all_inputs.update(info.get('named_inputs') or {})

    if info.get('output_names_mode') == 'replace':
        all_outputs = list(info.get('named_outputs') or [])
    elif info.get('output_names_mode') == 'extend':
        for out_name in info.get('named_outputs') or []:
            if out_name not in all_outputs:
                all_outputs.append(out_name)
  
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
