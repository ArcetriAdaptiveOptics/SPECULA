import ast
import pkgutil
import textwrap
from pathlib import Path


def get_class_short_doc(filepath):
    """Return a dict {classname: short_docstring} for all classes in a file."""
    results = {}
    try:
        source = filepath.read_text(encoding='utf-8')
        tree = ast.parse(source)
    except (SyntaxError, UnicodeDecodeError):
        return results

    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        docstring = ast.get_docstring(node)
        if not docstring:
            # fallback: try __init__ docstring
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
        results[node.name] = short
    return results


def scan_package(package_path, package_name):
    """Scan a package directory and return list of (module_name, filepath)."""
    modules = []
    if not Path(package_path).exists():
        return modules
    for importer, modname, ispkg in pkgutil.iter_modules([str(package_path)]):
        if not ispkg:
            modules.append((
                f"{package_name}.{modname}",
                Path(package_path) / f"{modname}.py"
            ))
    return sorted(modules)


def generate_rst_table(category_name, modules, description=''):
    """Generate RST content with a table listing class names and short descriptions."""
    valid_classes = {}
    for module_name, filepath in modules:
        for classname, short_doc in get_class_short_doc(filepath).items():
            if not classname.startswith('_'):
                valid_classes[f"{module_name}.{classname}"] = short_doc

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
        '   :widths: 30 70',
        '',
        '   * - Class',
        '     - Description',
    ]

    for full_name, short_doc in valid_classes.items():
        lines.append(f'   * - :class:`~{full_name}`')
        desc = short_doc if short_doc else '*No description available.*'
        wrapped_lines = textwrap.wrap(desc, width=60)
        if len(wrapped_lines) > 1:
            cell_content = '\n       | '.join(wrapped_lines)
            cell_content = '| ' + cell_content
        else:
            cell_content = desc
        lines.append(f'     - {cell_content}')

    lines.append('')
    return '\n'.join(lines)


def main():
    specula_path = Path(__file__).parent.parent.parent / 'specula'
    api_docs_path = Path(__file__).parent.parent / 'api'
    api_docs_path.mkdir(exist_ok=True)

    categories = [
        {
            'name': 'Processing Objects',
            'path': specula_path / 'processing_objects',
            'package': 'specula.processing_objects',
            'description': 'Processing objects for simulating AO system components.',
            'filename': 'processing_objects_summary',
        },
        {
            'name': 'Data Objects',
            'path': specula_path / 'data_objects',
            'package': 'specula.data_objects',
            'description': 'Data objects for representing simulation data.',
            'filename': 'data_objects_summary',
        },
    ]

    for cat in categories:
        print(f"Scanning {cat['path']}...")
        modules = scan_package(cat['path'], cat['package'])
        if not modules:
            print(f"  No modules found.")
            continue

        content = generate_rst_table(cat['name'], modules, cat['description'])
        out_file = api_docs_path / f"{cat['filename']}.rst"
        out_file.write_text(content, encoding='utf-8')
        print(f"  -> Generated {out_file}")

    print('\nDone.')


if __name__ == '__main__':
    main()