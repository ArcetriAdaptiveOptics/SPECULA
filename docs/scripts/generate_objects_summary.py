import importlib
import importlib.util
import inspect
import pkgutil
import re
import textwrap
import uuid
import warnings
from pathlib import Path


def format_port_name(name):
    """Render dynamic placeholders in a user-friendly style for docs."""
    return str(name).replace('{', '[').replace('}', ']')


def _first_doc_paragraph(docstring):
    if not docstring:
        return ''

    lines = docstring.splitlines()
    short_lines = []
    for line in lines:
        if line.strip() == '':
            break
        short_lines.append(line.strip())

    return ' '.join(short_lines)


def _normalize_inline_literals(text):
    """Convert single-backtick inline literals to RST double-backtick form."""
    return re.sub(r'(?<!`)`([^`\n]+)`(?!`)', r'``\1``', text)


def _get_short_doc(klass):
    docstring = inspect.getdoc(klass) or inspect.getdoc(getattr(klass, '__init__', None))
    short_doc = _first_doc_paragraph(docstring)
    # Keep inline literals from being interpreted as unresolved roles.
    return _normalize_inline_literals(short_doc)


def _is_optional_input(desc_obj):
    # InputDesc is a namedtuple(type, desc), but this parser also accepts
    # tuple-like or object-like variants used by tests and custom code.
    desc_text = ''
    if hasattr(desc_obj, 'desc'):
        desc_text = getattr(desc_obj, 'desc')
    elif isinstance(desc_obj, tuple) and len(desc_obj) >= 2:
        desc_text = desc_obj[1]

    return '(optional)' in str(desc_text).lower()


def _safe_call_class_port_method(klass, method_name):
    method = getattr(klass, method_name, None)
    if method is None or not callable(method):
        return None

    try:
        return method()
    except (AttributeError, KeyError, NameError, TypeError, ValueError, RuntimeError) as exc:
        warnings.warn(
            f"Skipping {klass.__module__}.{klass.__name__}.{method_name}() due to error: {exc}",
            RuntimeWarning,
        )
        return None


def _normalize_input_ports(raw_inputs):
    if not isinstance(raw_inputs, dict):
        return {}

    normalized = {}
    for name, desc in raw_inputs.items():
        normalized[str(name)] = _is_optional_input(desc)
    return normalized


def _normalize_output_ports(raw_outputs):
    if isinstance(raw_outputs, dict):
        keys = raw_outputs.keys()
    elif isinstance(raw_outputs, (list, tuple, set)):
        keys = raw_outputs
    else:
        return []

    names = []
    for key in keys:
        key_str = str(key)
        if key_str not in names:
            names.append(key_str)
    return names


def _build_class_info(klass, module_repr=''):
    raw_inputs = _safe_call_class_port_method(klass, 'input_names')
    raw_outputs = _safe_call_class_port_method(klass, 'output_names')

    return {
        'class': klass,
        'doc': _get_short_doc(klass),
        'bases': [base.__name__ for base in klass.__bases__ if hasattr(base, '__name__')],
        'named_inputs': _normalize_input_ports(raw_inputs),
        'named_outputs': _normalize_output_ports(raw_outputs),
        'module': module_repr or klass.__module__,
    }


def _iter_module_classes(module):
    classes = []
    for class_name, klass in inspect.getmembers(module, inspect.isclass):
        if class_name.startswith('_'):
            continue
        if klass.__module__ != module.__name__:
            continue
        classes.append((class_name, klass))
    return classes


def _load_module_from_file(filepath, module_name=None):
    if module_name is None:
        module_name = f"_specula_docs_summary_{filepath.stem}_{uuid.uuid4().hex}"

    spec = importlib.util.spec_from_file_location(module_name, filepath)
    if spec is None or spec.loader is None:
        return None

    module = importlib.util.module_from_spec(spec)

    try:
        spec.loader.exec_module(module)
    except (
        AttributeError,
        ImportError,
        NameError,
        RuntimeError,
        SyntaxError,
        TypeError,
        ValueError,
    ) as exc:
        warnings.warn(
            f"Skipping module {filepath} due to import error: {exc}",
            RuntimeWarning,
        )
        return None

    return module


def _load_module(module_name, filepath):
    try:
        return importlib.import_module(module_name), module_name
    except ImportError:
        module = _load_module_from_file(filepath, module_name=module_name)
        return module, (module_name if module is not None else str(filepath))


def extract_classes_from_file(filepath, module_name=None):
    """Return a dict with class info (doc, bases, inputs, outputs) for a file."""
    results = {}
    filepath = Path(filepath)
    if module_name:
        module, module_repr = _load_module(module_name, filepath)
    else:
        module = _load_module_from_file(filepath)
        module_repr = str(filepath)

    if module is None:
        return results

    for class_name, klass in _iter_module_classes(module):
        results[class_name] = _build_class_info(klass, module_repr=module_repr)

    return results


def scan_package(package_path, package_name):
    """Scan a package directory and return list of (module_name, filepath)."""
    modules = []
    package_path = Path(package_path)
    if not package_path.exists():
        return modules

    for _, modname, ispkg in pkgutil.iter_modules([str(package_path)]):
        if not ispkg:
            modules.append((
                f"{package_name}.{modname}",
                package_path / f"{modname}.py",
            ))

    return sorted(modules)


def _iter_module_class_infos(module_name, filepath):
    module, module_repr = _load_module(module_name, filepath)

    if module is None:
        return []

    infos = []
    for class_name, klass in _iter_module_classes(module):
        infos.append((module_name, class_name, _build_class_info(klass, module_repr=module_repr)))
    return infos


def generate_rst_table(category_name, modules, description='', include_io=False):
    """Generate an RST class summary table.

    If ``include_io`` is True, the table includes Inputs/Outputs columns.
    Class selection is unchanged: classes are listed even when they expose
    no named inputs or outputs.
    """
    valid_classes = []
    skipped_modules = []
    for module_name, filepath in modules:
        module_classes = _iter_module_class_infos(module_name, filepath)
        if not module_classes:
            skipped_modules.append(module_name)
        valid_classes.extend(module_classes)

    # For processing objects we require at least one importable class.
    # An empty summary is usually caused by hidden import errors and should fail docs build.
    if include_io and modules and not valid_classes:
        raise RuntimeError(
            'No classes extracted while generating processing-objects summary. '
            f'Modules scanned: {len(modules)}; modules without extracted classes: '
            f'{len(skipped_modules)}. '
            'This is commonly caused by import errors in processing object modules.'
        )

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

        desc = info['doc'] if info['doc'] else '*No description available.*'
        wrapped_lines = textwrap.wrap(
            desc,
            width=50,
            break_long_words=False,
            break_on_hyphens=False,
        )
        cell_content = '\n       | '.join(wrapped_lines)
        if len(wrapped_lines) > 1:
            cell_content = '| ' + cell_content
        lines.append(f'     - {cell_content}')

        if has_io:
            inputs = info.get('named_inputs') or {}
            outputs = info.get('named_outputs') or []

            in_list = [
                f"{format_port_name(k)} *(opt)*" if opt else format_port_name(k)
                for k, opt in inputs.items()
            ]
            out_list = [format_port_name(o) for o in outputs]
            in_str = ', '.join(in_list) if in_list else '-'
            out_str = ', '.join(out_list) if out_list else '-'

            in_lines = textwrap.wrap(
                in_str,
                width=30,
                break_long_words=False,
                break_on_hyphens=False,
            )
            in_wrapped = '\n       | '.join(in_lines)
            if len(in_lines) > 1:
                in_wrapped = '| ' + in_wrapped

            out_lines = textwrap.wrap(
                out_str,
                width=30,
                break_long_words=False,
                break_on_hyphens=False,
            )
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
            cat['description'],
            include_io=cat.get('include_io', False),
        )
        out_file = api_docs_path / f"{cat['filename']}.rst"
        out_file.write_text(content, encoding='utf-8')
        print(f"  -> Generated {out_file}")

    print('\nDone.')


if __name__ == '__main__':
    main()
