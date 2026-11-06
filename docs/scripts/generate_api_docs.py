"""
Generate the API pages (docs/api/*.rst) of SPECULA and of iteras, its
simulation framework.

The SPECULA pages list only the modules defined in SPECULA: the modules moved
to iteras are kept in SPECULA as aliases of the iteras modules, and are
documented only in the iteras pages.
"""
import pkgutil
import re
from pathlib import Path

# Last line of the modules moved to iteras, which are aliases of the iteras modules
ALIAS = re.compile(r'^sys\.modules\[__name__\] = ([\w.]+)$', re.MULTILINE)

ROOT = Path(__file__).parent.parent.parent
API_DOCS_PATH = Path(__file__).parent.parent / "api"

# Top-level modules that are not documented
SKIPPED_TOP_LEVEL = {'__init__', '_version'}


def is_alias(py_file):
    """True for a module moved to iteras, which is only an alias of the iteras module"""
    return ALIAS.search(py_file.read_text(encoding='utf-8')) is not None


def scan_package_modules(package_path, package_name):
    """Modules of a package (not its subpackages), without the aliases of iteras modules"""
    modules = []
    if not package_path.exists():
        return modules
    for _, modname, ispkg in pkgutil.iter_modules([str(package_path)]):
        if ispkg or modname in SKIPPED_TOP_LEVEL:
            continue
        if is_alias(package_path / f"{modname}.py"):
            continue
        modules.append(f"{package_name}.{modname}")
    return sorted(modules, key=lambda m: m.split('.')[-1])


def generate_simple_api_doc(title, modules, description=""):
    """RST page documenting the members of each module of *modules*"""
    content = f"""{title}
{'=' * len(title)}

{description}

"""
    for module in modules:
        module_title = module.split('.')[-1].replace('_', ' ').title()
        content += f"""
{module_title}
{'-' * len(module_title)}

.. automodule:: {module}
   :members:
   :undoc-members:
   :show-inheritance:


"""
    return content


# (package directory relative to ROOT, package name, page title, description, file name)
PAGES = [
    ('specula', 'specula', 'Core Modules API',
     'Core modules of SPECULA. The base classes and the simulation framework are '
     'in iteras: see :doc:`iteras_core`.',
     'core'),
    ('specula/processing_objects', 'specula.processing_objects', 'Processing Objects API',
     'Processing objects for simulating AO system components. The generic ones '
     '(filters, generators, data storage...) are in iteras: see '
     ':doc:`iteras_processing_objects`.',
     'processing_objects'),
    ('specula/data_objects', 'specula.data_objects', 'Data Objects API',
     'Data objects for representing simulation data. The generic ones are in '
     'iteras: see :doc:`iteras_data_objects`.',
     'data_objects'),
    ('specula/lib', 'specula.lib', 'Utility Functions API',
     'Utility functions and libraries. The generic ones are in iteras: see '
     ':doc:`iteras_lib`.',
     'lib'),
    ('iteras', 'iteras', 'iteras Core API',
     'Simulation framework: base classes, simulation, connections, accelerator, '
     'tracing and plugin registry.',
     'iteras_core'),
    ('iteras/processing_objects', 'iteras.processing_objects', 'iteras Processing Objects API',
     'Generic processing objects: filters, generators, operations, data storage '
     'and displays.',
     'iteras_processing_objects'),
    ('iteras/data_objects', 'iteras.data_objects', 'iteras Data Objects API',
     'Generic data objects.',
     'iteras_data_objects'),
    ('iteras/lib', 'iteras.lib', 'iteras Utility Functions API',
     'Generic utility functions and libraries.',
     'iteras_lib'),
    ('iteras/allocator', 'iteras.allocator', 'iteras Allocator API',
     'Automatic allocation of the objects on the available resources '
     '(see :doc:`../allocator`).',
     'iteras_allocator'),
    ('iteras/display', 'iteras.display', 'iteras Displays API',
     'Displays of the simulation outputs.',
     'iteras_display'),
]


def main():
    API_DOCS_PATH.mkdir(exist_ok=True)
    for path, package, title, description, filename in PAGES:
        modules = scan_package_modules(ROOT / path, package)
        print(f"{package}: {len(modules)} modules")
        output_file = API_DOCS_PATH / f"{filename}.rst"
        output_file.write_text(generate_simple_api_doc(title, modules, description),
                               encoding='utf-8')
        print(f"  -> Generated {output_file}")


if __name__ == "__main__":
    main()
