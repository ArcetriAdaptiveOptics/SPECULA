import pkgutil
import re
from pathlib import Path

# Last line of the modules moved to iteras, which are aliases of the iteras modules
ALIAS = re.compile(r'^sys\.modules\[__name__\] = ([\w.]+)$', re.MULTILINE)


def documented_module(py_file, module_name):
    """
    Name of the module to document for a file: for a module moved to iteras,
    the iteras module, since autodoc documents only the members defined in
    the module it is given.
    """
    match = ALIAS.search(py_file.read_text(encoding='utf-8'))
    return match.group(1) if match else module_name

def scan_package_modules(package_path, package_name):
    """Scan a package and return all its modules"""
    modules = []
    if not package_path.exists():
        return modules

    for importer, modname, ispkg in pkgutil.iter_modules([str(package_path)]):
        if not ispkg:  # Only include modules, not subpackages
            modules.append(documented_module(package_path / f"{modname}.py",
                                             f"{package_name}.{modname}"))
    return sorted(modules, key=lambda m: m.split('.')[-1])

def scan_base_classes(specula_path):
    """Scan main specula directory for base class files"""
    base_files = []
    base_patterns = ['base_', 'calib_', 'connection', 'field_', 'template_']  # Patterns to look for

    for py_file in specula_path.glob("*.py"):
        module_name = py_file.stem
        # Include files that start with 'base_' or contain 'connection'
        if any(pattern in module_name for pattern in base_patterns):
            base_files.append(documented_module(py_file, f"specula.{module_name}"))

    return sorted(base_files, key=lambda m: m.split('.')[-1])

def generate_simple_api_doc(category_name, modules, description="", include_members=True):
    """Generate simple RST content for all modules in a category"""
    title = f"{category_name} API"
    content = f"""{title}
{'=' * len(title)}

{description}

"""
    
    for module in modules:
        module_title = module.split('.')[-1].replace('_', ' ').title()
        members_block = ""
        if include_members:
            members_block = (
                "   :members:\n"
                "   :undoc-members:\n"
                "   :show-inheritance:\n"
            )
        content += f"""
{module_title}
{'-' * len(module_title)}

.. automodule:: {module}
{members_block}

"""
    return content

def main():
    # Base paths
    specula_path = Path(__file__).parent.parent.parent / "specula"
    iteras_path = Path(__file__).parent.parent.parent / "iteras"
    api_docs_path = Path(__file__).parent.parent / "api"

    # Create api directory if it doesn't exist
    api_docs_path.mkdir(exist_ok=True)

    # Define directories to scan
    directories = {
        "Processing Objects": {
            "path": specula_path / "processing_objects",
            "package": "specula.processing_objects",
            "description": "Processing objects for simulating AO system components.",
            "filename": "processing_objects"
        },
        "Data Objects": {
            "path": specula_path / "data_objects",
            "package": "specula.data_objects",
            "description": "Data objects for representing simulation data.",
            "filename": "data_objects"
        },
        "Utility Functions": {
            "path": specula_path / "lib",
            "package": "specula.lib",
            "description": "Utility functions and libraries.",
            "filename": "lib",
        }
    }

    # Generate API docs for each directory
    for category_name, config in directories.items():
        print(f"Scanning {config['path']} for modules...")

        # Scan for all modules in the package, and in the iteras package of
        # the same kind (the modules moved there are found in both)
        iteras_package = config["package"].replace("specula.", "iteras.", 1)
        iteras_modules = scan_package_modules(iteras_path / config["path"].name, iteras_package)
        all_modules = sorted(set(scan_package_modules(config["path"], config["package"])) |
                             set(iteras_modules), key=lambda m: m.split('.')[-1])

        if all_modules:
            print(f"  Found modules: {[m.split('.')[-1] for m in all_modules]}")

            # Generate content
            content = generate_simple_api_doc(
                category_name,
                all_modules,
                config["description"],
                include_members=config.get("include_members", True),
            )

            # Write the file
            output_file = api_docs_path / f"{config['filename']}.rst"
            with open(output_file, 'w') as f:
                f.write(content)

            print(f"  -> Generated {output_file}")
        else:
            print(f"  No modules found in {config['path']}")

    # Generate base_classes.rst by scanning main specula directory
    print(f"Scanning {specula_path} for base class modules...")
    base_modules = scan_base_classes(specula_path)

    if base_modules:
        print(f"  Found base modules: {[m.split('.')[-1] for m in base_modules]}")

        base_content = generate_simple_api_doc(
            "Base Classes",
            base_modules,
            "Base classes for SPECULA processing objects and data structures."
        )

        base_file = api_docs_path / "base_classes.rst"
        with open(base_file, 'w') as f:
            f.write(base_content)
        print(f"  -> Generated {base_file}")
    else:
        print(f"  No base class modules found in {specula_path}")

    print("\nAPI documentation generation complete!")
    print("\nGenerated files:")
    for rst_file in sorted(api_docs_path.glob("*.rst")):
        print(f"  - {rst_file.name}")


if __name__ == "__main__":
    main()