import ast
import json
import logging
import os
import sys
import typing
from pathlib import Path
from typing import Type, Union, Callable

import numpy as np

from docs.scripts.generate_objects_summary import scan_package

# All the classes are:
# AtmoEvolution.yml          BaseOperation.yml      DataStore.yml      FlaskServer.yml    ImRecCalibrator.yml           
# ModalAnalysisWFS.yml  MultiImRecCalibrator.yml    Slopec.yml
# AtmoInfiniteEvolution.yml  BaseProcessingObj.yml  DisplayServer.yml  FuncGenerator.yml  Integrator.yml                
# ModalAnalysis.yml     ProcessingContainer.yml   ShSlopec.yml           SnCalibrator.yml
# AtmoPropagation.yml        CCD.yml                DM.yml             IdealWFS.yml       LowPassFilter.yml             
# Modalrec.yml          PSF.yml                   ShSubapCalibrator.yml  Vibrations.yml
# AtmoRandomPhase.yml        DataSource.yml         Factory.yml        IirFilter.yml      MirrorCommandsCombinator.yml  
# ModulatedPyramid.yml  PyrSlopec.yml             SH.yml                 WindowedIntegration.yml


# TODO new generator classes not 
exposed_classes = ['Source', 'Pupilstop',
                   'FuncGenerator', 'BaseOperation', 'AtmoEvolution', 'AtmoInfiniteEvolution', 'AtmoPropagation',
                   'ModulatedPyramid', 'CCD', 'Slopec', 'PyrSlopec', 'Modalrec', 'Integrator', 'IirFilter', 'DM', 'PSF',
                   'DataStore'
                   ]

known_referenced_classes: list[str] = ["BaseProcessingObj"]
"""
List of all classes to detect if they are referenced as type in an __init__ statement.
Filled by create_scheme, used by map_python_type_to_json.
"""


def map_python_type_to_json(py_type: str | Type | None, *,
                            warn_parseerror: bool = True,
                            warn_untyped: bool = False,
                            debug_info: str = "") -> dict[str, str]:
    """
    Map Python-types to JSON-scheme types.
    Given type can be
    - None (no typing information available)
    - type itself (e.g. str, dict)
    - string (as returned by the AST parser)
    """
    # If we got a string, try to convert it to an actual type, except if it is a known class that is referenced
    # e.g. obj: SimulParams
    if isinstance(py_type, str):
        if py_type in known_referenced_classes:
            return {"type": "string"}

        # Create fake types of all known classes:
        #   Support things like List[Recmat] where Recmat is in known classes,
        #   but the eval fails since it is not imported.
        #   This is a bit hacky, but works. Alternatively we could check for List[Recmat] explicitly,
        #   since it is the only one.
        for k in known_referenced_classes:
            locals()[k] = type(k)

        try:
            py_type = eval(py_type)
        except NameError as e:
            if warn_parseerror:
                logging.warning(f"Type {py_type} not parseable, {e}. {debug_info}")
            return {"type": "string"}

    if py_type is type(None) or py_type is None:
        if warn_untyped:
            logging.warning(f"Untyped parameter, default to string. {debug_info}")
        return {"type": "string"}

    # e.g Union[float, List[float]]
    if typing.get_origin(py_type) is Union:
        return {"oneOf": [map_python_type_to_json(arg) for arg in typing.get_args(py_type)]}

    # e.g. List[str])
    if typing.get_origin(py_type) is list:
        args = typing.get_args(py_type)
        item_type = map_python_type_to_json(args[0]) if args else {"type": "string"}
        return {"type": "array", "items": item_type}

    # e.g. list (without element type)
    if py_type in (list, tuple):
        if warn_untyped:
            logging.warning(f"list/tuple without element type hint, default to number. {debug_info}")
        return {"type": "array",
                "items": "number"}  # number seems to be mostly correct, but propably the typing should be fixed

    if py_type is np.ndarray:
        return {"type": "array", "items": {"type": "number"}}

    if py_type is dict:
        return {"type": "object"}

    if py_type is Callable:
        # Note: not sure what we should do here, e.g. DisplayServer defines callables als init arguments
        return {"type": "string"}

    # Standard types
    mapping = {
        str: {"type": "string"},
        int: {"type": "integer"},
        float: {"type": "number"},
        bool: {"type": "boolean"}
    }

    if py_type in mapping:
        return mapping[py_type]

    logging.warning(f"Unhandled type {py_type}, fallback to string. {debug_info}")
    return {"type": "string"}


class ClassData:
    def __init__(self, class_name: str):
        self.class_name: str = class_name
        self.init_params = {}
        self.param_type = {}
        self.param_comments = {}
        self.param_required = {}
        self.inputs = {}
        self.outputs = []


class InitMethodVisitor(ast.NodeVisitor):
    """AST Visitor to extract parameters, inputs, and outputs from an __init__ method."""

    def __init__(self, class_name: str):
        self.data = ClassData(class_name=class_name)

    def visit_FunctionDef(self, node):
        """Visit the __init__ method and extract parameters, inputs, and outputs."""
        if node.name == "__init__":
            total_params = len(node.args.args) - 1  # Exclude 'self'
            num_defaults = len(node.args.defaults)

            # Extract type hints from function annotations
            annotations = {arg.arg: ast.unparse(arg.annotation) if arg.annotation else None for arg in node.args.args[1:]}  # Skip 'self'

            for i, arg in enumerate(node.args.args[1:]):  # Skip 'self'
                param_name = arg.arg
                param_type = annotations.get(param_name, None)
                is_optional = i >= (total_params - num_defaults)
                
                default_value = "None"
                if is_optional:
                    default_index = i - (total_params - num_defaults)
                    default_node = node.args.defaults[default_index]
                    
                    try:
                        default_value = ast.literal_eval(default_node)
                    except (ValueError, TypeError, AttributeError):
                        if isinstance(default_node, ast.Name):
                            default_value = default_node.id  # Handle names like `np`
                        else:
                            default_value = "None"

                # Construct comment with type and optional status
                comment = "Required" if not is_optional else f"Optional (default={default_value})"
                if param_type:
                    comment += f", type: {param_type}"

                self.data.init_params[param_name] = default_value
                self.data.param_comments[param_name] = comment
                self.data.param_type[param_name] = param_type
                self.data.param_required[param_name] = not is_optional

            # Visit the body of __init__ to extract inputs and outputs
            for statement in node.body:
                self.visit(statement)

    def visit_Assign(self, node):
        """Extract input and output specifications from self.inputs and self.outputs assignments."""
        if isinstance(node.targets[0], ast.Subscript):
            target = node.targets[0]
            if isinstance(target.value, ast.Attribute) and target.value.attr in ["inputs", "outputs"]:
                key = target.slice.value if isinstance(target.slice, ast.Constant) else target.slice

                if target.value.attr == "inputs" and isinstance(node.value, ast.Call):
                    # Extract input type from: self.inputs['input_name'] = InputValue(type=InputType)
                    for keyword in node.value.keywords:
                        if keyword.arg == "type":
                            input_type = ast.unparse(keyword.value)
                            self.data.inputs[key] = input_type

                elif target.value.attr == "outputs":
                    # Extract output from: self.outputs['out_value'] = self.out_value
                    self.data.outputs.append(key)


def extract_class_info(file_path, allowed=None):
    """Extracts class name, __init__ method parameters, default values, and types from a Python file."""
    if allowed is None:
        allowed = lambda _c: _c in exposed_classes

    with open(file_path, "r", encoding="utf-8") as file:
        tree = ast.parse(file.read(), filename=file_path)

    class_data = []

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            class_name = node.name
            if not allowed(class_name):
                continue
            visitor = InitMethodVisitor(class_name=class_name)
            visitor.visit(node)
            class_data.append(visitor.data)

    return class_data

def generate_yaml(data: ClassData, output_folder):
    """Generates a YAML file with class information, inputs, and outputs."""
    yaml_path = os.path.join(output_folder, f"{data.class_name}.yml")

    with open(yaml_path, "w", encoding="utf-8") as yaml_file:
        yaml_file.write(f"{data.class_name}:\n")

        # Write constructor parameters
        for param, value in data.init_params.items():
            # yaml_file.write(f"  {param}: {value}  # {comments[param]}\n")
            yaml_file.write(f"  {param}: {value}\n")

        # Write inputs
        if data.inputs:
            yaml_file.write("  inputs:\n")
            for input_name, input_type in data.inputs.items():
                # yaml_file.write(f"    {input_name}: {input_type}  # InputType\n")
                yaml_file.write(f"    {input_name}: {input_type}\n")

        # Write outputs as a YAML list
        if data.outputs:
            yaml_file.write(f"  outputs: {data.outputs}\n")

    print(f"Generated YAML: {yaml_path}")


def process_python_files(input_folder, output_folder):
    """Processes all Python files in a directory and generates YAML files."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file_name in os.listdir(input_folder):
        if file_name.endswith(".py"):
            file_path = os.path.join(input_folder, file_name)
            classes = extract_class_info(file_path)

            for classdata in classes:
                generate_yaml(classdata, output_folder)


def custom_json_dumps(obj, indent=4, max_line_len=60):
    """
    Formats JSON with indentation, but collapses short dictionaries
    and lists into a single line if they fit within max_line_len.
    """
    # Create a compact version first
    compact = json.dumps(obj, separators=(',', ': '))

    # If the entire object fits on one line, return it
    if len(compact) <= max_line_len:
        return compact

    # If it's a dictionary, decide whether to split or keep flat
    if isinstance(obj, dict):
        # If the dict is empty, keep it tight
        if not obj:
            return "{}"

        # Check if all items inside fit comfortably on one line
        # (We estimate the length including key-value pairs)
        estimated_len = sum(len(json.dumps(k)) + len(json.dumps(v)) + 4 for k, v in obj.items())
        if estimated_len <= max_line_len and not any(isinstance(v, (dict, list)) for v in obj.values()):
            return compact

        # Otherwise, break keys into multiple lines recursively
        space = " " * indent
        lines = []
        for k, v in obj.items():
            formatted_value = custom_json_dumps(v, indent, max_line_len)
            # Indent subsequent lines if the value spans multiple lines
            if "\n" in formatted_value:
                formatted_value = formatted_value.replace("\n", "\n" + space)
            lines.append(f"{space}{json.dumps(k)}: {formatted_value}")

        return "{\n" + ",\n".join(lines) + "\n}"

    # If it's a list, process elements recursively
    if isinstance(obj, list):
        if not obj:
            return "[]"

        space = " " * indent
        # Try to format all items; if any item has a newline, the list must split
        formatted_items = [custom_json_dumps(item, indent, max_line_len) for item in obj]

        if any("\n" in item for item in formatted_items) or len(compact) > max_line_len:
            lines = []
            for item in formatted_items:
                if "\n" in item:
                    item = item.replace("\n", "\n" + space)
                lines.append(f"{space}{item}")
            return "[\n" + ",\n".join(lines) + "\n]"

        return compact

    # Base case for primitive types (strings, numbers, booleans, None)
    return compact


def create_scheme(out_file: Path):
    specula_path = Path(__file__).parent.parent.parent / 'specula'

    categories = [
        {
            'path': specula_path / 'processing_objects',
            'package': 'specula.processing_objects',
        },
        {
            'path': specula_path / 'data_objects',
            'package': 'specula.data_objects',
        },
    ]

    schema = {
        "$schema": "https://json-schema.org",
        "title": "Auto-Generated Simulation Schema",
        "type": "object",
        "additionalProperties": {
            "type": "object",
            "oneOf": []
        }
    }

    all_class_infos: list[ClassData] = []
    for cat in categories:
        print(f"Scanning {cat['path']}...")
        modules: list[tuple[str, Path]] = scan_package(cat['path'], cat['package'])
        for modulename, modulepath in modules:
            all_class_infos.extend(extract_class_info(modulepath, allowed=lambda _: True))

    known_referenced_classes.extend([c.class_name for c in all_class_infos])

    for classdata in all_class_infos:
        class_scheme = {
            "properties": {
                "class": {"const": classdata.class_name},
            },
            "required": ["class"],
            "additionalProperties": False,
        }

        for param_name in classdata.param_type.keys():
            if param_name in ("self", "precision", "target_device_idx",):
                continue

            class_scheme["properties"][param_name] = map_python_type_to_json(
                classdata.param_type[param_name],
                debug_info=f"{classdata.class_name}.{param_name}")
            if classdata.param_required[param_name]:
                class_scheme["required"].append(param_name)

        if classdata.inputs:
            class_scheme["required"].append("inputs")
            class_scheme["properties"]["inputs"] = {
                "type": "object",
                "properties": {
                    k: map_python_type_to_json(str) for k in classdata.inputs.keys()
                },
                "required": list(classdata.inputs.keys()),
            }

        # Note: unconditionally add outputs.
        #   Some (child) classes dont declare their own output,
        #   and the AST parser does not detect outputs declared in parents.
        if classdata.outputs or True:
            # Filter out BinOp
            classdata.outputs = [o for o in classdata.outputs if isinstance(o, str)]
            assert all(isinstance(k, str) for k in classdata.outputs)
            # class_scheme["required"].append("outputs")
            class_scheme["properties"]["outputs"] = {
                "type": "array",
                "minItems": len(classdata.outputs),
                "maxItems": len(classdata.outputs),
                "items": {"type": "string"},
            }

        schema["additionalProperties"]["oneOf"].append(class_scheme)

    txt = custom_json_dumps(schema)
    out_file.open("w", encoding="utf-8").write(txt)
    print(f"  -> Generated {out_file}")
    print('\nDone.')


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python parse_classes.py <input_folder> <output_folder>")
        sys.exit(1)

    input_folder = sys.argv[1]
    output_folder = sys.argv[2]

    if not os.path.isdir(input_folder):
        print(f"Error: Input folder '{input_folder}' does not exist or is not a directory.")
        sys.exit(1)

    process_python_files(input_folder, output_folder)

    create_scheme(Path.home() / "specula.schema.json")
