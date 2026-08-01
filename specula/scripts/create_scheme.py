import json
import logging
import sys
import typing
from pathlib import Path
from typing import Type, Union, Callable, List, Optional

import numpy as np

from docs.scripts.generate_objects_summary import scan_package
from specula.scripts.parse_classes import extract_class_info, ClassData

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
    _dont_remove_imports_for_these = [List, Optional]

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
        # number seems to be mostly correct, but propably the typing should be fixed
        return {"type": "array", "items": {"type": "number"}}

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
        "$schema": "http://json-schema.org/draft-07/schema#",
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
                "tag": {"type": "string"},
            },
            "required": ["class"],
            "additionalProperties": False,
        }

        for param_name in classdata.param_type.keys():
            if param_name in ("self", "precision", "target_device_idx",):
                continue

            # Normal parameter without special postfix
            class_scheme["properties"][param_name] = map_python_type_to_json(
                classdata.param_type[param_name],
                debug_info=f"{classdata.class_name}.{param_name}")
            class_scheme["properties"][param_name + "_ref"] = {"type": "string"}
            class_scheme["properties"][param_name + "_data"] = {"type": "string"}
            class_scheme["properties"][param_name + "_object"] = {"type": "string"}
            if param_name.endswith("_dict"):
                class_scheme["properties"][param_name + "_ref"] = {"oneOf": [
                    {
                        "type": "string"
                    },
                    {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                ]}

            postfixes = ["", "_ref", "_data", "_object"]
            if "allOf" not in class_scheme:
                class_scheme["allOf"] = []

            # Only one of the postfixes can be used, or a tag
            class_scheme["allOf"].append(
                {
                    "oneOf": [{"required": [param_name + postfix]} for postfix in postfixes] + [{"required": ["tag"]}]
                })
            # If optional, also none of them can be uesd
            if not classdata.param_required[param_name]:
                class_scheme["allOf"][-1]["oneOf"].append({
                    "not": {
                        "anyOf": [{"required": [param_name + postfix]} for postfix in postfixes]
                    }
                })

        if classdata.inputs:
            class_scheme["required"].append("inputs")
            class_scheme["properties"]["inputs"] = {
                "type": "object",
                "properties": {
                    k: {
                        False: {"type": "string"},
                        # If the input name ends with _list it can also be a list of strings, otherwise a string
                        True: {"oneOf": [
                            {"type": "string"},
                            {
                                "type": "array",
                                "items": {"type": "string"},
                            }]},
                    }[k.endswith("_list")]
                    for k in classdata.inputs.keys()
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
                # Size is a bit flakey, due to outputs declared in parent classes
                # "minItems": len(classdata.outputs),
                # "maxItems": len(classdata.outputs),
                "items": {"type": "string"},
            }

        schema["additionalProperties"]["oneOf"].append(class_scheme)

    txt = custom_json_dumps(schema)
    out_file.open("w", encoding="utf-8").write(txt)
    print(f"  -> Generated {out_file}")
    print('\nDone.')


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


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python create_scheme.py <output_file>")
        sys.exit(1)

    output_file = Path(sys.argv[1]).expanduser().resolve()

    create_scheme(output_file)
