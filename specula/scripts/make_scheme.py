import inspect
import json
import logging
import typing
from typing import get_origin, get_args


def map_python_type_to_json(py_type) -> dict:
    """
    Map Python-types to JSON-scheme types
    """
    # e.g Union[float, List[float]]
    if get_origin(py_type) is typing.Union:
        return {"oneOf": [map_python_type_to_json(arg) for arg in get_args(py_type)]}

    # e.g. List[str])
    if get_origin(py_type) is list:
        args = get_args(py_type)
        item_type = map_python_type_to_json(args[0]) if args else {"type": "string"}
        return {"type": "array", "items": item_type}

    # Standard types
    mapping = {
        str: {"type": "string"},
        int: {"type": "integer"},
        float: {"type": "number"},
        bool: {"type": "boolean"}
    }

    if py_type in mapping:
        return mapping[py_type]

    logging.warning(f"Unhandled type {py_type}, fallback to string")
    return {"type": "string"}


def generate_json_schema(classes_list) -> dict:
    """
    Generate the yaml json-scheme from the given classes using their __init__ method.
    Uses the type annotations to infer the type. Parameters without a default values are required.
    """
    ignore_params = (
        "self",
        "target_device_idx",
        "precision",
    )
    
    schema = {
        "$schema": "https://json-schema.org",
        "title": "Auto-Generated Simulation Schema",
        "type": "object",
        "additionalProperties": {
            "type": "object",
            "oneOf": []
        }
    }

    for cls in classes_list:
        class_name = cls.__name__
        sig = inspect.signature(cls.__init__)

        properties = {
            "class": {"const": class_name}
        }
        required_fields = ["class"]

        for param_name, param in sig.parameters.items():
            if param_name in ignore_params:
                continue

            # get type hint, default to str
            py_type = param.annotation if param.annotation != inspect.Parameter.empty else str
            properties[param_name] = map_python_type_to_json(py_type)

            if param.default == inspect.Parameter.empty:
                required_fields.append(param_name)

        class_definition = {
            "required": required_fields,
            "additionalProperties": False,
            "properties": properties
        }

        schema["additionalProperties"]["oneOf"].append(class_definition)

    return schema


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
    from specula.data_objects.simul_params import SimulParams
    from specula.processing_objects.wave_generator import WaveGenerator

    target_classes = [SimulParams, WaveGenerator]

    generated_schema = generate_json_schema(target_classes)
    txt = custom_json_dumps(generated_schema)
    print(txt)

    with open("specula.schema.json", "w", encoding="utf-8") as f:
        f.write(txt)
