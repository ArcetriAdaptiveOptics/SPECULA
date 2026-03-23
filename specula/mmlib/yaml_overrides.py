import yaml
import ast

# Custom class to tell the YAML dumper to use quotes
class QuotedStr(str):
    pass

def quoted_presenter(dumper, data):
    return dumper.represent_scalar('tag:yaml.org,2002:str', data, style="'")

# Add the representer to SafeDumper
yaml.SafeDumper.add_representer(QuotedStr, quoted_presenter)

def parse_value(val_raw):
    val_clean = val_raw.strip()
    
    # Manually handle booleans to ensure they are never treated as strings
    if val_clean.lower() == 'true':
        return True
    if val_clean.lower() == 'false':
        return False
    
    try:
        # Detects if input is a quoted string, number, or list
        parsed = ast.literal_eval(val_clean)
        if isinstance(parsed, str):
            return QuotedStr(parsed)
        if isinstance(parsed, list):
            # Recursively handle strings inside lists
            return [QuotedStr(x) if isinstance(x, str) else x for x in parsed]
        return parsed
    except (ValueError, SyntaxError):
        # Fallback for unquoted strings in the input string
        return val_clean

def set_nested_value(d, keys, value):
    """Recursively creates nested dictionaries for keys like ['obj', 'sub', 'param']."""
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = value

def write_yaml_overrides(input_string):
    input_string = input_string[1:-1]

    data_dict = {}
    
    # Split by comma while respecting commas inside lists [ ]
    pairs = []
    bracket_level = 0
    current = []
    for char in input_string:
        if char == '[': bracket_level += 1
        elif char == ']': bracket_level -= 1
        if char == ',' and bracket_level == 0:
            pairs.append("".join(current).strip())
            current = []
        else:
            current.append(char)
    if current:
        pairs.append("".join(current).strip())

    for pair in pairs:
        if ':' not in pair: continue
        key_path, val_raw = [x.strip() for x in pair.split(':', 1)]
        
        value = parse_value(val_raw)
        keys = key_path.split('.')
        keys[0] += '_override'
        set_nested_value(data_dict, keys, value)

    # Mode 'w' overwrites the file every time
    with open('temp_overrides.yml', 'w') as f:
        yaml.dump(data_dict, f, Dumper=yaml.SafeDumper, default_flow_style=False, sort_keys=False)

# # Custom class to tell the YAML dumper to use quotes
# class QuotedStr(str):
#     pass

# def quoted_presenter(dumper, data):
#     return dumper.represent_scalar('tag:yaml.org,2002:str', data, style="'")

# yaml.add_representer(QuotedStr, quoted_presenter)

# def write_yaml_overrides(input_string):
#     data_dict = {}

#     input_string = input_string[1:-1]
    
#     # 1. Split by comma, respecting brackets [ ]
#     pairs = []
#     bracket_level = 0
#     current_char_list = []
#     for char in input_string:
#         if char == '[': bracket_level += 1
#         elif char == ']': bracket_level -= 1
#         if char == ',' and bracket_level == 0:
#             pairs.append("".join(current_char_list).strip())
#             current_char_list = []
#         else:
#             current_char_list.append(char)
#     pairs.append("".join(current_char_list).strip())

#     # 2. Parse pairs
#     for pair in pairs:
#         if ':' not in pair: continue
#         key_part, val_raw = [x.strip() for x in pair.split(':', 1)]
        
#         # 3. Use ast.literal_eval to detect types based on input quotes
#         try:
#             parsed_val = ast.literal_eval(val_raw)
            
#             if isinstance(parsed_val, str):
#                 # If it was a quoted string in input, wrap it for the dumper
#                 val = QuotedStr(parsed_val)
#             elif isinstance(parsed_val, list):
#                 # Handle lists: wrap strings inside the list if they were quoted
#                 val = [QuotedStr(x) if isinstance(x, str) else x for x in parsed_val]
#             else:
#                 val = parsed_val # float, int, bool
#         except (ValueError, SyntaxError):
#             val = val_raw # Fallback

#         # 4. Nesting logic (obj.param)
#         if '.' in key_part:
#             obj, param = key_part.split('.', 1)
#             obj += '_override'
#             data_dict.setdefault(obj, {})[param] = val
#         else:
#             data_dict[key_part] = val

#     # 5. Overwrite the file
#     with open('temp_overrides.yml', 'w') as f:
#         yaml.dump(data_dict, f, default_flow_style=False, sort_keys=False)

# def write_yaml_overrides(input_string):
#     """
#     Parses an override string into a YAML file with correct types.
#     Handles floats, lists, and quoted strings.
#     """

    
#     # Custom representer to force single quotes around all strings
#     def string_representer(dumper, data):
#         return dumper.represent_scalar('tag:yaml.org,2002:str', data, style="'")

#     yaml.add_representer(str, string_representer)

#     data_dict = {}
    
#     # 1. Split by comma, but respect commas inside brackets [ ]
#     # We use a simple loop to avoid splitting lists incorrectly
#     pairs = []
#     bracket_level = 0
#     current_pair = []
#     for char in input_string:
#         if char == '[': bracket_level += 1
#         elif char == ']': bracket_level -= 1
        
#         if char == ',' and bracket_level == 0:
#             pairs.append("".join(current_pair).strip())
#             current_pair = []
#         else:
#             current_pair.append(char)
#     pairs.append("".join(current_pair).strip())

#     # 2. Process each pair
#     for pair in pairs:
#         if ':' not in pair:
#             continue
            
#         key_part, value_raw = pair.split(':', 1)
#         key_part = key_part.strip()
#         value_raw = value_raw.strip()

#         # 3. Convert string values to Python types (float, list, etc.)
#         try:
#             # ast.literal_eval handles numbers and lists safely
#             value = ast.literal_eval(value_raw)
#         except (ValueError, SyntaxError):
#             # Fallback for plain strings that aren't wrapped in quotes
#             value = value_raw
        
#         # 4. Handle nested object.param structure
#         if '.' in key_part:
#             obj, param = key_part.split('.', 1)
#             obj += '_override'
#             if obj not in data_dict:
#                 data_dict[obj] = {}
#             data_dict[obj][param] = value
#         else:
#             data_dict[key_part] = value

#     # 5. Write to file (mode 'w' overwrites every time)
#     with open('temp_overrides.yml', 'w') as f:
#         yaml.dump(data_dict, f, default_flow_style=False, sort_keys=False)

# def write_yaml_overrides(input_string):
#     """
#     Parses a string of overrides and writes them to temp_overrides.yml.
#     Format: "obj1.param1: value1, obj2.param2: 'string2'"
#     """
#     data_dict = {}
    
#     input_string = input_string[1:-1]
#     # Split the string into individual "key: value" pairs
#     pairs = [item.strip() for item in input_string.split(',')]
    
#     for pair in pairs:
#         if ':' not in pair:
#             continue
            
#         key_part, value = pair.split(':', 1)
#         key_part = key_part.strip()
#         value = value.strip()
        
#         # Strip explicit quotes if they wrap the entire value
#         if (value.startswith("'") and value.endswith("'")) or (value.startswith('"') and value.endswith('"')):
#             value = value[1:-1]
        
#         # Handle the nesting (e.g., obj1.param1)
#         if '.' in key_part:
#             obj, param = key_part.split('.', 1)
#             obj += '_override'
#             if obj not in data_dict:
#                 data_dict[obj] = {}
#             data_dict[obj][param] = value
#         else:
#             data_dict[key_part] = value

#     # Write the dictionary to the file in block style
#     with open('temp_overrides.yml', 'w') as f:
#         # default_flow_style=False ensures the clean, indented YAML look
#         yaml.dump(data_dict, f, default_flow_style=False, sort_keys=False)