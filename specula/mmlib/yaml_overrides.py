import yaml
import ast

# Custom classes to control YAML formatting styles
class QuotedStr(str): pass
class InlineList(list): pass

def write_yaml_overrides(input_string):
    def quoted_presenter(dumper, data):
        return dumper.represent_scalar('tag:yaml.org,2002:str', data, style="'")

    def inline_list_presenter(dumper, data):
        return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)

    # Register custom styles
    yaml.SafeDumper.add_representer(QuotedStr, quoted_presenter)
    yaml.SafeDumper.add_representer(InlineList, inline_list_presenter)

    def parse_value(val_raw):
        val_clean = val_raw.strip()
        # Handle booleans first (case-insensitive)
        if val_clean.lower() == 'true': return True
        if val_clean.lower() == 'false': return False
        
        try:
            parsed = ast.literal_eval(val_clean)
            if isinstance(parsed, str):
                return QuotedStr(parsed)
            if isinstance(parsed, list):
                # Wrap list and its internal strings
                return InlineList([QuotedStr(x) if isinstance(x, str) else x for x in parsed])
            return parsed
        except (ValueError, SyntaxError):
            return val_clean

    def set_nested_value(d, keys, value):
        keys[0] += '_override'
        for key in keys[:-1]:
            d = d.setdefault(key, {})
        d[keys[-1]] = value

    # Remove {} brackets
    input_string = input_string[1:-1]

    # Split input string by comma, respecting bracketed lists
    data_dict = {}
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
    if current: pairs.append("".join(current).strip())

    # Build the dictionary
    for pair in pairs:
        if ':' not in pair: continue
        key_path, val_raw = [x.strip() for x in pair.split(':', 1)]
        set_nested_value(data_dict, key_path.split('.'), parse_value(val_raw))

    # Write file: default_flow_style=False keeps objects as blocks
    with open('temp_overrides.yml', 'w') as f:
        yaml.dump(data_dict, f, Dumper=yaml.SafeDumper, default_flow_style=False, sort_keys=False)