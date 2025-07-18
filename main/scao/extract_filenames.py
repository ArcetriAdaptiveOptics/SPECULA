import yaml
import argparse

import specula
specula.init(-1)

from specula.lib.utils import import_class, get_type_hints
from specula.calib_manager import CalibManager


def extract_tagged_data(yaml_file_path):
    with open(yaml_file_path, 'r') as file:
        data = yaml.safe_load(file)

    extracted = []

    suffix_map = {
        "_tag": "tag",
        "_object": "object",
        "_data": "data"
    }

    if isinstance(data, dict):
        for entry_name, entry in data.items():
            if isinstance(entry, dict):
                cls = entry.get("class")
                for key, value in entry.items():
                    match_type = None
                    base_key = key

                    if key == "tag":
                        match_type = "tag"
                    else:
                        for suffix, t in suffix_map.items():
                            if key.endswith(suffix):
                                match_type = t
                                base_key = key[:-len(suffix)]
                                break

                    if match_type:
                        extracted.append({
                            "entry": entry_name,
                            "key": base_key,
                            "value": value,
                            "class": cls,
                            "type": match_type
                        })
    else:
        print("YAML root is not a dictionary.")

    return extracted


def main():
    parser = argparse.ArgumentParser(description="Extract specific keys and class values from a YAML file.")
    parser.add_argument("yaml_file", help="Path to the YAML file")
    args = parser.parse_args()

    results = extract_tagged_data(args.yaml_file)

    with open(args.yaml_file, 'r') as file:
        data = yaml.safe_load(file)
        root_dir = data['main']['root_dir']

    cm = CalibManager(root_dir)
    paths = []
    for item in results:
        if item['type'] == 'tag':
            path = cm.filename(item['class'], item['value'])
        elif item['type'] == 'object':
            path = cm.filename(item['key'], item['value'])
        elif item['type'] == 'data':
            path = cm.filename('data', item['value'])

        paths.append(path)

    unique_res = set(paths)

    for r in unique_res:

        print('rsync -PavzR ' + r + ' frossi04@data.leonardo.cineca.it:/leonardo_work/try25_rossi/MORFEO_DATA')


if __name__ == "__main__":
    main()

