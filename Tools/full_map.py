import json
import os
import argparse
from typing import Any, Dict, List, Union

def load_json(filename: str) -> Dict[str, Any]:
    with open(filename, 'r') as f:
        return json.load(f)

def save_json(filename: str, data: Dict[str, Any]) -> None:
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)

def update_data(data: Union[Dict[str, Any], List[Any]], base_dir: str) -> Union[Dict[str, Any], List[Any]]:
    if isinstance(data, dict):
        for key, value in data.items():
            data[key] = update_data(value, base_dir)
    elif isinstance(data, list):
        for i, item in enumerate(data):
            data[i] = update_data(item, base_dir)
    elif isinstance(data, str) and data.endswith('.json'):
        file_path = os.path.join(base_dir, data)
        if os.path.isfile(file_path):
            data = load_json(file_path)
    return data

def main(input_file: str, output_file: str = None) -> None:
    if output_file is None:
        output_file = 'full_' + os.path.basename(input_file)
    base_dir = os.path.dirname(os.path.abspath(input_file))
    data = load_json(input_file)
    updated_data = update_data(data, base_dir)
    save_json(output_file, updated_data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Expand a JSON file with nested JSON files.')
    parser.add_argument('input_file', type=str, help='Input JSON file')
    parser.add_argument('--output_file', type=str, default=None, help='Output JSON file')
    args = parser.parse_args()
    main(args.input_file, args.output_file)
