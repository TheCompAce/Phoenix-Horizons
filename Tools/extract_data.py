import json
import argparse
import os

def extract_data(keys, data):
    result = {}
    for key in keys:
        if key in data:
            value = data[key]
            if isinstance(value, str) and key.endswith("File"):
                if os.path.isfile(value):
                    with open(value, 'r') as file:
                        result[key[:-4]] = json.load(file)  # remove 'File' from key
                else:
                    print(f"Warning: File {value} not found.")
            elif isinstance(value, list) and key == "systemArchitecture":
                result[key] = []
                for item in value:
                    result[key].append(extract_data(["name", "description", "designFile"], item))
            else:
                result[key] = value
    return result

def main():
    parser = argparse.ArgumentParser(description='Extract data from JSON file.')
    parser.add_argument('json_file', help='The JSON file to extract data from.')
    parser.add_argument('-k', '--keys', help='Comma-separated list of keys to extract.')
    args = parser.parse_args()

    keys = args.keys.split(',')

    with open(args.json_file, 'r') as file:
        data = json.load(file)

    result = extract_data(keys, data)

    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
