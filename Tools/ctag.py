import ast
import os
import argparse
import requests
import json
import hashlib
from ctags.PythonTagGenerator import PythonTagGenerator

def generate_ctags(path, exclude_dirs, diff):
    if not diff:
        # If diff option is False, remove the cache.json file
        if os.path.exists('cache.json'):
            os.remove('cache.json')

    exclude_dirs = set(exclude_dirs)
    tag_file = "ctags/tags.json"  # Change to JSON
    os.makedirs(os.path.dirname(tag_file), exist_ok=True)
    cache = CacheManager()
    with open(tag_file, 'w') as f:
        tag_generator = PythonTagGenerator()
        tags = []
        metrics = []
        for root, dirs, files in os.walk(path):
            dirs[:] = [d for d in dirs if d not in exclude_dirs]  # Skip excluded directories
            for filename in files:
                if filename.endswith('.py'):
                    file_path = os.path.join(root, filename)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as source_file:
                            code = source_file.read()
                    except UnicodeDecodeError:
                        with open(file_path, 'r', encoding='ISO-8859-1') as source_file:
                            code = source_file.read()

                    file_hash = hashlib.md5(code.encode()).hexdigest()
                    if cache.get(file_path) == file_hash:
                        continue
                    cache.set(file_path, file_hash)
                    try:
                        tag_generator.process_source_code(file_path, code)  # Call generate_tags instead of visit
                        tags.extend(tag_generator.tags)
                        metrics.extend(tag_generator.metrics)

                    except SyntaxError as e:
                        print(f"Syntax error in file {file_path}: {e}")

            
        with open(tag_file, 'w') as f:
            json.dump(tags, f)

        tag_generator.write_metrics_to_file(metrics)
        
    cache.cleanup()

def get_openai_key():
    try:
        with open('options.json', 'r') as file:
            options = json.load(file)
            if 'openai_key' in options:
                return options['openai_key']
    except FileNotFoundError:
        pass

    openai_key = input("Please enter your OpenAI API key: ")

    try:
        with open('options.json', 'r') as file:
            options = json.load(file)
    except FileNotFoundError:
        options = {}

    options['openai_key'] = openai_key

    with open('options.json', 'w') as file:
        json.dump(options, file)

    return openai_key

class CacheManager:
    def __init__(self, cache_file='tmp/cache.json'):
        self.cache_file = cache_file
        os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
        try:
            with open(self.cache_file, 'r') as f:
                self.cache = json.load(f)
        except FileNotFoundError:
            self.cache = {}

    def get(self, key):
        return self.cache.get(key, None)

    def set(self, key, value):
        self.cache[key] = value
        with open(self.cache_file, 'w') as f:
            json.dump(self.cache, f)

    def cleanup(self):
        keys_to_remove = [key for key in self.cache.keys() if not os.path.exists(key)]
        for key in keys_to_remove:
            del self.cache[key]
        with open(self.cache_file, 'w') as f:
            json.dump(self.cache, f)

def analyze_code_with_openai(code, openai_key):
    cache = CacheManager()
    code_hash = hashlib.md5(code.encode()).hexdigest()
    cached_result = cache.get(code_hash)
    if cached_result is not None:
        return cached_result

    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {openai_key}',
    }
    data = {
        'model': 'text-davinci-002',
        'prompt': code,
        'temperature': 0.5,
        'max_tokens': 100,
    }
    response = requests.post('https://api.openai.com/v1/engines/davinci/completions', headers=headers, data=json.dumps(data))
    result = response.json()['choices'][0]['text'].strip()
    cache.set(code_hash, result)
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs='+')
    parser.add_argument("--exclude", "-e", default='', type=str)
    parser.add_argument('--diff', '-d', action='store_true', help='Use existing cache.json file.')
    parser.add_argument("--analyze", action='store_true')
    args = parser.parse_args()
    exclude_dirs = args.exclude.split(',') if args.exclude else []
    for path in args.paths:
        generate_ctags(path, exclude_dirs, args.diff)

        if args.analyze:
            openai_key = get_openai_key()
            with open(path, 'r') as file:
                code = file.read()
                print(analyze_code_with_openai(code, openai_key))
