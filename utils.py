import json

def load_json(path):
    with open(path, 'r') as json_file:
        dataset = json.load(json_file)
    return dataset