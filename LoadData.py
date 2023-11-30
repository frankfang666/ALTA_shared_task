import json

def read_json_objs(filename):
    with open(filename) as f:
        objs = []
        for obj in f.readlines():
            try:
                objs.append(json.loads(obj))
            except json.decoder.JSONDecodeError:
                continue
    return objs

def write_json_objs(filename, objs):
    with open(filename, 'w+') as f:
        for obj in objs:
            f.write(json.dumps(obj)+'\n')