import json


def read_json_objs(filename):
    with open(filename) as f:
        objs = []
        for obj in f.readlines():
            objs.append(json.loads(obj))
    return objs

