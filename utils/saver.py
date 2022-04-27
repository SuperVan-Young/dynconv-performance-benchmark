import json

def log_ls_to_json(ls, path):
    """save list to path
    """
    with open(path, "a") as f:
        for l in ls:
            f.write(json.dumps(l)+"\n")

def load_json_to_ls(path):
    r = []
    with open(path, "r") as f:
        for l in f.readlines():
            r.append(json.loads(l))
    return r