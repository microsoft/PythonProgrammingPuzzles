import json

def get_lambda_arg_name(lam):
    assert lam.startswith("lambda ")
    return lam[len("lambda "):lam.index(":")].strip()


def stringify(const):
    if type(const) is str:
        return json.dumps(const)
    return str(const)


def color_str(obj, code="\033[0;36m"):
    return code + str(obj) + '\033[0m'

