import json
import logging
import inspect
import io
import os

my_path = os.path.dirname(__file__)

def check_hashseed(desired_seed = 0):
    if os.environ.get('PYTHONHASHSEED') != desired_seed:
        info(f"Ideally set PYTHONHASHSEED={desired_seed} for perfect reproducibility")
        return False
    return True


def inv_dict(d):
    ans = {}
    for k, v in d.items():
        if v not in ans:
            ans[v] = []
        ans[v].append(k)
    return ans


def remove_docstring(f):
    """Remove docstring"""
    assert '\n    """' in f, f"No triple quote docstring (after four spaces) in: \n{f}"
    i = f.index('\n    """')
    j = f.index('"""', i + 8)
    return f[:i + 1] + f[j + 4:]


def get_docstring(f):
    assert '\n    """' in f, f"No triple quote docstring (after four spaces) in: \n{f}"
    i = f.index('\n    """')
    j = f.index('"""', i + 8)
    docstring = f[i + 1:j + 3]
    if not docstring.strip(' "'):
        warn(f"Empty docstring in:\n{f}")
    return docstring


def flatten(it):
    return (e for a in it for e in (flatten(a) if isinstance(a, (tuple, list)) else (a,)))

def save_json(obj, filename, make_dirs_if_necessary=False, indent=2, **kwargs):
    """Saves compressed file if filename ends with '.gz'"""
    import json
    if make_dirs_if_necessary:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
    if filename.endswith(".gz"):
        import gzip
        with gzip.open(filename, "wt") as f:
            return json.dump(obj, f, indent=indent, **kwargs)
    with open(filename, "w", encoding="utf8") as f:
        return json.dump(obj, f, indent=indent, **kwargs)

def load_json(filename):
    """Loads compressed file if filename ends with '.gz'"""
    import json
    if filename.endswith(".gz"):
        import gzip
        with gzip.open(filename, "rt") as f:
            return json.load(f)
    with open(filename, "r", encoding="utf8") as f:
        return json.load(f)


def stringify(const):
    if type(const) is str:
        return json.dumps(const)
    return str(const)


def dedup(stuff):
    seen = set()
    return [a for a in stuff if a not in seen and not seen.add(a)]


def color_str(obj, code="\033[0;36m"):
    return code + str(obj) + '\033[0m'


_configured = False


def configure_logging(stdio_level=logging.INFO,
                      file_level=logging.DEBUG,
                      filename=os.path.join(my_path, ".problems.log")):
    global _configured
    if _configured:
        warning("Re-configuring logging")
    stdio_handler = logging.StreamHandler()
    stdio_handler.setLevel(stdio_level)
    file_hanlder = logging.FileHandler(filename)
    file_hanlder.setLevel(file_level)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message).200s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=min(stdio_level, file_level),
        handlers=[stdio_handler, file_hanlder]
    )

    _configured = True
    _get_or_create_logger().debug("Configured logging")


_loggers = {}


def _get_or_create_logger():
    global _configured, _loggers
    if not _configured:
        configure_logging()
    name = "_"
    for frame in inspect.stack():
        name = inspect.getmodule(frame[0]).__name__
        if name != __name__:
            break
    if name not in _loggers:
        _loggers[name] = logging.getLogger(name)
    return _loggers[name]


def print_to_string(*args, end="", **kwargs):
    with io.StringIO() as buf:
        print(*args, file=buf, end=end, **kwargs)
        return buf.getvalue()


def debug(*args, **kwargs):
    _get_or_create_logger().debug(print_to_string(*args, **kwargs))


def info(*args, **kwargs):
    _get_or_create_logger().info(print_to_string(*args, **kwargs))


log = info


def warning(*args, **kwargs):
    _get_or_create_logger().warning(print_to_string(*args, **kwargs))


warn = warning


def error(*args, **kwargs):
    _get_or_create_logger().error(print_to_string(*args, **kwargs))
