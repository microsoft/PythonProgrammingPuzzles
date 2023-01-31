import json
import logging
import inspect
import io
import os
import sys
import time
from transformers import AutoTokenizer


os.environ["WANDB_DISABLED"] = "true" 
os.environ["TOKENIZERS_PARALLELISM"] = "false"
my_path = os.path.dirname(__file__)


def load_tokenizer(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def num_tokens(s: str, tokenizer, verbose=False):

    start_time = time.time()
    if verbose:
        info(f"Tokenizing {pretty_int(len(s))} chars ({pretty_int(len(s.splitlines()))} lines)")
    # ans = _tokenizer(s, return_tensors="pt").input_ids.shape[1] # produces annoying warnings
    ans = tokenizer(s, return_tensors="pt", max_length=10 + len(s), truncation=True).input_ids.shape[1]
            
    duration_mins = (time.time() - start_time)/60
    if verbose:
        info(f"Num tokens: {ans:,} in {duration_mins:.2f} mins")
    return ans

  
def create_experiment_outpath(out: str, bSaveCommand=True):
    """
    Create the output directory and return its name. Also stores the command line in command.sh
    Date format is like Jan-1-2020
    """
    output_path = str(out).replace("<date>", time.strftime("%b%d-%H-%M-%S"))
    os.makedirs(output_path, exist_ok=True)  # ran into error due to non-atomic check
    if bSaveCommand:
        save_text_file(' '.join([sys.executable] + sys.argv) + "\n", f"{output_path}/command.sh")
        # make command.sh executable:
        os.chmod(f"{output_path}/command.sh", 0o755)
    return output_path

def pretty_int(n: int) -> str:
    """Converts an integer to a string with commas, with M for millions and B for billions"""
    if n > 1_000_000_000:
        return f"{n/1_000_000_000:.1f}B"
    if n > 1_000_000:
        return f"{n/1_000_000:.1f}M"
    return f"{n:,}"



def test_puzzle(f, x):
    """Checks if x is of the correct type and makes f return True (literally True, not an integer or whatever)

    :param f: Puzzle
    :param x: candidate answer
    :return:
    """
    answer_type = list(f.__annotations__.values())[0]
    if not type_check(x, answer_type):
        raise TypeError
    return f(x) is True



def type_check(obj, typ):
    """
    check if obj is of type `typ` where `typ` is a `typing` module type annotation, eg List[int]
    The way we do this to be compatible across versions is we first convert the type to a string.
    """

    type_str = str(typ).replace("typing.", "")
    if type_str.startswith("<class '"):
        type_str = type_str[8:-2]


    def helper(obj, type_st: str):
        """test if obj is of type type_st"""
        t = {"str": str, "int": int, "float": float, "bool": bool}.get(type_st)
        if t is not None:
            return type(obj) == t
        assert type_st.endswith("]"), f"Strange type `{type_st}`"
        inside = type_st[type_st.index("[")+1:-1].split(", ")
        if type_st.startswith("List["):
            [i] = inside
            return isinstance(obj, list) and all(type_check(elem, i) for elem in obj)
        if type_st.startswith("Set"):
            [i] = inside
            return isinstance(obj, set) and all(type_check(elem, i) for elem in obj)
        print(f"type not handled: {typ}")
        return True

    return helper(obj, type_str)



def check_hashseed(desired_seed = 0):
    if os.environ.get('PYTHONHASHSEED') != desired_seed:
        info(f"Ideally set PYTHONHASHSEED={desired_seed} for perfect reproducibility")
        return False
    return True


def inv_dict(d):
    """Invert a dictionary to {val: [list of keys], ...}"""
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
    if str(filename).endswith(".gz"):
        import gzip
        with gzip.open(filename, "wt") as f:
            return json.dump(obj, f, indent=indent, **kwargs)
    with open(filename, "w", encoding="utf8") as f:
        return json.dump(obj, f, indent=indent, **kwargs)

def load_text_file(filename):
    """Loads text file, decompresses if filename ends with '.gz'"""
    if str(filename).endswith(".gz"):
        import gzip
        with gzip.open(filename, "rt") as f:
            return f.read()
    with open(filename, "r", encoding="utf8") as f:
        return f.read()


def save_text_file(contents: str, filename):
    """Save text file, compresses if filename ends with '.gz'"""
    if str(filename).endswith(".gz"):
        import gzip
        with gzip.open(filename, "wt") as f:
            f.write(contents)
    else:
        with open(filename, "w", encoding="utf8") as f:
            f.write(contents)


def load_json(filename):
    """Loads compressed file if filename ends with '.gz'"""
    import json
    if str(filename).endswith(".gz"):
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
                      path=os.path.join(my_path, "../logs/"),
                      filename=os.path.basename(sys.argv[0]).replace(".py", "") + ".log"):
    global _configured
    if _configured:
        warning("Re-configuring logging")
    # create path if necessary
    os.makedirs(path, exist_ok=True)
    stdio_handler = logging.StreamHandler()
    stdio_handler.setLevel(stdio_level)
    file_hanlder = logging.FileHandler(os.path.join(path, filename))
    file_hanlder.setLevel(file_level)

    logging.basicConfig(
        format="%(asctime)s:%(levelname)s:%(name)s:%(message).512s",
        datefmt="%Y/%m/%d %H:%M:%S",
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


  
_std_errs = None

def silence_std_err(quiet=True):
    global _std_errs
    if _std_errs is None:
        _std_errs = {"orig": os.dup(2), "devnull": os.open(os.devnull, os.O_RDWR)}
    if quiet:
        os.dup2(_std_errs["devnull"], 2)  # to avoid printing the s_push parser when parsing stuff with "((((()))))"
    else:
        os.dup2(_std_errs["orig"], 2)


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

