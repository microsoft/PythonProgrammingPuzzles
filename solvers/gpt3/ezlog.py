import os
import logging
import inspect
import io

my_path = os.path.dirname(__file__)


def color_str(obj, code="\033[0;36m"):
    return code + str(obj) + '\033[0m'


_configured = False


def configure_logging(stdio_level=logging.INFO,
                      file_level=logging.DEBUG,
                      filename=".easy.log",
                      filepath=os.path.join(my_path, "logs")):
    os.makedirs(filepath, exist_ok=True)
    filename = os.path.join(filepath, filename)
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
    #
    # fh = logging.FileHandler('spam.log')
    # fh.setLevel(logging.DEBUG)
    # # create console handler with a higher log level
    # ch = logging.StreamHandler()
    # ch.setLevel(logging.ERROR)
    # # create formatter and add it to the handlers
    # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # fh.setFormatter(formatter)
    # ch.setFormatter(formatter)
    # # add the handlers to the logger
    # logger.addHandler(fh)
    # logger.addHandler(ch)

    _configured = True
    _get_or_create_logger().debug("Configured logging")


_loggers = {}


def _get_or_create_logger():
    global _configured, _loggers
    if not _configured:
        configure_logging()
    try:
        for frame in inspect.stack():
            name = inspect.getmodule(frame[0]).__name__
            if name != __name__:
                break
    except:
        name = "_"
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
