import time
import logging

logger = logging.getLogger(__name__)

def timeit(method):
    def timed(*args, **kw):
        tick = time.time()
        result = method(*args, **kw)
        tock = time.time()
        logger.debug(f'{method.__name__}: {tock - tick:.3f}s')

        return result
    return timed
