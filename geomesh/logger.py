import logging
import sys


def init(log_level=None):
    if log_level is not None:
        logging.basicConfig(level={
            "info": logging.INFO,
            "debug": logging.DEBUG,
            "warning": logging.WARNING,
        }[log_level])
        logging.getLogger('matplotlib').setLevel(logging.WARNING)
        logging.getLogger('fiona').setLevel(logging.CRITICAL)
        logging.getLogger('rasterio').setLevel(logging.WARNING)


log_level = "debug" if "--debug" in sys.argv else None
init(log_level)
