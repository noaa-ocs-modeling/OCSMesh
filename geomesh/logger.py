import logging


class Logger:

    def __get__(self, obj, val):
        logger = obj.__dict__.get('logger')
        if logger is None:
            logger = logging.getLogger(f"{obj.__class__.__name__}")
            obj.__dict__['logger'] = logger
        return logger

    def __set__(self, obj, val):
        if not isinstance(val, logging.Logger):
            raise TypeError(
                f'Property logger must be of type {logging.Logger}, '
                f'not type {type(val)}.')
        obj.__dict__['logger'] = val
