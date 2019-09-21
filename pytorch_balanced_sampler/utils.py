import logging
import logging.config


logging_level_dict = {
    0: logging.WARNING,
    1: logging.INFO,
    2: logging.DEBUG
}


DEFAULT_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'simple': {
            'format': "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'DEBUG',
            'formatter': 'simple',
            'stream': 'ext://sys.stdout'
        }
    }
}


def setup_logging(config=DEFAULT_CONFIG):
    """Setup logging configuration"""
    logging.config.dictConfig(config)


def setup_logger(cls, name='', verbose=0):
    logger = logging.getLogger(name)
    if verbose not in logging_level_dict:
        raise KeyError(f'Verbose option {verbose} for {name} not valid. '
                        'Valid options are {logging_level_dict.keys()}.')
    logger.setLevel(logging_level_dict[verbose])
    return logger


setup_logging()
