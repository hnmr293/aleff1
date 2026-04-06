from logging import getLogger


_logger = getLogger(__name__)


def loglevel(level: int):
    _logger.setLevel(level)


def debug(msg: str):
    _logger.debug(msg)
