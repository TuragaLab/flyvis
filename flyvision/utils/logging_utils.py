from functools import lru_cache


@lru_cache(100)
def warn_once(logger, msg: str):
    logger.warning(msg)