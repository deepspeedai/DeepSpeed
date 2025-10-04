# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import functools
import logging
import sys
import os
import torch
from deepspeed.utils.torch import required_torch_version

log_levels = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}


class LoggerFactory:

    @staticmethod
    def create_logger(name=None, level=logging.WARNING):
        """create a logger

        Args:
            name (str): name of the logger
            level: level of logger

        Raises:
            ValueError is name is None
        """

        if name is None:
            raise ValueError("name for logger cannot be None")

        formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] "
                                      "[%(filename)s:%(lineno)d:%(funcName)s] %(message)s")

        logger_ = logging.getLogger(name)
        logger_.setLevel(level)
        logger_.propagate = False
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(level)
        ch.setFormatter(formatter)
        logger_.addHandler(ch)
        if required_torch_version(min_version=2.6) and os.getenv("DISABLE_LOGS_WHILE_COMPILING", "0") == "1":
            excluded_set = {
                item.strip()
                for item in os.getenv("LOGGER_METHODS_TO_EXCLUDE_FROM_DISABLE", "").split(",")
            }
            ignore_set = {'info', 'debug', 'error', 'warning', 'critical', 'exception', 'isEnabledFor'} - excluded_set
            for method in ignore_set:
                original_logger = getattr(logger_, method)
                torch._dynamo.config.ignore_logger_methods.add(original_logger)
        return logger_


logger = LoggerFactory.create_logger(name="DeepSpeed", level=logging.WARNING)


@functools.lru_cache(None)
def warning_once(*args, **kwargs):
    """
    This method is identical to `logger.warning()`, but will emit the warning with the same message only once

    Note: The cache is for the function arguments, so 2 different callers using the same arguments will hit the cache.
    The assumption here is that all warning messages are unique across the code. If they aren't then need to switch to
    another type of cache that includes the caller frame information in the hashing function.
    """
    logger.warning(*args, **kwargs)


logger.warning_once = warning_once


def print_configuration(args, name):
    logger.info("{}:".format(name))
    for arg in sorted(vars(args)):
        dots = "." * (29 - len(arg))
        logger.info("  {} {} {}".format(arg, dots, getattr(args, arg)))


def get_dist_msg(message, ranks=None):
    from deepspeed import comm as dist
    """Return a message with rank prefix when one of following conditions is met:

      + not dist.is_initialized()
      + dist.get_rank() in ranks if ranks is not None or ranks = [-1]

    If neither is met, `None` is returned.

    Example: "hello" => "[Rank 0] hello"

    Args:
        message (str)
        ranks (list)
    """
    should_log = not dist.is_initialized()
    ranks = ranks or []
    my_rank = dist.get_rank() if dist.is_initialized() else -1
    if ranks and not should_log:
        should_log = ranks[0] == -1
        should_log = should_log or (my_rank in set(ranks))
    if should_log:
        return "[Rank {}] {}".format(my_rank, message)
    else:
        return None


def log_dist(message, ranks=None, level=logging.INFO):
    """Log message when get_dist_msg() deems it should be logged, see its docstring for details.

    Args:
        message (str)
        ranks (list)
        level (int)
    """
    final_message = get_dist_msg(message, ranks)
    if final_message is not None:
        logger.log(level, final_message)


def print_dist(message, ranks=None):
    """print message when get_dist_msg() deems it should be logged, see its docstring for details.

    Use this function instead of `log_dist` when the log level shouldn't impact whether the message should be printed or not.

    Args:
        message (str)
        ranks (list)
    """
    final_message = get_dist_msg(message, ranks)
    if final_message is not None:
        print(final_message)


@functools.lru_cache(None)
def _log_dist_once_cached(message, ranks_key, level):
    ranks_arg = list(ranks_key) if ranks_key is not None else None
    log_dist(message, ranks=ranks_arg, level=level)


def log_dist_once(message, ranks=None, level=logging.INFO):
    # Identical to `log_dist`, but will emit each unique message only once per process.
    # ranks is a list which is unhashable, so convert to tuple for caching
    ranks_key = tuple(ranks) if ranks is not None else None
    _log_dist_once_cached(message, ranks_key, level)


logger.log_dist_once = log_dist_once


def print_json_dist(message, ranks=None, path=None):
    from deepspeed import comm as dist
    """Print message when one of following condition meets

    + not dist.is_initialized()
    + dist.get_rank() in ranks if ranks is not None or ranks = [-1]

    Args:
        message (str)
        ranks (list)
        path (str)

    """
    should_log = not dist.is_initialized()
    ranks = ranks or []
    my_rank = dist.get_rank() if dist.is_initialized() else -1
    if ranks and not should_log:
        should_log = ranks[0] == -1
        should_log = should_log or (my_rank in set(ranks))
    if should_log:
        message['rank'] = my_rank
        import json
        with open(path, 'w') as outfile:
            json.dump(message, outfile)
            os.fsync(outfile)


def get_log_level_from_string(log_level_str):
    """converts a log level string into its numerical equivalent. e.g. "info" => `logging.INFO`
    """
    log_level_str = log_level_str.lower()
    if log_level_str not in log_levels:
        raise ValueError(
            f"{log_level_str} is not one of the valid logging levels. Valid log levels are {log_levels.keys()}.")
    return log_levels[log_level_str]


def set_log_level_from_string(log_level_str, custom_logger=None):
    """Sets a log level in the passed `logger` and its handlers from string. e.g. "info" => `logging.INFO`

    Args:
        log_level_str: one of 'debug', 'info', 'warning', 'error', 'critical'
        custom_logger: if `None` will use the default `logger` object
    """
    log_level = get_log_level_from_string(log_level_str)
    if custom_logger is None:
        custom_logger = logger
    custom_logger.setLevel(log_level)
    for handler in custom_logger.handlers:
        handler.setLevel(log_level)


def get_current_level():
    """
    Return logger's current log level
    """
    return logger.getEffectiveLevel()


def should_log_le(max_log_level_str):
    """
    Args:
        max_log_level_str: maximum log level as a string

    Returns ``True`` if the current log_level is less or equal to the specified log level. Otherwise ``False``.

    Example:

        ``should_log_le("info")`` will return ``True`` if the current log level is either ``logging.INFO`` or ``logging.DEBUG``
    """

    if not isinstance(max_log_level_str, str):
        raise ValueError(f"{max_log_level_str} is not a string")

    max_log_level = get_log_level_from_string(max_log_level_str)
    return get_current_level() <= max_log_level
