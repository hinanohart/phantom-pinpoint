"""Lightweight logging helper used across the package.

Avoids ``logging.basicConfig`` so that downstream applications retain control
over root configuration.  Emits ``INFO`` level by default to stderr in the
``phantom_pinpoint`` namespace.
"""

from __future__ import annotations

import logging
import os
import sys
from logging import Logger

_LOGGER_NAME = "phantom_pinpoint"
_DEFAULT_LEVEL = os.environ.get("PHANTOM_PINPOINT_LOGLEVEL", "INFO").upper()
_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_DATEFMT = "%Y-%m-%dT%H:%M:%S"


def get_logger(name: str | None = None) -> Logger:
    """Return a namespaced logger with a single stderr handler.

    Parameters
    ----------
    name:
        Sub-logger name.  Joined to ``phantom_pinpoint`` with a dot.  Pass
        ``None`` to obtain the package root logger.

    Returns
    -------
    logging.Logger
        A logger that has exactly one ``StreamHandler`` writing to ``stderr``.
        The handler is attached only on first call to keep idempotency.
    """
    full = _LOGGER_NAME if name is None else f"{_LOGGER_NAME}.{name}"
    logger = logging.getLogger(full)
    if not getattr(logger, "_phantom_pinpoint_configured", False):
        handler = logging.StreamHandler(stream=sys.stderr)
        handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
        logger.addHandler(handler)
        logger.setLevel(_DEFAULT_LEVEL)
        logger.propagate = False
        logger._phantom_pinpoint_configured = True  # type: ignore[attr-defined]
    return logger
