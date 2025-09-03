# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause
import logging
from logging import CRITICAL, DEBUG, ERROR, FATAL, INFO, WARNING

import structlog
from structlog import get_logger

from megatron.energon.logging.setup import configure_structlog

logger = structlog.get_logger(__name__)

# Ensure std-lib users don't see "No handlers could be found"
logging.getLogger(__name__).addHandler(logging.NullHandler())
logging.captureWarnings(True)


# Add TRACE level
TRACE = 5
logging.addLevelName(TRACE, "TRACE")


def trace(self, message, *args, **kwargs):
    if self.isEnabledFor(TRACE):
        self._log(TRACE, message, args, **kwargs)


logging.Logger.trace = trace  # type: ignore


__all__ = [
    "configure_structlog",
    "get_logger",
    "CRITICAL",
    "DEBUG",
    "ERROR",
    "FATAL",
    "INFO",
    "TRACE",
    "WARNING",
]
