# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

import structlog
from structlog.dev import ConsoleRenderer, RichTracebackFormatter
from structlog.processors import ExceptionRenderer, JSONRenderer, TimeStamper
from structlog.stdlib import ProcessorFormatter
from structlog.tracebacks import ExceptionDictTransformer


def unwrap_callables(logger, name, event_dict):
    for key, value in event_dict.items():
        if callable(value):
            event_dict[key] = value()
    return event_dict


def configure_structlog(
    console_log_level: int = logging.INFO,
    file_log_level: int = logging.INFO,
    log_path: Path | None = None,
):
    """Configure structlog for the application.

    Logging to console can happen at a different level than logging to file.
    If you provide a log_path, it will be used to log to a file, otherwise it will
    log to console only.

    Args:
        console_log_level: The log level to log to the console.
        file_log_level: The log level to log to the file.
        log_path: The path to the file to log to. If not provided, logging will only happen to the console.
    """

    minimum_log_level = min(console_log_level, file_log_level)

    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        TimeStamper(fmt="%Y-%m-%d %H:%M:%S"),
        structlog.processors.StackInfoRenderer(),
        unwrap_callables,
    ]

    structlog.configure(
        processors=shared_processors + [ProcessorFormatter.wrap_for_formatter],
        wrapper_class=structlog.make_filtering_bound_logger(minimum_log_level),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(minimum_log_level)
    root_logger.handlers.clear()  # start clean

    console_formatter = ProcessorFormatter(
        processor=ConsoleRenderer(
            exception_formatter=RichTracebackFormatter(
                show_locals=True,  # Display local variables in the traceback
                locals_max_length=10,  # Limit variable display length
            )
        ),
        foreign_pre_chain=shared_processors,
    )

    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(console_log_level)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    if log_path:
        file_formatter = ProcessorFormatter(
            processors=[
                ProcessorFormatter.remove_processors_meta,
                ExceptionRenderer(ExceptionDictTransformer(show_locals=True)),
                JSONRenderer(),
            ],
            foreign_pre_chain=shared_processors,
        )
        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setLevel(file_log_level)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
