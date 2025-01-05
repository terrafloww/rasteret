""" Logging configuration for the rasteret package. """

import logging
import sys
from typing import Optional


def setup_logger(
    level: Optional[str] = "INFO", customname: Optional[str] = "rasteret"
) -> None:
    """
    Set up library-wide logging configuration.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    # Create formatters
    detailed_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(detailed_formatter)

    # Configure root logger for the package
    root_logger = logging.getLogger(name=customname)
    root_logger.setLevel(getattr(logging, level))

    # Configure logging - suppress httpx logs
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    # Remove existing handlers and add our handler
    root_logger.handlers = []
    root_logger.addHandler(console_handler)

    # Prevent propagation to root logger
    root_logger.propagate = False

    return root_logger
