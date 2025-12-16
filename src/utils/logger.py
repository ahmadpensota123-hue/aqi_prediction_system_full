"""
Logging Utilities
=================

This module provides centralized logging configuration for the AQI Prediction System.
It uses Loguru for enhanced logging capabilities including:
- Colored console output
- File rotation
- Structured logging
- Easy configuration

Why Loguru instead of standard logging?
- Much simpler API (no handlers, formatters, etc.)
- Built-in colors and formatting
- Automatic file rotation
- Exception catching with full traceback
- Thread-safe
"""

import sys
from pathlib import Path
from loguru import logger

from src.config.settings import get_settings


def setup_logging(
    log_level: str = None,
    log_file: str = None,
    rotation: str = "10 MB",
    retention: str = "7 days",
) -> None:
    """
    Configure logging for the application.

    This function sets up:
    1. Console logging with colors
    2. File logging with rotation (optional)

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Path to log file (optional)
        rotation: When to rotate log file (e.g., "10 MB", "1 day")
        retention: How long to keep log files

    Example:
        >>> setup_logging(log_level="DEBUG", log_file="logs/app.log")
    """
    settings = get_settings()

    # Use provided log level or fall back to settings
    level = log_level or settings.app.log_level

    # Remove default logger
    logger.remove()

    # Console logging format
    console_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )

    # File logging format (no colors)
    file_format = (
        "{time:YYYY-MM-DD HH:mm:ss} | "
        "{level: <8} | "
        "{name}:{function}:{line} | "
        "{message}"
    )

    # Add console handler
    logger.add(sys.stdout, format=console_format, level=level, colorize=True)

    # Add file handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        logger.add(
            log_file,
            format=file_format,
            level=level,
            rotation=rotation,
            retention=retention,
            compression="zip",
        )

    logger.info(f"Logging configured with level: {level}")


def get_logger(name: str = None):
    """
    Get a logger instance.

    Args:
        name: Optional name for the logger (usually __name__)

    Returns:
        Logger instance with the specified name

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing started")
        >>> logger.error("Something went wrong", exc_info=True)
    """
    if name:
        return logger.bind(name=name)
    return logger


class LogContext:
    """
    Context manager for adding contextual information to logs.

    Useful for tracking request IDs, user IDs, or other context
    through a chain of function calls.

    Example:
        >>> with LogContext(request_id="abc123", user="john"):
        ...     logger.info("Processing request")
        # Output: ... | request_id=abc123 user=john | Processing request
    """

    def __init__(self, **kwargs):
        self.context = kwargs
        self._token = None

    def __enter__(self):
        self._token = logger.configure(extra=self.context)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Context is automatically cleaned up
        pass


def log_execution_time(func):
    """
    Decorator to log function execution time.

    Example:
        >>> @log_execution_time
        ... def slow_function():
        ...     time.sleep(1)
        ...
        >>> slow_function()
        # Output: slow_function executed in 1.00s
    """
    import functools
    import time

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start_time
        logger.debug(f"{func.__name__} executed in {elapsed:.2f}s")
        return result

    return wrapper


def log_exception(func):
    """
    Decorator to log exceptions with full traceback.

    Example:
        >>> @log_exception
        ... def risky_function():
        ...     raise ValueError("Something went wrong")
    """
    import functools

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.exception(f"Exception in {func.__name__}: {e}")
            raise

    return wrapper


# Initialize logging with default settings when module is imported
setup_logging()
