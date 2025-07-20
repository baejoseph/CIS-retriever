import time
import humanfriendly
import logging
from functools import wraps
from logger import logger

def log_time(label: str):
    """
    Decorator to time a function/method and log the elapsed time in natural language.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            logger.info(f"Started process '{label}'")
            try:
                return func(*args, **kwargs)
            finally:
                elapsed = time.perf_counter() - start
                human_time = humanfriendly.format_timespan(elapsed)
                logger.info(f"Process '{label}' took {human_time}.")
        return wrapper
    return decorator

class ProcessTimer:
    def __init__(self):
        self._start_times = {}
    
    def mark(self, label: str):
        """
        Call this to start timing for the given label.
        """
        self._start_times[label] = time.perf_counter()
        logger.info(f"Started process '{label}'")

    def done(self, label: str):
        """
        Call this to stop timing for the given label.
        Logs the elapsed time in natural language.
        """
        start = self._start_times.pop(label, None)
        if start is None:
            logger.warning(f"No start time recorded for process '{label}'")
            return
        elapsed_seconds = time.perf_counter() - start
        human_time = humanfriendly.format_timespan(elapsed_seconds)
        logger.info(f"Process '{label}' took {human_time}.")