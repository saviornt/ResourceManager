import os
import logging
import sys
import atexit
from datetime import datetime
from typing import Optional, List

# Global list to track handlers for proper cleanup
_handlers_to_cleanup: List[logging.Handler] = []

def setup_logging(logs_dir: str = "./logs", 
                  log_level: int = logging.INFO, 
                  app_name: Optional[str] = None) -> None:
    """
    Configures logging for the application with both console and file handlers.
    
    Args:
        logs_dir: Directory to store log files. Will be created if it doesn't exist.
        log_level: Logging level (e.g., logging.DEBUG, logging.INFO)
        app_name: Name of the app for log file naming. If None, uses 'app'.
    """
    global _handlers_to_cleanup
    
    # Ensure logs directory exists
    os.makedirs(logs_dir, exist_ok=True)

    # Set up root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Remove existing handlers to avoid duplicates when called multiple times
    for handler in root_logger.handlers[:]:
        try:
            handler.flush()
            handler.close()
            root_logger.removeHandler(handler)
        except Exception:
            pass  # Ignore errors when removing handlers
    
    # Configure formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler for immediate feedback
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    _handlers_to_cleanup.append(console_handler)
    
    # File handler for persistent logs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    app_name = app_name or "app"
    log_filename = f"{app_name}_log_{timestamp}.log"
    log_path = os.path.join(logs_dir, log_filename)
    
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    _handlers_to_cleanup.append(file_handler)
    
    logging.info(f"Logging configured: console and file ({log_path})")
    
    # Register shutdown function to properly close handlers
    atexit.register(shutdown_logging)

def get_logger(name: str, log_level: Optional[int] = None) -> logging.Logger:
    """
    Gets a logger with the specified name and optional log level.
    
    Args:
        name: Logger name
        log_level: Optional log level override
        
    Returns:
        A configured logger
    """
    logger = logging.getLogger(name)
    if log_level is not None:
        logger.setLevel(log_level)
    return logger

def shutdown_logging() -> None:
    """
    Safely shuts down all logging handlers to prevent I/O errors during cleanup.
    """
    global _handlers_to_cleanup
    
    # First disable all logging temporarily
    logging.disable(logging.CRITICAL)
    
    root_logger = logging.getLogger()
    
    # Close each tracked handler properly
    for handler in _handlers_to_cleanup:
        try:
            # First remove the handler from the logger to prevent new log entries
            if handler in root_logger.handlers:
                root_logger.removeHandler(handler)
            
            # Then flush and close properly
            handler.flush()
            handler.close()
        except Exception:
            # Ignore any errors during shutdown
            pass
    
    # Clear the tracked handlers list
    _handlers_to_cleanup.clear()
    
    # Re-enable logging (though at this point, all handlers are gone)
    logging.disable(logging.NOTSET) 