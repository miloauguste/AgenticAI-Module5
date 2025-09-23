"""
Colored logging utility for competitive analysis application
Ensures all errors are logged to stdout with colors for better visibility
"""
import sys
import logging
from typing import Optional

class ColoredFormatter(logging.Formatter):
    """Custom formatter to add colors to log messages"""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
        'RESET': '\033[0m'       # Reset
    }
    
    def format(self, record):
        # Get the color for the log level
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset = self.COLORS['RESET']
        
        # Format the timestamp
        timestamp = self.formatTime(record, '%Y-%m-%d %H:%M:%S')
        
        # Create colored log message
        colored_message = (
            f"{color}[{timestamp}] "
            f"{record.levelname:8} "
            f"{record.name}: "
            f"{record.getMessage()}{reset}"
        )
        
        # Add exception info if present
        if record.exc_info:
            colored_message += f"\n{color}{self.formatException(record.exc_info)}{reset}"
        
        return colored_message

def setup_colored_logging(
    level: int = logging.INFO,
    logger_name: Optional[str] = None,
    force_stdout: bool = True
) -> logging.Logger:
    """
    Set up colored logging to stdout
    
    Args:
        level: Logging level (default: INFO)
        logger_name: Name of the logger (default: root logger)
        force_stdout: Force all logs to stdout instead of stderr
    
    Returns:
        Configured logger
    """
    # Get logger
    logger = logging.getLogger(logger_name)
    
    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Create console handler (force to stdout for all errors)
    console_handler = logging.StreamHandler(sys.stdout if force_stdout else sys.stderr)
    console_handler.setLevel(level)
    
    # Create colored formatter
    formatter = ColoredFormatter()
    console_handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(console_handler)
    logger.setLevel(level)
    
    # Prevent propagation to parent loggers to avoid duplicate messages
    logger.propagate = False
    
    return logger

def log_error(message: str, exception: Optional[Exception] = None, logger_name: str = "ERROR"):
    """
    Log an error message with red color to stdout
    
    Args:
        message: Error message to log
        exception: Optional exception to include
        logger_name: Name for the logger
    """
    logger = setup_colored_logging(level=logging.ERROR, logger_name=logger_name)
    if exception:
        logger.error(f"{message}: {exception}", exc_info=True)
    else:
        logger.error(message)

def log_warning(message: str, logger_name: str = "WARNING"):
    """
    Log a warning message with yellow color to stdout
    
    Args:
        message: Warning message to log
        logger_name: Name for the logger
    """
    logger = setup_colored_logging(level=logging.WARNING, logger_name=logger_name)
    logger.warning(message)

def log_info(message: str, logger_name: str = "INFO"):
    """
    Log an info message with green color to stdout
    
    Args:
        message: Info message to log
        logger_name: Name for the logger
    """
    logger = setup_colored_logging(level=logging.INFO, logger_name=logger_name)
    logger.info(message)

def log_success(message: str, logger_name: str = "SUCCESS"):
    """
    Log a success message with bright green color to stdout
    
    Args:
        message: Success message to log
        logger_name: Name for the logger
    """
    logger = setup_colored_logging(level=logging.INFO, logger_name=logger_name)
    # Use info level but with SUCCESS prefix
    logger.info(f"[SUCCESS] {message}")

def log_critical(message: str, exception: Optional[Exception] = None, logger_name: str = "CRITICAL"):
    """
    Log a critical error message with magenta color to stdout
    
    Args:
        message: Critical error message to log
        exception: Optional exception to include
        logger_name: Name for the logger
    """
    logger = setup_colored_logging(level=logging.CRITICAL, logger_name=logger_name)
    if exception:
        logger.critical(f"{message}: {exception}", exc_info=True)
    else:
        logger.critical(message)

def configure_application_logging():
    """
    Configure logging for the entire application
    Sets up colored logging for all modules
    """
    # Configure root logger
    root_logger = setup_colored_logging(level=logging.INFO)
    
    # Configure specific module loggers
    modules = [
        'competitive_agent',
        'vector_store', 
        'react_agent',
        'data_processor',
        'query_analyzer',
        'cli_interface',
        'streamlit_app'
    ]
    
    for module in modules:
        module_logger = setup_colored_logging(level=logging.INFO, logger_name=module)
        
    # Also configure external library loggers to reduce noise but show errors
    external_loggers = ['httpx', 'urllib3', 'requests']
    for ext_logger in external_loggers:
        ext_log = setup_colored_logging(level=logging.WARNING, logger_name=ext_logger)

def print_colored_error(message: str, include_timestamp: bool = True):
    """
    Print an error message directly to stdout with color (non-logging version)
    
    Args:
        message: Error message to print
        include_timestamp: Whether to include timestamp
    """
    RED = '\033[31m'
    RESET = '\033[0m'
    
    if include_timestamp:
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"{RED}[{timestamp}] ERROR: {message}{RESET}", file=sys.stdout, flush=True)
    else:
        print(f"{RED}ERROR: {message}{RESET}", file=sys.stdout, flush=True)

def print_colored_warning(message: str, include_timestamp: bool = True):
    """
    Print a warning message directly to stdout with color (non-logging version)
    
    Args:
        message: Warning message to print
        include_timestamp: Whether to include timestamp
    """
    YELLOW = '\033[33m'
    RESET = '\033[0m'
    
    if include_timestamp:
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"{YELLOW}[{timestamp}] WARNING: {message}{RESET}", file=sys.stdout, flush=True)
    else:
        print(f"{YELLOW}WARNING: {message}{RESET}", file=sys.stdout, flush=True)

def print_colored_success(message: str, include_timestamp: bool = True):
    """
    Print a success message directly to stdout with color (non-logging version)
    
    Args:
        message: Success message to print
        include_timestamp: Whether to include timestamp
    """
    GREEN = '\033[32m'
    RESET = '\033[0m'
    
    if include_timestamp:
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"{GREEN}[{timestamp}] SUCCESS: {message}{RESET}", file=sys.stdout, flush=True)
    else:
        print(f"{GREEN}SUCCESS: {message}{RESET}", file=sys.stdout, flush=True)