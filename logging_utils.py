import logging
import sys

def setup_logging():
    """
    Configures a root logger to print to stdout with a consistent format.
    """
    # Get the root logger
    logger = logging.getLogger()
    
    # Clear any existing handlers to avoid duplicate logs
    if logger.hasHandlers():
        logger.handlers.clear()
        
    # Set the logging level (e.g., INFO, DEBUG)
    logger.setLevel(logging.INFO)
    
    # Create a handler to write to standard output
    handler = logging.StreamHandler(sys.stdout)
    
    # Create a formatter and set it for the handler
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler.setFormatter(formatter)
    
    # Add the handler to the logger
    logger.addHandler(handler)
    
    return logger

# You can create a default logger instance to be imported by other modules
log = setup_logging()
