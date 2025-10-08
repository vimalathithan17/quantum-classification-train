import logging
import sys
import atexit
from logging.handlers import QueueHandler, QueueListener
from queue import Queue

def setup_logging():
    """
    Configures a root logger to be asynchronous using a QueueHandler.
    This prevents logging I/O from blocking the main application thread.
    """
    # Get the root logger
    logger = logging.getLogger()
    
    # Clear any existing handlers to avoid duplicate logs
    if logger.hasHandlers():
        logger.handlers.clear()
        
    # Set the logging level (e.g., INFO, DEBUG)
    logger.setLevel(logging.INFO)
    
    # Create a queue for log records
    log_queue = Queue(-1)  # A queue of infinite size

    # Create a handler that writes to standard output
    # This will be the handler used by the listener
    stream_handler = logging.StreamHandler(sys.stdout)
    
    # Create a formatter and set it for the handler
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    stream_handler.setFormatter(formatter)
    
    # The QueueHandler is what the root logger will use. It's very fast.
    queue_handler = QueueHandler(log_queue)
    logger.addHandler(queue_handler)
    
    # The QueueListener pulls from the queue and directs to the real handler.
    # It runs in a background thread.
    listener = QueueListener(log_queue, stream_handler)
    listener.start()
    
    # Ensure the listener is stopped when the application exits
    atexit.register(listener.stop)
    
    return logger

# You can create a default logger instance to be imported by other modules
log = setup_logging()
