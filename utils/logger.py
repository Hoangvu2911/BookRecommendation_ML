#!/usr/bin/env python3
"""
Logging configuration for Book Recommendation System
"""

import logging
import sys
from config import LOGGING_CONFIG

def setup_logger(name=__name__):
    """
    Setup logger with both file and console handlers
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Don't add handlers if already configured
    if logger.handlers:
        return logger
    
    logger.setLevel(LOGGING_CONFIG['level'])
    
    # Create formatters
    formatter = logging.Formatter(LOGGING_CONFIG['format'])
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(LOGGING_CONFIG['level'])
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    try:
        file_handler = logging.FileHandler(LOGGING_CONFIG['log_file'])
        file_handler.setLevel(LOGGING_CONFIG['level'])
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        logger.warning(f"Could not setup file logging: {e}")
    
    return logger
