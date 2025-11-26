import logging
import os
from logging.handlers import RotatingFileHandler
from src.config import Config


def setup_logger(name: str = __name__) -> logging.Logger:
    logger = logging.getLogger(name)
    log_level = getattr(logging, Config.LOG_LEVEL.upper(), logging.INFO)
    logger.setLevel(log_level)
    
    os.makedirs(os.path.dirname(Config.LOG_FILE), exist_ok=True)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    file_handler = RotatingFileHandler(
        Config.LOG_FILE,
        maxBytes=10485760,
        backupCount=5
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger


logger = setup_logger()
