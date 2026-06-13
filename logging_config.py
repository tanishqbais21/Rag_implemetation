import logging
import json
import os
from logging.handlers import RotatingFileHandler
from datetime import datetime

class AILogger:
    def __init__(self, name: str = "AI_Project", log_dir: str = "logs"):
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        
        # Formatter: Timestamp - Name - Level - Message
        formatter = logging.Formatter('%(asctime)s | %(name)s | %(levelname)s | %(message)s')
        
        # Console Handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # Rotating File Handler (1MB limit per file, keep 3 backups)
        file_handler = RotatingFileHandler(
            f"{log_dir}/{name}.log", maxBytes=1*1024*1024, backupCount=3
        )
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    def info(self, msg: str, data: dict = None):
        if data:
            msg = f"{msg} | Data: {json.dumps(data)}"
        self.logger.info(msg)

    def error(self, msg: str, exc_info=True):
        self.logger.error(msg, exc_info=exc_info)