# src/utils/logger.py

import logging
import os
from datetime import datetime

def setup_logger():
    """Setup logging dengan file dan console output"""
    
    # Create logs directory if not exists
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Log filename dengan timestamp
    log_file = os.path.join(log_dir, f"sisah_{datetime.now().strftime('%Y%m%d')}.log")
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    # Set debug level for specific modules if DEBUG_MODE is on
    if os.getenv("DEBUG_MODE", "False").lower() == "true":
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger = logging.getLogger(__name__)
    logger.info("=" * 50)
    logger.info(f"Logger initialized - Log file: {log_file}")
    logger.info("=" * 50)
    
    return logger