# core/utils.py
import logging

logger = logging.getLogger(__name__)

def setup_global_logging(level_str: str = "INFO"):
    """
    Configures global logging. Can be called from app.py.
    Alternative to basicConfig in settings.py if more control is needed.
    """
    level = getattr(logging, level_str.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler()
            # You could add a FileHandler here as well
        ]
    )
    logger.info(f"Global logging configured to level: {level_str}")

# Add other utility functions here as needed, e.g., for text cleaning, unique ID generation, etc.
