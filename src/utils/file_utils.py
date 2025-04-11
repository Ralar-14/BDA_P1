import os
import glob
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define base directories for data lake zones
BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data")

# Data lake zones
LANDING_ZONE = os.path.join(BASE_DIR, "landing_zone")
FORMATTED_ZONE = os.path.join(BASE_DIR, "formatted_zone")
TRUSTED_ZONE = os.path.join(BASE_DIR, "trusted_zone")
ANALYTICS_ZONE = os.path.join(BASE_DIR, "analytics_zone")

def ensure_directories():
    """
    Ensures that all necessary directories for the data lake exist.
    """
    try:
        for directory in [BASE_DIR, LANDING_ZONE, FORMATTED_ZONE, TRUSTED_ZONE, ANALYTICS_ZONE]:
            if not os.path.exists(directory):
                os.makedirs(directory)
                logger.info(f"Created directory: {directory}")
            else:
                logger.info(f"Directory already exists: {directory}")
    except Exception as e:
        logger.error(f"Error creating directories: {e}")
        raise

def get_file_paths(directory, file_extension):
    """
    Gets all file paths in a directory with the specified file extension.
    
    Args:
        directory (str): The directory to search.
        file_extension (str): The file extension to match (e.g., ".csv").
        
    Returns:
        list: A list of file paths.
    """
    try:
        # Make sure the directory exists
        if not os.path.exists(directory):
            logger.warning(f"Directory does not exist: {directory}")
            return []
        
        # Get all files with the specified extension
        pattern = os.path.join(directory, f"*{file_extension}")
        file_paths = glob.glob(pattern)
        
        if not file_paths:
            logger.warning(f"No {file_extension} files found in {directory}")
        else:
            logger.info(f"Found {len(file_paths)} {file_extension} files in {directory}")
            
        return file_paths
    except Exception as e:
        logger.error(f"Error getting file paths: {e}")
        return []

def clean_directory(directory):
    """
    Removes all files in a directory.
    
    Args:
        directory (str): The directory to clean.
    """
    try:
        if os.path.exists(directory):
            for file_path in glob.glob(os.path.join(directory, "*")):
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    logger.info(f"Removed file: {file_path}")
            logger.info(f"Directory cleaned: {directory}")
        else:
            logger.warning(f"Directory does not exist: {directory}")
    except Exception as e:
        logger.error(f"Error cleaning directory {directory}: {e}")
        raise