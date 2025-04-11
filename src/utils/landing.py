import os
import datetime
import logging
import kaggle
import shutil
from .file_utils import LANDING_ZONE

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# List of datasets to ingest from Kaggle
datasets = [
    {
        "kaggle_id": "asaniczka/top-spotify-songs-in-73-countries-daily-updated",
        "dataset_name": "top-spotify-songs-by-country",
        "update": True
    },
    {
        "kaggle_id": "maharshipandya/-spotify-tracks-dataset",
        "dataset_name": "spotify-tracks-dataset",
        "update": False
    },
    {
        "kaggle_id": "terminate9298/songs-lyrics",
        "dataset_name": "songs-lyrics",
        "update": False
    }
]

def log(message):
    """Logging function to timestamp each message"""
    logger.info(message)

def setup_kaggle_credentials():
    """
    Sets up Kaggle API credentials if not already set.
    User can provide credentials in two ways:
    1. As environment variables KAGGLE_USERNAME and KAGGLE_KEY
    2. By placing a kaggle.json file in ~/.kaggle/
    """
    try:
        # Check if credentials exist
        kaggle_creds_file = os.path.join(os.path.expanduser('~'), '.kaggle', 'kaggle.json')
        
        if not (os.environ.get('KAGGLE_USERNAME') and os.environ.get('KAGGLE_KEY')) and not os.path.exists(kaggle_creds_file):
            logger.warning("Kaggle API credentials not found. Please ensure they are set up.")
            return False
        
        # Test authentication
        kaggle.api.authenticate()
        log("Kaggle API authentication successful")
        return True
        
    except Exception as e:
        logger.error(f"Error setting up Kaggle credentials: {e}")
        return False

def data_collector_kaggle(kaggle_dataset):
    """
    Downloads a dataset from Kaggle and saves it to the landing zone.

    Parameters:
    kaggle_dataset (dict): A dictionary containing the Kaggle dataset information.
    """
    # Extract dataset information
    kaggle_id = kaggle_dataset["kaggle_id"]
    dataset_name = kaggle_dataset["dataset_name"]

    # Create a temporary directory for the dataset
    dataset_folder = os.path.join(LANDING_ZONE, "temp_dataset")
    os.makedirs(dataset_folder, exist_ok=True)

    try:  
        log(f"Downloading dataset: {kaggle_id}")

        kaggle.api.dataset_download_files(
            kaggle_id,
            path=dataset_folder,
            unzip=True
        )

        # Search for CSV files in the downloaded data
        for filename in os.listdir(dataset_folder):
            if filename.endswith(".csv"):
                # Rename the file to the dataset name
                new_filename = f"{dataset_name}.csv"
                old_path = os.path.join(dataset_folder, filename)
                new_path = os.path.join(dataset_folder, new_filename)
                os.rename(old_path, new_path)

                # Move the file to the landing zone
                final_path = os.path.join(LANDING_ZONE, new_filename)
                shutil.move(new_path, final_path)
                log(f"Moved file to {final_path}")
            else:
                log(f"File {filename} is not a CSV file. Skipping.")
            
        # Remove the temporary dataset folder after processing
        shutil.rmtree(dataset_folder)
        return True

    except Exception as e:
        # Remove the dataset folder if it exists
        if os.path.exists(dataset_folder):
            shutil.rmtree(dataset_folder)

        # Log the error
        logger.error(f"Error downloading dataset '{kaggle_id}': {e}")
        return False

def download_and_store_datasets(update=False):
    """
    Downloads and stores datasets from Kaggle into the landing zone.
    
    Args:
        update (bool): If True, only downloads datasets marked for update.
    
    Returns:
        bool: True if all datasets were processed successfully, False otherwise.
    """
    log("Starting the creation of the Landing Zone using Kaggle API")
    
    # Check Kaggle credentials
    if not setup_kaggle_credentials():
        return False
    
    success = True
    for kaggle_dataset in datasets:
        if update and not kaggle_dataset["update"]:
            log(f"Skipping dataset '{kaggle_dataset['dataset_name']}' as update is set to False.")
            continue
        
        try:
            dataset_name = kaggle_dataset["dataset_name"]
            result = data_collector_kaggle(kaggle_dataset)
            if result:
                log(f"Dataset '{dataset_name}' processed successfully.")
            else:
                log(f"Failed to process dataset '{dataset_name}'.")
                success = False
        except Exception as e:
            logger.error(f"Error processing dataset '{kaggle_dataset['dataset_name']}': {e}")
            success = False

    if success:
        log("All datasets have been processed successfully.")
    else:
        log("Some datasets failed to process.")
    
    log("Landing Zone creation completed.")
    return success

if __name__ == '__main__':
    # This allows the script to be run directly for testing
    download_and_store_datasets(update=False)