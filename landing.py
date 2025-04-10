import os
import datetime
import kaggle
import shutil

# Logging function to timestamp each message
def log(message):
    print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}")

# Base directory and Landing Zone directory structure
BASE_DIR = "./data"
LANDING_ZONE_DIR = os.path.join(BASE_DIR, "landing_zone")
os.makedirs(LANDING_ZONE_DIR, exist_ok=True)

kaggle.api.authenticate()

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

def data_collector_kaggle(kaggle_dataset: dict) -> None:
    """
    Downloads a dataset from Kaggle and saves it to the landing zone.

    Parameters:
    kaggle_dataset (dict): A dictionary containing the Kaggle dataset information.
    """

    # Excract dataset information
    kaggle_id = kaggle_dataset["kaggle_id"]
    dataset_name = kaggle_dataset["dataset_name"]

    # Create a directory for the dataset (to change the dataset name)
    dataset_folder = os.path.join(LANDING_ZONE_DIR, "dataset_name")
    os.makedirs(dataset_folder, exist_ok=True)

    try:  
        log(f"Downloading dataset: {kaggle_id}")

        kaggle.api.dataset_download_files(
            kaggle_id,
            path=dataset_folder,
            unzip=True
        )

        # Search the CSV downloaded file 
        for filename in os.listdir(dataset_folder):
            if filename.endswith(".csv"):
                # Rename the file to the dataset name
                new_filename = f"{dataset_name}.csv"
                old_path = os.path.join(dataset_folder, filename)
                new_path = os.path.join(dataset_folder, new_filename)
                os.rename(old_path, new_path)

                # Move the file to the landing zone
                final_path = os.path.join(LANDING_ZONE_DIR, new_filename)
                shutil.move(new_path, final_path)
            else:
                log(f"File downloaded is not a CSV file. Skipping.")
            
        # Remove the dataset folder after processing
        shutil.rmtree(dataset_folder)

    except Exception as e:
         # Remove the dataset folder if it exists
        if os.path.exists(dataset_folder):
            shutil.rmtree(dataset_folder)

        # Log the error
        log(f"Error downloading dataset '{kaggle_id}': {e}")
       
        return

    # Log the successful download
    log(f"Dataset '{dataset_name}' downloaded successfully.")

def download_and_store_datasets(update: bool = False) -> None:
    """
    Downloads and stores datasets from Kaggle into the landing zone.
    """
    log("Starting the creation of the Landing Zone using Kaggle API")

    for kaggle_dataset in datasets:
        if update and not kaggle_dataset["update"]:
            log(f"Skipping dataset '{kaggle_dataset['dataset_name']}' as update is set to False.")
            continue
        try:
            dataset_name = kaggle_dataset["dataset_name"]
            data_collector_kaggle(kaggle_dataset)
            log(f"Dataset '{dataset_name}' processed successfully.")
        except Exception as e:
            log(f"Error processing dataset '{dataset_name}': {e}")

    log("All datasets have been processed.")
    log("Landing Zone creation completed.")

if __name__ == '__main__':

    # Download and store datasets for the first time (update=False)
    download_and_store_datasets(update=False)

    # Update datasets (update=True)
    # download_and_store_datasets(update=True)
