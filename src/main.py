import os
import sys
import logging
import importlib.util
from utils.spark_session import create_spark_session
from utils.file_utils import ensure_directories, get_file_paths, LANDING_ZONE, FORMATTED_ZONE, TRUSTED_ZONE, ANALYTICS_ZONE
from jobs.data_processing import process_spotify_tracks, process_top_songs, process_song_lyrics
from jobs.data_analysis import analyze_top_artists, analyze_song_popularity_by_country, analyze_audio_features_correlation
from jobs.data_visualization import visualize_top_artists, visualize_country_popularity, visualize_audio_features_correlation

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def import_landing_script():
    """
    Imports the landing.py script from utils directory.
    """
    try:
        # Import the landing module directly from utils
        from utils.landing import download_and_store_datasets, setup_kaggle_credentials
        
        logger.info("Successfully imported landing module")
        return {
            "download_and_store_datasets": download_and_store_datasets,
            "setup_kaggle_credentials": setup_kaggle_credentials
        }
    except Exception as e:
        logger.error(f"Failed to import landing module: {e}")
        return None

def ensure_data_downloaded():
    """
    Ensures that the necessary data is downloaded from Kaggle.
    """
    # Check if landing_zone exists and has CSV files
    if not os.path.exists(LANDING_ZONE) or not [f for f in os.listdir(LANDING_ZONE) if f.endswith('.csv')]:
        logger.info("No data found in landing zone. Downloading from Kaggle...")
        # Import landing script and download data
        landing = import_landing_script()
        if landing:
            landing["download_and_store_datasets"](update=False)
        else:
            logger.error("Could not download data. Exiting...")
            sys.exit(1)
    else:
        logger.info("Data already exists in landing zone")

def run_etl_pipeline():
    """
    Runs the complete ETL pipeline for Spotify data analysis.
    """
    logger.info("Starting Spotify Data Analysis ETL Pipeline")
    
    # Ensure the necessary data is downloaded
    ensure_data_downloaded()
    
    # Create Spark session
    logger.info("Creating Spark session")
    spark = create_spark_session("Spotify_Data_Analysis")
    
    # Ensure all directories exist
    ensure_directories()
    
    # Process data - Extract from landing zone, transform, and load to formatted zone
    logger.info("Starting data processing phase")
    
    try:
        # Get all CSV files from landing zone
        csv_files = get_file_paths(LANDING_ZONE, ".csv")
        
        # Process each dataset
        formatted_paths = {}
        
        for csv_file in csv_files:
            file_name = os.path.basename(csv_file)
            output_path = os.path.join(FORMATTED_ZONE, file_name.replace('.csv', ''))
            
            if "spotify-tracks-dataset" in file_name:
                process_spotify_tracks(spark, csv_file, output_path)
                formatted_paths["tracks"] = output_path
            elif "top-spotify-songs-by-country" in file_name:
                process_top_songs(spark, csv_file, output_path)
                formatted_paths["top_songs"] = output_path
            elif "songs-lyrics" in file_name:
                process_song_lyrics(spark, csv_file, output_path)
                formatted_paths["lyrics"] = output_path
            else:
                logger.warning(f"Unknown dataset: {file_name}. Skipping.")
        
        logger.info("Data processing phase completed")
        
        # Data analysis - Generate insights and save to trusted zone
        logger.info("Starting data analysis phase")
        
        # Create trusted zone directory
        os.makedirs(TRUSTED_ZONE, exist_ok=True)
        
        # Perform analyses
        analysis_paths = {}
        
        if "tracks" in formatted_paths:
            top_artists_path = os.path.join(TRUSTED_ZONE, "top_artists")
            analyze_top_artists(spark, formatted_paths["tracks"], top_artists_path)
            analysis_paths["top_artists"] = top_artists_path
            
            feature_correlations_path = os.path.join(TRUSTED_ZONE, "feature_correlations")
            analyze_audio_features_correlation(spark, formatted_paths["tracks"], feature_correlations_path)
            analysis_paths["feature_correlations"] = feature_correlations_path
        
        if "top_songs" in formatted_paths:
            country_popularity_path = os.path.join(TRUSTED_ZONE, "country_popularity")
            analyze_song_popularity_by_country(spark, formatted_paths["top_songs"], country_popularity_path)
            analysis_paths["country_popularity"] = country_popularity_path
        
        logger.info("Data analysis phase completed")
        
        # Data visualization - Create visualizations and save to analytics zone
        logger.info("Starting data visualization phase")
        
        # Create analytics directory
        os.makedirs(ANALYTICS_ZONE, exist_ok=True)
        
        # Create visualizations
        if "top_artists" in analysis_paths:
            visualize_top_artists(spark, analysis_paths["top_artists"], ANALYTICS_ZONE)
        
        if "country_popularity" in analysis_paths:
            visualize_country_popularity(spark, analysis_paths["country_popularity"], ANALYTICS_ZONE)
        
        if "feature_correlations" in analysis_paths:
            visualize_audio_features_correlation(spark, analysis_paths["feature_correlations"], ANALYTICS_ZONE)
        
        logger.info("Data visualization phase completed")
        
        logger.info("ETL Pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"ETL Pipeline failed: {e}")
        raise
    finally:
        # Stop Spark session
        spark.stop()
        logger.info("Spark session stopped")

if __name__ == "__main__":
    run_etl_pipeline()