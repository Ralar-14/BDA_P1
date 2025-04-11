import os
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def visualize_top_artists(spark: SparkSession, top_artists_path: str, output_path: str, limit: int = 20):
    """
    Create a visualization of top artists by popularity.
    
    Args:
        spark (SparkSession): The Spark session
        top_artists_path (str): Path to the top artists data
        output_path (str): Path to save the visualization
        limit (int): Number of top artists to include
    """
    logger.info(f"Creating visualization of top {limit} artists by popularity")
    
    try:
        # Read the top artists data
        top_artists_df = spark.read.parquet(top_artists_path)
        
        # Convert to pandas for visualization
        top_artists_pd = top_artists_df.limit(limit).toPandas()
        
        # Create the visualization
        plt.figure(figsize=(12, 8))
        sns.barplot(x='avg_popularity', y='artists', data=top_artists_pd)
        plt.title(f'Top {limit} Artists by Average Track Popularity')
        plt.xlabel('Average Popularity')
        plt.ylabel('Artist')
        plt.tight_layout()
        
        # Save the visualization
        os.makedirs(output_path, exist_ok=True)
        plt.savefig(os.path.join(output_path, 'top_artists.png'))
        logger.info(f"Successfully saved top artists visualization to {output_path}/top_artists.png")
        
        plt.close()
        
    except Exception as e:
        logger.error(f"Error creating top artists visualization: {e}")
        raise

def visualize_country_popularity(spark: SparkSession, country_popularity_path: str, output_path: str, limit: int = 15):
    """
    Create a visualization of song popularity by country.
    
    Args:
        spark (SparkSession): The Spark session
        country_popularity_path (str): Path to the country popularity data
        output_path (str): Path to save the visualization
        limit (int): Number of top countries to include
    """
    logger.info(f"Creating visualization of song popularity for top {limit} countries")
    
    try:
        # Read the country popularity data
        country_popularity_df = spark.read.parquet(country_popularity_path)
        
        # Convert to pandas for visualization
        country_popularity_pd = country_popularity_df.limit(limit).toPandas()
        
        # Create the visualization
        plt.figure(figsize=(12, 8))
        sns.barplot(x='total_streams', y='country', data=country_popularity_pd)
        plt.title(f'Top {limit} Countries by Total Streams')
        plt.xlabel('Total Streams')
        plt.ylabel('Country')
        plt.tight_layout()
        
        # Save the visualization
        os.makedirs(output_path, exist_ok=True)
        plt.savefig(os.path.join(output_path, 'country_popularity.png'))
        logger.info(f"Successfully saved country popularity visualization to {output_path}/country_popularity.png")
        
        plt.close()
        
    except Exception as e:
        logger.error(f"Error creating country popularity visualization: {e}")
        raise

def visualize_audio_features_correlation(spark: SparkSession, feature_correlations_path: str, output_path: str):
    """
    Create a visualization of audio feature correlations with popularity.
    
    Args:
        spark (SparkSession): The Spark session
        feature_correlations_path (str): Path to the feature correlations data
        output_path (str): Path to save the visualization
    """
    logger.info("Creating visualization of audio feature correlations with popularity")
    
    try:
        # Read the feature correlations data
        correlations_df = spark.read.parquet(feature_correlations_path)
        
        # Convert to pandas for visualization
        correlations_pd = correlations_df.toPandas()
        
        # Sort by absolute correlation value
        correlations_pd['abs_correlation'] = correlations_pd['correlation_with_popularity'].abs()
        correlations_pd = correlations_pd.sort_values('abs_correlation', ascending=False)
        
        # Create the visualization
        plt.figure(figsize=(10, 8))
        sns.barplot(x='correlation_with_popularity', y='feature', data=correlations_pd)
        plt.title('Audio Feature Correlations with Track Popularity')
        plt.xlabel('Correlation Coefficient')
        plt.ylabel('Audio Feature')
        plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        plt.tight_layout()
        
        # Save the visualization
        os.makedirs(output_path, exist_ok=True)
        plt.savefig(os.path.join(output_path, 'audio_feature_correlations.png'))
        logger.info(f"Successfully saved audio feature correlations visualization to {output_path}/audio_feature_correlations.png")
        
        plt.close()
        
    except Exception as e:
        logger.error(f"Error creating audio feature correlations visualization: {e}")
        raise