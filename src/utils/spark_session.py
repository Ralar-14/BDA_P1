import os
import logging
from pyspark.sql import SparkSession

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_spark_session(app_name="Spotify_ETL"):
    """
    Creates and returns a Spark session.
    
    Args:
        app_name (str): The name of the Spark application.
        
    Returns:
        SparkSession: A configured Spark session.
    """
    logger.info(f"Creating Spark session with app name: {app_name}")
    
    try:
        # Create a SparkSession with appropriate settings
        spark = (SparkSession.builder
                .appName(app_name)
                .config("spark.sql.execution.arrow.pyspark.enabled", "true")
                .config("spark.sql.shuffle.partitions", "10")
                .config("spark.driver.memory", "2g")
                .config("spark.executor.memory", "4g")
                .config("spark.default.parallelism", "4")
                .config("spark.sql.adaptive.enabled", "true")
                .getOrCreate())
        
        # Set log level to ERROR to reduce verbosity
        spark.sparkContext.setLogLevel("ERROR")
        
        logger.info("Spark session created successfully")
        return spark
    
    except Exception as e:
        logger.error(f"Error creating Spark session: {e}")
        raise