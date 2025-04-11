import os
import logging
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, when, regexp_replace, trim, lower, upper, to_date, year, month, dayofmonth, explode, split

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def process_spotify_tracks(spark, input_path, output_path):
    """
    Process the Spotify tracks dataset.
    
    Args:
        spark (SparkSession): The Spark session.
        input_path (str): The input file path.
        output_path (str): The output directory path.
    """
    logger.info(f"Processing Spotify tracks dataset from {input_path}")
    
    try:
        # Read the CSV file
        df = spark.read.option("header", "true").option("inferSchema", "true").csv(input_path)
        
        # Print schema and count before processing
        logger.info("Original schema:")
        df.printSchema()
        count_before = df.count()
        logger.info(f"Count before processing: {count_before}")
        
        # Clean and transform the data
        processed_df = df.select(
            col("track_id").alias("track_id"),
            col("track_name").alias("track_name"),
            col("artists").alias("artist_name"),
            col("album_name"),
            col("popularity").cast("integer").alias("popularity"),
            col("duration_ms").cast("long").alias("duration_ms"),
            col("explicit").cast("boolean").alias("explicit"),
            col("danceability").cast("double").alias("danceability"),
            col("energy").cast("double").alias("energy"),
            col("key").cast("integer").alias("key"),
            col("loudness").cast("double").alias("loudness"),
            col("mode").cast("integer").alias("mode"),
            col("speechiness").cast("double").alias("speechiness"),
            col("acousticness").cast("double").alias("acousticness"),
            col("instrumentalness").cast("double").alias("instrumentalness"),
            col("liveness").cast("double").alias("liveness"),
            col("valence").cast("double").alias("valence"),
            col("tempo").cast("double").alias("tempo")
        )
        
        # Remove rows with null track_id or track_name
        processed_df = processed_df.filter(
            col("track_id").isNotNull() & 
            col("track_name").isNotNull()
        )
        
        # Print schema and count after processing
        logger.info("Processed schema:")
        processed_df.printSchema()
        count_after = processed_df.count()
        logger.info(f"Count after processing: {count_after}")
        logger.info(f"Removed {count_before - count_after} rows during processing")
        
        # Write the processed data as Parquet
        processed_df.write.mode("overwrite").parquet(output_path)
        logger.info(f"Processed Spotify tracks data saved to {output_path}")
        
        return output_path
        
    except Exception as e:
        logger.error(f"Error processing Spotify tracks dataset: {e}")
        raise

def process_top_songs(spark, input_path, output_path):
    """
    Procesa el dataset de Spotify con el siguiente schema:
    
    root
     |-- spotify_id: string (nullable = true)
     |-- name: string (nullable = true)
     |-- artists: string (nullable = true)
     |-- daily_rank: string (nullable = true)
     |-- daily_movement: string (nullable = true)
     |-- weekly_movement: string (nullable = true)
     |-- country: string (nullable = true)
     |-- snapshot_date: string (nullable = true)
     |-- popularity: string (nullable = true)
     |-- is_explicit: string (nullable = true)
     |-- duration_ms: string (nullable = true)
     |-- album_name: string (nullable = true)
     |-- album_release_date: string (nullable = true)
     |-- danceability: string (nullable = true)
     |-- energy: string (nullable = true)
     |-- key: string (nullable = true)
     |-- loudness: string (nullable = true)
     |-- mode: string (nullable = true)
     |-- speechiness: string (nullable = true)
     |-- acousticness: double (nullable = true)
     |-- instrumentalness: double (nullable = true)
     |-- liveness: double (nullable = true)
     |-- valence: double (nullable = true)
     |-- tempo: double (nullable = true)
     |-- time_signature: double (nullable = true)
    
    Args:
        spark (SparkSession): La sesión de Spark.
        input_path (str): Ruta del archivo CSV de entrada.
        output_path (str): Ruta del directorio de salida en HDFS.
    """
    logger.info(f"Procesando datos de Spotify desde {input_path}")
    
    try:
        # Leer el archivo CSV
        df = spark.read.option("header", "true").option("inferSchema", "true").csv(input_path)
        
        # Imprimir schema y contar filas antes del procesamiento
        logger.info("Schema original:")
        df.printSchema()
        count_before = df.count()
        logger.info(f"Filas antes del procesamiento: {count_before}")
        
        # Seleccionar y transformar las columnas según el nuevo schema
        processed_df = df.select(
            col("spotify_id").alias("spotify_id"),
            col("name").alias("track_name"),        
            col("artists").alias("artist_name"),
            col("daily_rank").cast("integer").alias("daily_rank"),
            col("daily_movement").alias("daily_movement"),
            col("weekly_movement").alias("weekly_movement"),
            col("country").alias("country"),
            col("snapshot_date").alias("snapshot_date"),
            col("popularity").cast("integer").alias("popularity"),
            col("is_explicit").alias("is_explicit"),
            col("duration_ms").cast("long").alias("duration_ms"),
            col("album_name").alias("album_name"),
            col("album_release_date").alias("album_release_date"),
            col("danceability").alias("danceability"),
            col("energy").alias("energy"),
            col("key").alias("key"),
            col("loudness").alias("loudness"),
            col("mode").alias("mode"),
            col("speechiness").alias("speechiness"),
            col("acousticness").cast("double").alias("acousticness"),
            col("instrumentalness").cast("double").alias("instrumentalness"),
            col("liveness").cast("double").alias("liveness"),
            col("valence").cast("double").alias("valence"),
            col("tempo").cast("double").alias("tempo"),
            col("time_signature").cast("double").alias("time_signature")
        )
        
        # Manejo de valores nulos para algunas columnas numéricas
        processed_df = processed_df.na.fill({
            "daily_rank": 0,
            "popularity": 0,
            "duration_ms": 0,
            "acousticness": 0.0,
            "instrumentalness": 0.0,
            "liveness": 0.0,
            "valence": 0.0,
            "tempo": 0.0,
            "time_signature": 0.0
        })
        
        # Convertir snapshot_date y album_release_date a formato fecha (ajustar el patrón si es necesario)
        processed_df = processed_df.withColumn("snapshot_date", to_date(col("snapshot_date"), "yyyy-MM-dd"))
        processed_df = processed_df.withColumn("album_release_date", to_date(col("album_release_date"), "yyyy-MM-dd"))
        
        # Extraer año, mes y día a partir de snapshot_date para posibles análisis adicionales
        processed_df = processed_df.withColumn("snapshot_year", year(col("snapshot_date"))) \
                                   .withColumn("snapshot_month", month(col("snapshot_date"))) \
                                   .withColumn("snapshot_day", dayofmonth(col("snapshot_date")))
        
        # Limpiar el campo country: eliminar espacios extras y convertir a mayúsculas
        processed_df = processed_df.withColumn(
            "country", 
            upper(trim(regexp_replace(col("country"), "\\s+", " ")))
        )
        
        # Filtrar filas donde el nombre de la canción (track_name) no sea nulo
        processed_df = processed_df.filter(col("track_name").isNotNull())
        
        # Imprimir el schema y contar las filas después del procesamiento
        logger.info("Schema procesado:")
        processed_df.printSchema()
        count_after = processed_df.count()
        logger.info(f"Filas después del procesamiento: {count_after}")
        logger.info(f"Se removieron {count_before - count_after} filas durante el procesamiento")
        
        # Escribir el DataFrame procesado en formato Parquet en HDFS
        processed_df.write.mode("overwrite").parquet(output_path)
        logger.info(f"Datos procesados de Spotify guardados en {output_path}")
        
        return output_path
        
    except Exception as e:
        logger.error(f"Error al procesar los datos de Spotify: {e}")
        raise

def process_song_lyrics(spark, input_path, output_path):
    """
    Process the song lyrics dataset.
    
    Args:
        spark (SparkSession): The Spark session.
        input_path (str): The input file path.
        output_path (str): The output directory path.
    """
    logger.info(f"Processing song lyrics from {input_path}")
    
    try:
        # Read the CSV file with proper encoding
        df = spark.read.option("header", "true") \
                      .option("inferSchema", "true") \
                      .option("encoding", "utf-8") \
                      .option("multiline", "true") \
                      .option("escape", "\"") \
                      .csv(input_path)
        
        # Print schema and count before processing
        logger.info("Original schema:")
        df.printSchema()
        count_before = df.count()
        logger.info(f"Count before processing: {count_before}")
        
        # Clean and transform the data based on the actual columns in songs-lyrics.csv
        processed_df = df.select(
            col("song_id").cast("integer").alias("song_id"),
            col("singer_name").alias("artist_name"),
            col("song_name").alias("track_name"),
            col("song_href").alias("song_url")
        )
        
        # Clean artist and track names
        processed_df = processed_df.withColumn("artist_name", trim(col("artist_name"))) \
                                 .withColumn("track_name", trim(col("track_name")))
        
        # Filter rows with valid song_id and track_name
        processed_df = processed_df.filter(col("song_id").isNotNull() & col("track_name").isNotNull())
        
        # Print schema and count after processing
        logger.info("Processed schema:")
        processed_df.printSchema()
        count_after = processed_df.count()
        logger.info(f"Count after processing: {count_after}")
        logger.info(f"Removed {count_before - count_after} rows during processing")
        
        # Write the processed data as Parquet
        processed_df.write.mode("overwrite").parquet(output_path)
        logger.info(f"Processed song lyrics data saved to {output_path}")
        
        return output_path
        
    except Exception as e:
        logger.error(f"Error processing song lyrics dataset: {e}")
        raise