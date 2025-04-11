import os
import logging
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.functions import (
    col, count, avg, sum, desc, countDistinct, round, when, corr
)
from pyspark.sql.window import Window

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def analyze_top_artists(spark: SparkSession, tracks_path: str, output_path: str):
    """
    Analiza y determina los artistas principales en base a la popularidad de las pistas.
    (Uso de la tabla de tracks procesada).
    
    Args:
        spark (SparkSession): La sesión de Spark.
        tracks_path (str): Ruta a los datos procesados (tabla de tracks) en formato Parquet.
        output_path (str): Ruta donde se guardarán los resultados.
    """
    logger.info("Analizando top artistas basados en la popularidad de las pistas")
    try:
        tracks_df = spark.read.parquet(tracks_path)
        tracks_df.printSchema()  # Verificar que el esquema es el correcto

        artist_popularity = tracks_df.groupBy("artist_name") \
            .agg(
                round(avg("popularity"), 2).alias("avg_popularity"),
                count("track_id").alias("track_count")
            ) \
            .filter(col("track_count") >= 5) \
            .orderBy(desc("avg_popularity"))

        logger.info("Top artistas por popularidad:")
        artist_popularity.show(10, truncate=False)

        if output_path:
            artist_popularity.write.mode("overwrite") \
                .csv(os.path.join(output_path, "top_artists"), header=True)
            logger.info(f"Se han guardado los top artistas en {output_path}/top_artists")

        return artist_popularity

    except Exception as e:
        logger.error(f"Error analizando top artistas: {e}")
        raise


def analyze_song_popularity_by_country(spark: SparkSession, top_songs_path: str, output_path: str):
    """
    Analiza la popularidad de las canciones por país utilizando la tabla de Top Songs.
    
    Dado que ya no existe la columna 'streams', se utiliza el número de entradas
    (cantidad de filas) en el chart para estimar la presencia de una canción en el ranking.
    
    Args:
        spark (SparkSession): La sesión de Spark.
        top_songs_path (str): Ruta a la tabla de Top Songs procesada (Parquet).
        output_path (str): Ruta donde se guardarán los resultados.
    """
    logger.info("Analizando popularidad de canciones por país")
    try:
        df = spark.read.parquet(top_songs_path)
        df.printSchema()

        # 1. Top 10 países con mayor número de entradas en charts
        country_entries = df.groupBy("country") \
            .agg(
                count("*").alias("chart_entries"),
                countDistinct("track_name").alias("unique_tracks")
            ) \
            .orderBy(desc("chart_entries")) \
            .limit(10)

        logger.info("Top 10 países (por número de entradas en charts):")
        country_entries.show(10, truncate=False)

        if output_path:
            country_entries.write.mode("overwrite") \
                .csv(os.path.join(output_path, "top_countries_by_entries"), header=True)
            logger.info(f"Se han guardado los datos en {output_path}/top_countries_by_entries")

        # 2. Top 10 artistas globalmente según apariciones en charts
        top_artists = df.groupBy("artist_name") \
            .agg(
                count("*").alias("chart_appearances"),
                countDistinct("country").alias("countries_charted")
            ) \
            .orderBy(desc("chart_appearances")) \
            .limit(10)

        logger.info("Top 10 artistas globales según apariciones en charts:")
        top_artists.show(10, truncate=False)

        if output_path:
            top_artists.write.mode("overwrite") \
                .csv(os.path.join(output_path, "top_artists_global"), header=True)
            logger.info(f"Se han guardado los datos en {output_path}/top_artists_global")

        # 3. Top songs por país: calcular ranking con base en 'daily_rank'
        # Se asume que 'daily_rank' es numérico donde un valor menor es mejor posición.
        window_spec = Window.partitionBy("country").orderBy("daily_rank")
        top_songs_by_country = df.withColumn("rank_in_country", F.dense_rank().over(window_spec)) \
            .filter(col("rank_in_country") <= 5) \
            .select("country", "track_name", "artist_name", "daily_rank", "rank_in_country") \
            .orderBy("country", "rank_in_country")

        logger.info("Top songs por país (ejemplo):")
        top_songs_by_country.show(20, truncate=False)

        if output_path:
            top_songs_by_country.write.mode("overwrite") \
                .csv(os.path.join(output_path, "top_songs_by_country"), header=True)
            logger.info(f"Se han guardado los datos en {output_path}/top_songs_by_country")

        # 4. Tendencias en el tiempo, usando 'snapshot_year' y 'snapshot_month' si existen
        if "snapshot_year" in df.columns and "snapshot_month" in df.columns:
            trends_over_time = df.groupBy("snapshot_year", "snapshot_month") \
                .agg(
                    count("*").alias("chart_entries"),
                    countDistinct("track_name").alias("unique_tracks")
                ) \
                .orderBy("snapshot_year", "snapshot_month")
            logger.info("Tendencias de charts en el tiempo:")
            trends_over_time.show(truncate=False)

            if output_path:
                trends_over_time.write.mode("overwrite") \
                    .csv(os.path.join(output_path, "chart_trends_over_time"), header=True)
                logger.info(f"Se han guardado las tendencias en {output_path}/chart_trends_over_time")
        else:
            logger.warning("No se encontraron columnas de fecha ('snapshot_year', 'snapshot_month'); se omite el análisis de tendencias.")

        return True

    except Exception as e:
        logger.error(f"Error analizando top songs por país: {e}")
        raise


def analyze_audio_features_correlation(spark: SparkSession, tracks_path: str, output_path: str):
    """
    Analiza la correlación entre características de audio y la popularidad de las pistas.
    
    Args:
        spark (SparkSession): La sesión de Spark.
        tracks_path (str): Ruta a la tabla de tracks procesada (Parquet).
        output_path (str): Ruta para guardar los resultados.
    """
    logger.info("Analizando correlación entre características de audio y popularidad")
    try:
        tracks_df = spark.read.parquet(tracks_path)

        audio_features = [
            "danceability", "energy", "loudness", "speechiness",
            "acousticness", "instrumentalness", "liveness", "valence", "tempo"
        ]
        features_df = tracks_df.select("track_id", "popularity", *audio_features)

        correlation_results = []
        for feature in audio_features:
            corr_value = features_df.stat.corr("popularity", feature)
            correlation_results.append((feature, corr_value))

        correlation_df = spark.createDataFrame(correlation_results, ["feature", "correlation_with_popularity"])

        logger.info("Correlaciones de características de audio:")
        correlation_df.orderBy(desc("correlation_with_popularity")).show(truncate=False)

        if output_path:
            correlation_df.write.mode("overwrite") \
                .csv(os.path.join(output_path, "feature_correlations"), header=True)
            logger.info(f"Se han guardado las correlaciones en {output_path}/feature_correlations")

        return correlation_df

    except Exception as e:
        logger.error(f"Error analizando la correlación de características de audio: {e}")
        raise


def analyze_song_features(spark: SparkSession, tracks_data_path: str, output_path: str):
    """
    Analiza las características de audio de las pistas.
    
    Args:
        spark (SparkSession): La sesión de Spark.
        tracks_data_path (str): Ruta a la tabla de tracks procesada (Parquet).
        output_path (str): Ruta para guardar los resultados.
    """
    logger.info(f"Analizando características de audio desde {tracks_data_path}")
    try:
        df = spark.read.parquet(tracks_data_path)
        df.printSchema()

        avg_features = df.select(
            avg("danceability").alias("avg_danceability"),
            avg("energy").alias("avg_energy"),
            avg("loudness").alias("avg_loudness"),
            avg("speechiness").alias("avg_speechiness"),
            avg("acousticness").alias("avg_acousticness"),
            avg("instrumentalness").alias("avg_instrumentalness"),
            avg("liveness").alias("avg_liveness"),
            avg("valence").alias("avg_valence"),
            avg("tempo").alias("avg_tempo"),
            avg("duration_ms").alias("avg_duration_ms")
        )

        logger.info("Promedios de características de audio:")
        avg_features.show()

        if output_path:
            avg_features.write.mode("overwrite") \
                .csv(os.path.join(output_path, "average_audio_features"), header=True)

        feature_cols = [
            "danceability", "energy", "loudness", "speechiness",
            "acousticness", "instrumentalness", "liveness", "valence", "tempo"
        ]
        correlations = []
        for feature in feature_cols:
            corr_value = df.select(corr(col(feature), col("popularity")).alias("correlation")).collect()[0][0]
            correlations.append((feature, corr_value))
        correlation_df = spark.createDataFrame(correlations, ["feature", "correlation_with_popularity"])
        logger.info("Correlación entre características y popularidad:")
        correlation_df.orderBy(desc("correlation_with_popularity")).show()

        if output_path:
            correlation_df.write.mode("overwrite") \
                .csv(os.path.join(output_path, "feature_popularity_correlation"), header=True)

        danceable_songs = df.orderBy(desc("danceability")).select("track_name", "artist_name", "danceability", "popularity").limit(10)
        energetic_songs = df.orderBy(desc("energy")).select("track_name", "artist_name", "energy", "popularity").limit(10)
        acoustic_songs = df.orderBy(desc("acousticness")).select("track_name", "artist_name", "acousticness", "popularity").limit(10)

        logger.info("Canciones más bailables:")
        danceable_songs.show()
        logger.info("Canciones más energéticas:")
        energetic_songs.show()
        logger.info("Canciones más acústicas:")
        acoustic_songs.show()

        if output_path:
            danceable_songs.write.mode("overwrite") \
                .csv(os.path.join(output_path, "most_danceable_songs"), header=True)
            energetic_songs.write.mode("overwrite") \
                .csv(os.path.join(output_path, "most_energetic_songs"), header=True)
            acoustic_songs.write.mode("overwrite") \
                .csv(os.path.join(output_path, "most_acoustic_songs"), header=True)

        if "release_year" in df.columns:
            songs_by_year = df.filter(col("release_year").isNotNull()) \
                .groupBy("release_year") \
                .agg(
                    count("*").alias("song_count"),
                    avg("popularity").alias("avg_popularity")
                ) \
                .orderBy("release_year")
            logger.info("Distribución de canciones por año:")
            songs_by_year.show()

            if output_path:
                songs_by_year.write.mode("overwrite") \
                    .csv(os.path.join(output_path, "songs_by_year"), header=True)
        else:
            logger.info("No se encontró la columna 'release_year'; se omite el análisis de distribución por año.")

        return True

    except Exception as e:
        logger.error(f"Error analizando las características de audio: {e}")
        raise


def analyze_lyrics_and_songs(spark: SparkSession, lyrics_path: str, tracks_path: str, output_path: str):
    """
    Realiza un análisis conjunto entre la tabla de lyrics y la de tracks.
    
    Se asume que la tabla de lyrics procesada contiene al menos: song_id, artist_name, track_name, song_url.
    Y la de tracks (procesada) contiene: track_id, track_name, artist_name, popularity, etc.
    
    Args:
        spark (SparkSession): La sesión de Spark.
        lyrics_path (str): Ruta a la tabla de lyrics procesada (Parquet).
        tracks_path (str): Ruta a la tabla de tracks procesada (Parquet).
        output_path (str): Ruta donde se guardarán los resultados.
    """
    logger.info("Realizando análisis conjunto entre lyrics y tracks")
    try:
        lyrics_df = spark.read.parquet(lyrics_path)
        tracks_df = spark.read.parquet(tracks_path)

        # Seleccionar columnas para el join
        lyrics_for_join = lyrics_df.select(
            col("artist_name").alias("lyrics_artist"),
            col("track_name").alias("lyrics_track")
        )
        tracks_for_join = tracks_df.select(
            col("artist_name"),
            col("track_name"),
            col("popularity"),
            "danceability", "energy", "loudness", "speechiness",
            "acousticness", "instrumentalness", "liveness", "valence", "tempo"
        )

        joined_df = lyrics_for_join.join(
            tracks_for_join,
            (lyrics_for_join.lyrics_artist == tracks_for_join.artist_name) &
            (lyrics_for_join.lyrics_track == tracks_for_join.track_name),
            "inner"
        )

        match_count = joined_df.count()
        logger.info(f"Se encontraron {match_count} coincidencias entre lyrics y tracks")

        if match_count > 0 and output_path:
            joined_df.write.mode("overwrite") \
                .csv(os.path.join(output_path, "lyrics_tracks_combined"), header=True)
            logger.info(f"Se ha guardado la tabla combinada en {output_path}/lyrics_tracks_combined")

        return True

    except Exception as e:
        logger.error(f"Error en el análisis conjunto de lyrics y tracks: {e}")
        raise
