# Análisis de Datos de Spotify con PySpark

Este proyecto implementa un sistema de procesamiento de datos utilizando PySpark para analizar datasets de Spotify. El sistema sigue un enfoque de data lake con diferentes zonas de almacenamiento:

- **Landing Zone**: Datos crudos descargados de Kaggle
- **Formatted Zone**: Datos limpios y estructurados
- **Trusted Zone**: Resultados de análisis 
- **Analytics Zone**: Visualizaciones y resultados finales

## Estructura del Proyecto

```
pyspark-project/
│
├── data/
│   ├── landing_zone/      # Datos crudos
│   ├── formatted_zone/    # Datos procesados
│   ├── trusted_zone/      # Resultados de análisis
│   └── analytics_zone/    # Visualizaciones y resultados finales
│
├── src/
│   ├── main.py            # Script principal
│   │
│   ├── jobs/
│   │   ├── data_processing.py      # Procesamiento de datos
│   │   ├── data_analysis.py        # Análisis de datos
│   │   └── data_visualization.py   # Visualización de datos
│   │
│   └── utils/
│       ├── spark_session.py        # Configuración de Spark
│       └── file_utils.py           # Utilidades para manejo de archivos
│
└── requirements.txt       # Dependencias
```

## Datasets

El sistema trabaja con los siguientes datasets de Spotify:

1. **Top Spotify Songs por País**: Canciones más populares en diferentes países
2. **Spotify Tracks Dataset**: Información detallada sobre canciones, incluyendo características de audio
3. **Songs Lyrics**: Letras de canciones

## Requisitos

- Python 3.8+
- Java 8+
- PySpark 3.4.1
- Kaggle API configurada (para descargar los datasets)

## Instalación

1. Clonar el repositorio:
   ```
   git clone <repo_url>
   cd pyspark-project
   ```

2. Instalar dependencias:
   ```
   pip install -r requirements.txt
   ```

3. Configurar la API de Kaggle:
   - Descarga tu archivo kaggle.json desde tu cuenta de Kaggle
   - Colócalo en `~/.kaggle/kaggle.json` (Linux/Mac) o `C:\Users\<usuario>\.kaggle\kaggle.json` (Windows)
   - Asegúrate de que los permisos son correctos: `chmod 600 ~/.kaggle/kaggle.json` (solo en Linux/Mac)

## Uso

1. **Descargar datos**:
   - Los datos se descargan automáticamente usando el script `landing.py` en la raíz del proyecto
   - Ejecuta: `python ../landing.py`

2. **Ejecutar el pipeline completo**:
   ```
   cd src
   python main.py
   ```

3. **Ver resultados**:
   - Los resultados de análisis se guardan en `data/trusted_zone/`
   - Las visualizaciones se guardan en `data/analytics_zone/`

## Análisis realizados

1. **Top artistas por popularidad**: Identificación de los artistas más populares basado en métricas de Spotify.
2. **Popularidad de canciones por país**: Análisis de la popularidad de canciones desglosado por países.
3. **Correlación de características de audio**: Análisis de cómo las diferentes características de audio se correlacionan con la popularidad de las canciones.

## Extensión del proyecto

Este sistema puede ser extendido:

- Añadiendo nuevas fuentes de datos
- Implementando más análisis avanzados
- Creando un sistema de recomendación de música
- Añadiendo un dashboard interactivo