import pandas as pd
import plotly.express as px
import plotly.io as pio
from datetime import datetime
import re

def prepare_data(df, country, start_date, end_date):
    # Convertir las fechas de entrada a objetos datetime
    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date).date()
    if isinstance(end_date, str):
        end_date = pd.to_datetime(end_date).date()
        
    # Filtrar y limpiar datos
    df = df[
        (df['country'] == country) &
        (df['snapshot_date'].between(start_date, end_date)) &
        (df['daily_rank'] <= 15)
    ].copy()
    
    # Limpieza de caracteres especiales
    df['track_name'] = df['track_name'].apply(lambda x: re.sub(r'[^\w\s\-()]', '', x) if isinstance(x, str) else '')
    df['artist_name'] = df['artist_name'].apply(lambda x: re.sub(r'[^\w\s\-()]', '', x) if isinstance(x, str) else '')
    
    # Crear identificador único y valores de ranking invertido
    df['song_id'] = df['track_name'] + ' - ' + df['artist_name']
    df['rank_value'] = 16 - df['daily_rank']
    
    # Crear estructura completa de posiciones
    all_dates = sorted(df['snapshot_date'].unique())  # Ordenar cronológicamente
    
    # Enfoque alternativo para evitar el error de MultiIndex duplicado
    positions = range(1, 16)
    
    # Crear un DataFrame base con todas las combinaciones posibles de fechas y posiciones
    base_df = pd.DataFrame([(date, pos) for date in all_dates for pos in positions], 
                          columns=['snapshot_date', 'position'])
    
    # Preparar los datos existentes
    df_sorted = df.sort_values(['snapshot_date', 'daily_rank'])
    
    # Manejar posibles duplicados: quedarse con el primero en caso de conflicto
    df_cleaned = df_sorted.drop_duplicates(['snapshot_date', 'daily_rank'], keep='first')
    
    # Renombrar 'daily_rank' a 'position' para hacer el merge
    df_cleaned = df_cleaned.rename(columns={'daily_rank': 'position'})
    
    # Hacer merge para conseguir todas las combinaciones fecha-posición
    df_complete = pd.merge(
        base_df, 
        df_cleaned, 
        on=['snapshot_date', 'position'], 
        how='left'
    )
    
    # Encontrar la primera aparición de cada canción
    all_songs = df_complete[df_complete['song_id'].notna()]['song_id'].unique()
    
    first_appearance = {}
    for song in all_songs:
        first_date = df_complete[df_complete['song_id'] == song]['snapshot_date'].min()
        first_appearance[song] = first_date
    
    # Añadir información al DataFrame completo
    df_complete['first_appearance'] = df_complete.apply(
        lambda row: first_appearance.get(row['song_id'], pd.NaT) if pd.notna(row['song_id']) else pd.NaT, 
        axis=1
    )
    
    # Marcar si es la primera aparición de la canción
    df_complete['is_new'] = df_complete.apply(
        lambda row: row['snapshot_date'] == row['first_appearance'] if pd.notna(row['first_appearance']) else False,
        axis=1
    )
    
    # Rellenar valores nulos
    return df_complete.fillna({'song_id': '', 'rank_value': 0})

def create_bar_race(df, country, output_file=None):
    import os
    import webbrowser
    
    # Gestionar la ubicación del archivo de salida
    if output_file is None:
        # Crear carpeta animations si no existe
        animations_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'animations')
        if not os.path.exists(animations_dir):
            os.makedirs(animations_dir)
        output_file = os.path.join(animations_dir, f'top15_spotify_{country}.html')
    else:
        # Si se dio un nombre de archivo, colocarlo en la carpeta animations
        animations_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'animations')
        if not os.path.exists(animations_dir):
            os.makedirs(animations_dir)
        # Si es una ruta absoluta, usar el nombre del archivo
        if os.path.isabs(output_file):
            output_file = os.path.join(animations_dir, os.path.basename(output_file))
        else:
            output_file = os.path.join(animations_dir, output_file)
    
    # Configuración de colores
    unique_songs = df[df['song_id'] != '']['song_id'].unique()
    colors = px.colors.qualitative.Plotly + px.colors.qualitative.Dark24
    color_map = {song: colors[i % len(colors)] for i, song in enumerate(unique_songs)}
    
    # Crear figura animada con las fechas en orden cronológico
    fig = px.bar(
        df,
        x='rank_value',
        y='position',
        color='song_id',
        animation_frame='snapshot_date',
        orientation='h',
        color_discrete_map=color_map,
        range_x=[0, 25],  # Rango extendido para barras más largas
        range_y=[15.5, 0.5],  # Invertir eje Y
        labels={'rank_value': 'Puntuación', 'position': 'Posición'},
        title=f'Evolución del Top 15 Musical - {country}',
        height=900,  # Aumentar altura
        width=1600   # Aumentar ancho significativamente
    )
    
    # Personalización avanzada
    fig.update_layout(
        showlegend=False,
        plot_bgcolor='rgb(20, 20, 20)',
        paper_bgcolor='rgb(10, 10, 10)',
        font=dict(color='white', size=14),
        yaxis=dict(
            title='',
            tickvals=list(range(1, 16)),
            ticktext=[f'Posición {i}' for i in range(1, 16)],
            gridcolor='rgba(255, 255, 255, 0.1)'
        ),
        xaxis=dict(visible=False),
        hoverlabel=dict(
            bgcolor='rgba(0, 0, 0, 0.7)',
            font_size=14
        ),
        title=dict(
            x=0.5,
            font=dict(size=24),
            xanchor='center'
        ),
        margin=dict(l=50, r=30, t=80, b=50)  # Ajustar márgenes
    )
    
    # Ajustar barras sin texto por ahora
    fig.update_traces(
        hovertemplate='<b>%{customdata}</b><br>Posición: %{y}<br>Puntuación: %{x}',
        marker=dict(line=dict(color='rgba(255, 255, 255, 0.5)', width=1)),
    )
    
    # Configurar velocidad constante (2 segundos exactos por frame)
    fig.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = 2000
    fig.layout.updatemenus[0].buttons[0].args[1]['transition']['duration'] = 500
    
    # Procesar frames para mejorar la animación
    for i, frame in enumerate(fig.frames):
        # Ordenar el frame por posición para garantizar que se procesan correctamente
        frame_data_sorted = sorted([(trace.y[0], trace) for trace in frame.data], key=lambda x: x[0])
        
        # Procesar cada traza en el frame
        for j, (_, trace) in enumerate(frame_data_sorted):
            # Añadir customdata para el hovering
            if not hasattr(trace, 'customdata') or trace.customdata is None:
                trace.customdata = [trace.name]
            
            # Añadir información de texto explícitamente
            if hasattr(trace, 'x') and len(trace.x) > 0 and trace.x[0] > 0:
                # Solo añadir texto si la barra tiene valor
                trace.text = [trace.name]
                trace.textposition = 'inside'
                trace.insidetextanchor = 'start'  # Mostrar texto al inicio de la barra
                trace.textfont = dict(
                    size=15,  # Texto más grande
                    color='white',
                    family='Arial, sans-serif'
                )
                # Ajustar para que el texto siempre sea visible
                if trace.x[0] < 5:  # Si la barra es pequeña
                    trace.textposition = 'outside'
                    trace.outsidetextfont = dict(
                        size=15,
                        color='white',
                        family='Arial, sans-serif'
                    )
            else:
                # Barras sin valor no tienen texto
                trace.text = ['']
                trace.textposition = 'none'
        
        # Si no es el primer frame, manejar nuevas canciones
        if i > 0:
            prev_frame = fig.frames[i-1]
            
            # Identificar canciones nuevas en este frame
            current_songs = {t.name for t in frame.data}
            prev_songs = {t.name for t in prev_frame.data}
            new_songs = current_songs - prev_songs
            
            # Manejar cada traza nueva
            for trace in frame.data:
                if trace.name in new_songs:
                    # Asegurar que barras nuevas empiecen desde cero
                    trace.base = 0
                    # Estilo especial para barras nuevas
                    trace.marker.line.width = 2
                    trace.marker.line.color = 'rgba(255, 255, 255, 0.8)'
    
    # Ordenar cronológicamente
    fig.frames = sorted(fig.frames, key=lambda x: pd.Timestamp(x.name))
    
    # Exportar a HTML con mejores configuraciones
    fig.write_html(
        output_file,
        auto_play=True,
        include_plotlyjs='cdn',
        full_html=True,
        config={
            'scrollZoom': False,
            'displayModeBar': False,
            'responsive': True
        }
    )
    
    # Mejorar el HTML para pantalla completa y mejor rendimiento
    with open(output_file, 'r', encoding='utf-8') as file:
        html_content = file.read()
    
    # Añadir estilos CSS para mejorar la apariencia
    full_screen_html = html_content.replace(
        '<body>',
        '''<body style="margin:0; padding:0; overflow:hidden; background-color: #121212;">
        <style>
        html, body {
            width: 100%;
            height: 100%;
            margin: 0;
            padding: 0;
        }
        .plotly-graph-div {
            height: 100vh !important;
            width: 100vw !important;
        }
        .bar-animation text {
            dominant-baseline: middle;
        }
        </style>'''
    ).replace(
        '<div id="',
        '<div class="bar-animation" style="position:absolute; width:100%; height:100%;" id="'
    )
    
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write(full_screen_html)
    
    # Abrir el archivo en el navegador predeterminado
    webbrowser.open('file://' + os.path.abspath(output_file), new=2)
    
    return output_file

def visualizationPipeline(country, start_date, end_date):
    df = pd.read_parquet("C:/Users/maxmg/Documents/GitHub/BDA_P1/src/max/data/exploitation_zone/data_visualization")
    df = prepare_data(df, country, start_date, end_date)
    create_bar_race(df, country)

# Ejemplo de uso
if __name__ == '__main__':
    # Definir país, fecha de inicio y fecha de fin
    country = 'ES'
    start_date = '2025-01-01'
    end_date = '2025-04-04'
    
    # Ejecutar la visualización
    visualizationPipeline(country, start_date, end_date)