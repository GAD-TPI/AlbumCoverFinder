# -*- coding: utf-8 -*-
# Declaración de codificación de archivos, esencial para manejar caracteres UTF-8 (como acentos o la ñ).
"""
Aplicación Streamlit para la Búsqueda de Imágenes por Similitud (Content-Based Image Retrieval - CBIR).

Este script implementa una aplicación web que permite a los usuarios buscar imágenes (carátulas
de álbumes) en un dataset grande, utilizando características extraídas por la red neuronal
convolucional ResNet50 y métricas de distancia (Euclidiana y Coseno) para determinar la similitud.

La aplicación ahora está configurada para solo permitir búsquedas mediante la subida de un archivo
externo, eliminando la complejidad del 'self-match' (una imagen encontrándose a sí misma).
"""

# --- Importaciones de Librerías ---
import os
import logging
import datetime
import pandas as pd # Para la manipulación de datos y generación del CSV de resultados
from PIL import Image, ImageDraw, ImageFont # Para la manipulación de imágenes y generación de los mosaicos

# Configuración para suprimir mensajes de log y optimización de TensorFlow/Keras
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import streamlit as st # La librería principal para construir la interfaz web
import glob # Para buscar archivos en el sistema de archivos
import numpy as np # Para operaciones numéricas eficientes
from tensorflow import keras
from keras._tf_keras.keras.preprocessing import image # Herramientas para cargar y preprocesar imágenes
from keras.applications.resnet50 import ResNet50, preprocess_input # Modelo pre-entrenado y su preprocesamiento
from keras.models import Model # Para definir el extractor de características
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances # Métricas de distancia (similitud)

# --- Configuración de Rutas del Proyecto ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # Directorio base del script
DATA_DIR = os.path.join(BASE_DIR, 'data') # Contenedor general de datos
FEATURES_DIR = os.path.join(BASE_DIR, 'features') # Carpeta para guardar los vectores de características precalculados
RESULTS_DIR = os.path.join(BASE_DIR, 'results') # Carpeta donde se guardan los resultados de cada consulta (CSV y mosaicos)
DATASET_PATH = os.path.join(DATA_DIR, 'dataset') # Ruta donde residen las imágenes del dataset
FEATURES_FILE = os.path.join(FEATURES_DIR, 'dataset_features.npy') # Nombre del archivo para guardar los vectores
NAMES_FILE = os.path.join(FEATURES_DIR, 'dataset_names.npy') # Nombre del archivo para guardar los nombres de las imágenes

# --- Funciones de Carga de Modelo y Datos ---

@st.cache_resource
def cargar_modelo(): 
    """
    Carga y configura el modelo ResNet50.

    Usa @st.cache_resource para asegurar que el modelo se cargue en memoria UNA SOLA VEZ, 
    optimizando el rendimiento de la aplicación.
    """
    # ResNet50: Red neuronal convolucional pre-entrenada en ImageNet.
    # include_top=False: Excluye las capas finales de clasificación (solo nos interesa la extracción de features).
    # pooling='avg': Aplica Global Average Pooling, transformando el mapa de features en un vector de 2048 dimensiones.
    base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    
    # Define el modelo extractor: input es la imagen, output es el vector de pooling.
    feature_extractor = Model(inputs=base_model.input, outputs=base_model.output)
    return feature_extractor

def cargar_imagenes_dataset(carpeta): 
    """
    Busca todas las rutas de imágenes (jpg, jpeg, png, bmp) dentro de la carpeta del dataset.
    """
    rutas_absolutas = []
    # Busca en todos los subdirectorios ('**')
    for ext in ('*.jpg', '*.jpeg', '*.png', '*.bmp'):
        rutas_absolutas.extend(glob.glob(os.path.join(carpeta, '**', ext), recursive=True))
    
    # Genera rutas relativas (nombres de archivo) para usarlas como identificadores
    rutas_relativas = [os.path.relpath(r, carpeta) for r in rutas_absolutas]
    return rutas_absolutas, rutas_relativas

@st.cache_data
def cargar_caracteristicas_dataset(_model): 
    """
    Carga los vectores de características precalculados o los genera si es necesario.

    Usa @st.cache_data para evitar recalcular los features si el código no cambia.
    """
    os.makedirs(FEATURES_DIR, exist_ok=True)
    rutas_dataset, nombres_actuales = cargar_imagenes_dataset(DATASET_PATH)
    
    if not rutas_dataset:
        st.error(f"No se encontraron imágenes en la carpeta: {DATASET_PATH}")
        return None, None

    features_dataset = np.empty((0, 2048)) # Inicializa matriz de features
    nombres_cacheados = []
    
    # Intenta cargar datos existentes (cache)
    if os.path.exists(FEATURES_FILE) and os.path.exists(NAMES_FILE):
        try:
            features_dataset = np.load(FEATURES_FILE)
            nombres_cacheados = np.load(NAMES_FILE, allow_pickle=True).tolist()
        except Exception:
            # Si falla la carga del cache, reinicia la base de datos de características
            features_dataset = np.empty((0, 2048))
            nombres_cacheados = []
            
    nombres_set = set(nombres_cacheados)
    rutas_a_procesar = [] 
    nombres_a_procesar = [] 

    # Identifica las imágenes que necesitan procesamiento (nuevas o no cacheadas)
    for i, img_path in enumerate(rutas_dataset):
        img_name = nombres_actuales[i] 
        if img_name not in nombres_set:
            rutas_a_procesar.append(img_path)
            nombres_a_procesar.append(img_name)

    # Procesa imágenes nuevas en lotes (batches) para optimizar el uso de la GPU/CPU
    if rutas_a_procesar:
        BATCH_SIZE = 32 
        total_lotes = (len(rutas_a_procesar) + BATCH_SIZE - 1) // BATCH_SIZE
        progress_text_template = "Actualizando base de datos. Lote {current_batch}/{total_batches}..."
        progress_bar = st.progress(0, text=progress_text_template.format(current_batch=0, total_lotes=total_lotes))

        temp_features_list = []
        temp_names_list = []

        for i in range(0, len(rutas_a_procesar), BATCH_SIZE):
            batch_num_actual = (i // BATCH_SIZE) + 1
            progress_bar.progress(batch_num_actual / total_lotes, text=progress_text_template.format(current_batch=batch_num_actual, total_lotes=total_lotes))

            batch_paths = rutas_a_procesar[i : i + BATCH_SIZE]
            batch_names = nombres_a_procesar[i : i + BATCH_SIZE]
            batch_images_arrays = []
            valid_batch_names = []

            for j, img_path in enumerate(batch_paths):
                try:
                    # Carga y redimensiona (224x224)
                    img = image.load_img(img_path, target_size=(224, 224)) 
                    img_array = image.img_to_array(img)
                    batch_images_arrays.append(img_array)
                    valid_batch_names.append(batch_names[j]) 
                except Exception:
                    continue
            
            if not batch_images_arrays: continue
            
            batch_array = np.array(batch_images_arrays)
            batch_preprocessed = preprocess_input(batch_array) # Preprocesamiento ResNet50
            batch_features = _model.predict(batch_preprocessed, verbose=0) # Extracción
            
            temp_features_list.append(batch_features)
            temp_names_list.extend(valid_batch_names)
            
        # Concatena los resultados
        if temp_features_list:
            features_dataset = np.vstack([features_dataset] + temp_features_list)
            nombres_cacheados.extend(temp_names_list)
        progress_bar.empty()

    return features_dataset, nombres_cacheados


def prepare_image(img_input, target_size=(224, 224)): 
    """Preprocesa una imagen (resize, to_array, expand_dims, preprocess_input) para el modelo."""
    img = image.load_img(img_input, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) # Añade la dimensión de batch (requerida por Keras)
    return preprocess_input(img_array)

def _extract_features(img_input, model): 
    """Extrae el vector de 2048 características de una imagen usando ResNet50."""
    img_preprocessed = prepare_image(img_input)
    features = model.predict(img_preprocessed, verbose=0)
    return features.flatten() # Retorna el vector plano

def buscar_vecinos(features_dataset, features_query, nombres_dataset, 
                         radio_euc, radio_cos, top_k=10):
    """
    Busca las k imágenes más cercanas (k-NN) al vector de consulta, 
    filtrando por los radios de búsqueda para cada métrica.
    """
    
    # Calcula la distancia entre la consulta y CADA elemento del dataset
    dist_euc_all = euclidean_distances([features_query], features_dataset)[0]
    dist_cos_all = cosine_distances([features_query], features_dataset)[0]
    
    nombres_dataset = np.array(nombres_dataset)
    
    # En esta versión simplificada (solo carga de archivos), 'ignorar_self' es innecesario, 
    # ya que la imagen de consulta no existe en el dataset de búsqueda.
    dist_euc_filtered = dist_euc_all
    nombres_euc_filtered = nombres_dataset
    dist_cos_filtered = dist_cos_all
    nombres_cos_filtered = nombres_dataset
    
    # --- Procesamiento Euclidiana ---
    idx_euc_sorted = np.argsort(dist_euc_filtered) # Índices ordenados por distancia ASCENDENTE
    resultados_euc = []
    for i in range(len(idx_euc_sorted)):
        idx = idx_euc_sorted[i]
        dist = dist_euc_filtered[idx]
        
        if dist > radio_euc: break # Filtro por radio
        
        resultados_euc.append({'nombre': nombres_euc_filtered[idx], 'dist': dist})
        if len(resultados_euc) >= top_k: break # Filtro por máximo de resultados (10)

    # --- Procesamiento Coseno ---
    idx_cos_sorted = np.argsort(dist_cos_filtered)
    resultados_cos = []
    for i in range(len(idx_cos_sorted)):
        idx = idx_cos_sorted[i]
        dist = dist_cos_filtered[idx]
        if dist > radio_cos: break
        resultados_cos.append({'nombre': nombres_cos_filtered[idx], 'dist': dist})
        if len(resultados_cos) >= top_k: break

    return {'euc': resultados_euc, 'cos': resultados_cos}

def guardar_consulta_y_resultados(query_image, query_name, resultados, subfolder_name):
    """
    Función de persistencia: genera la carpeta de la consulta, guarda el CSV con resultados
    unificados (Euclidiana y Coseno) y los mosaicos de imágenes.
    """
    
    # 1. Creación de la carpeta de resultados (Ej: results/consulta_8582_20251116_213000)
    query_dir = os.path.join(RESULTS_DIR, subfolder_name)
    os.makedirs(query_dir, exist_ok=True)
    
    # 2. Guardar la imagen de consulta
    query_image.save(os.path.join(query_dir, f"consulta_{query_name}.jpg"))
    
    # 3. Generar CSV Unificado con Columna 'Unanime'
    
    max_len = max(len(resultados['euc']), len(resultados['cos']))
    
    data = []
    for i in range(max_len):
        row = {'Puesto': i + 1}
        
        nombre_euc = 'N/A'
        nombre_cos = 'N/A'
        
        # Llenar datos de Euclidiana
        if i < len(resultados['euc']):
            nombre_euc = resultados['euc'][i]['nombre']
            row['Nombre_Archivo_Euc'] = nombre_euc
            row['Distancia_Euc'] = resultados['euc'][i]['dist']
        else:
            row['Nombre_Archivo_Euc'] = 'N/A'
            row['Distancia_Euc'] = 'N/A'
            
        # Llenar datos de Coseno
        if i < len(resultados['cos']):
            nombre_cos = resultados['cos'][i]['nombre']
            row['Nombre_Archivo_Cos'] = nombre_cos
            row['Distancia_Cos'] = resultados['cos'][i]['dist']
        else:
            row['Nombre_Archivo_Cos'] = 'N/A'
            row['Distancia_Cos'] = 'N/A'
        
        # Columna: UNANIME (Verifica si ambas métricas coincidieron en el archivo en ese puesto)
        if nombre_euc != 'N/A' and nombre_cos != 'N/A':
            row['Unanime'] = 1 if nombre_euc == nombre_cos else 0
        else:
            row['Unanime'] = 0 
            
        data.append(row)
        
    df = pd.DataFrame(data)
    
    csv_file = os.path.join(query_dir, f"resultados_unificados.csv")
    df.to_csv(csv_file, index=False)
    
    # 4. Generar Imágenes Consolidadas (Mosaico) para la visualización
    
    def generar_imagen_consolidada(metric_key, metric_results, metric_name):
        if not metric_results: return
        
        IMG_SIZE = 150 # Tamaño de cada carátula en el mosaico
        IMG_SPACING = 5
        INFO_HEIGHT = 40 # Altura para texto de puesto/distancia
        TITLE_HEIGHT = 60 # Altura para el encabezado
        
        rows = 3
        cols = 4 # La cuadrícula es 4x3 (1 slot para Consulta + 11 slots para resultados, max 10)
        
        ancho_final = 4 * IMG_SIZE + (cols + 1) * IMG_SPACING
        alto_final = TITLE_HEIGHT + rows * IMG_SIZE + rows * INFO_HEIGHT + (rows + 1) * IMG_SPACING
        
        img_compuesta = Image.new('RGB', (ancho_final, alto_final), color='white')
        draw = ImageDraw.Draw(img_compuesta)
        
        try:
            font_title = ImageFont.truetype("arial.ttf", 20)
            font_info = ImageFont.truetype("arial.ttf", 16)
        except IOError:
            font_title = ImageFont.load_default()
            font_info = ImageFont.load_default()
            
        # Dibuja el Título
        title_text = f"10 imágenes más similares a {query_name} (Distancia {metric_name})"
        text_w, text_h = draw.textbbox((0, 0), title_text, font=font_title)[2:]
        draw.text(((ancho_final - text_w) / 2, IMG_SPACING), title_text, fill='black', font=font_title)
        draw.line([(0, TITLE_HEIGHT - IMG_SPACING), (ancho_final, TITLE_HEIGHT - IMG_SPACING)], fill='gray', width=1)

        y_offset = TITLE_HEIGHT # Punto de inicio Y para el contenido
        
        # Dibuja la Consulta (Posición 0,0)
        q_img_resized = query_image.resize((IMG_SIZE, IMG_SIZE))
        img_compuesta.paste(q_img_resized, (IMG_SPACING, y_offset + IMG_SPACING))
        draw.text((IMG_SPACING, y_offset + IMG_SIZE + IMG_SPACING), f"CONSULTA: {query_name}", fill='black', font=font_info)
        
        # Dibuja los Resultados (Posiciones 1 a 10)
        for i, res in enumerate(metric_results):
            puesto = i + 1
            
            # Cálculo de la posición en la cuadrícula
            if i < 3:
                row_idx = 0
                col_idx = i + 1
            elif i < 7:
                row_idx = 1
                col_idx = i - 3
            else:
                row_idx = 2
                col_idx = i - 7
            
            x_start = col_idx * IMG_SIZE + (col_idx + 1) * IMG_SPACING
            y_start = y_offset + row_idx * (IMG_SIZE + INFO_HEIGHT) + IMG_SPACING
            
            try:
                res_img_path = os.path.join(DATASET_PATH, res['nombre'])
                res_img = Image.open(res_img_path).resize((IMG_SIZE, IMG_SIZE))
                img_compuesta.paste(res_img, (x_start, y_start))
                
                text_line_1 = f"#{puesto} ({res['nombre']})"
                text_line_2 = f"Dist: {res['dist']:.2f}"
                
                draw.text((x_start, y_start + IMG_SIZE), text_line_1, fill='black', font=font_info)
                draw.text((x_start, y_start + IMG_SIZE + 18), text_line_2, fill='black', font=font_info)
                
            except FileNotFoundError:
                draw.rectangle([x_start, y_start, x_start + IMG_SIZE, y_start + IMG_SIZE], fill="gray")
                draw.text((x_start, y_start + IMG_SIZE), f"NO ENCONTRADO", fill='black', font=font_info)
        
        img_compuesta.save(os.path.join(query_dir, f"imagen_consolidada_{metric_key}.jpg"))
        
    generar_imagen_consolidada('euclidiana', resultados['euc'], 'Euclidiana')
    generar_imagen_consolidada('coseno', resultados['cos'], 'Coseno')
    
    return query_dir

# --- INTERFAZ PRINCIPAL DE STREAMLIT ---

# 1. Definición de estilos CSS (Inyección HTML) para mejorar la estética
STYLING_CSS = """
<style>
/* Centra el título principal y aplica una tipografía formal */
h1 {
    text-align: center;
    font-family: 'Times New Roman', Times, serif; 
    font-weight: 700;
    color: #333333;
    border-bottom: 2px solid #DDDDDD;
    padding-bottom: 10px;
}

/* Centra el botón de búsqueda y define su ancho máximo */
div.stButton > button {
    display: block;
    margin: 0 auto;
    width: 100%; 
    max-width: 250px; 
    background-color: #4CAF50; 
    border-radius: 8px;
}

/* Estilo para títulos de sección (h2/subheader) */
h2 {
    color: #555555;
    border-left: 5px solid #007bff; 
    padding-left: 10px;
    margin-top: 10px;
    margin-bottom: 15px;
}

/* Ajustes visuales para la imagen y los inputs */
.stImage { margin-bottom: -15px; }
.stNumberInput { margin-bottom: 15px; }
</style>
"""
st.markdown(STYLING_CSS, unsafe_allow_html=True)
st.set_page_config(page_title="Buscador de Carátulas", layout="wide") 

st.title("Buscador de Carátulas de Álbumes Similares")

# Carga inicial del modelo (cacheado)
feature_extractor = cargar_modelo()

# Carga inicial de la base de datos de características (cacheado)
with st.spinner('Cargando base de datos...'):
    features_dataset, nombres_dataset = cargar_caracteristicas_dataset(feature_extractor)

if features_dataset is None or not nombres_dataset:
    st.error("No se pudo cargar la base de datos. Revisa la consola.")
else:
    st.success(f"Base de datos lista: {len(nombres_dataset)} imágenes cargadas.")
    st.markdown("---")
    
    # Inicialización de variables para almacenar la consulta
    query_features = None
    query_image = None
    query_name = ""

    # --- Seccion de Entrada de la Consulta (Solo por Carga de Archivo) ---
    uploaded_file = st.file_uploader(
        "Sube una imagen de consulta aquí:",
        type=["jpg", "jpeg", "png"]
    )
    
    if uploaded_file is not None:
        # Procesa el archivo subido
        query_image = Image.open(uploaded_file)
        query_name = uploaded_file.name
        
        uploaded_file.seek(0) # Vuelve el puntero al inicio para que Keras lo pueda leer
        query_features = _extract_features(uploaded_file, feature_extractor) 
        
        # Estructura de columnas para el diseño lateral (Imagen | Controles)
        col_img, col_controls = st.columns([1, 1.5])
        
        with col_img:
            st.image(query_image, caption='Imagen de consulta', width=250)
        
        with col_controls:
            # Campos de entrada para definir los radios de búsqueda
            radio_euc = st.number_input("Radio de Búsqueda (Euclidiana):", min_value=0.0, value=30.0, step=0.5, key="euc_carga")
            radio_cos = st.number_input("Radio de Búsqueda (Coseno):", min_value=0.0, max_value=2.0, value=0.45, step=0.01, key="cos_carga")
            
            st.markdown('<div style="height: 15px;"></div>', unsafe_allow_html=True)
            if st.button('Buscar imágenes similares', type="primary", key="btn_carga"):
                
                # Inicia la búsqueda
                with st.spinner('Buscando...'):
                    resultado = buscar_vecinos(
                        features_dataset, 
                        query_features, 
                        nombres_dataset,
                        radio_euc,
                        radio_cos
                    )
                    
                    # Guarda el resultado en el estado de la sesión
                    st.session_state['resultado'] = resultado
                    st.session_state['query_info'] = (query_image, query_name, radio_euc, radio_cos)
                    st.session_state['run_search'] = True # Bandera para mostrar los resultados
                
                st.rerun() # Fuerza el rerenderizado para mover la visualización de resultados al final


    # --- LÓGICA DE VISUALIZACIÓN DE RESULTADOS ---

    # Inicializa el estado en el primer run
    if 'run_search' not in st.session_state:
        st.session_state['run_search'] = False

    # Este bloque se ejecuta solo si se completó una búsqueda exitosamente
    if st.session_state['run_search']:
        
        resultado = st.session_state['resultado']
        query_image, query_name, radio_euc, radio_cos = st.session_state['query_info']
        
        # Genera el nombre de la carpeta (consulta_NOMBRE_YYYYMMDD_HHMMSS)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        clean_query_name = query_name.split('.')[0] 
        subfolder_name = f"consulta_{clean_query_name}_{timestamp}"
        
        # Persistencia de datos
        query_dir = guardar_consulta_y_resultados(query_image, query_name, resultado, subfolder_name)
        
        st.markdown("---")
        st.success(f"Resultados guardados en: {query_dir}")
        st.subheader('Resultados de la Búsqueda')
        
        # Función auxiliar para renderizar los resultados en la interfaz
        def render_results(metric_name, results, radio):
            with st.expander(f"**{metric_name}** | {len(results)} resultados encontrados (Radio ≤ {radio})", expanded=True):
                
                if not results:
                    st.info("No se encontraron resultados en este radio.")
                    return
                
                cols = st.columns(4) 
                
                for i, res in enumerate(results):
                    try:
                        img_path = os.path.join(DATASET_PATH, res['nombre'])
                        img = Image.open(img_path) 
                        
                        puesto = i + 1
                        caption_text = f"**#{puesto}** (Dist: {res['dist']:.2f})"
                        
                        with cols[i % 4]:
                            st.image(img, caption=caption_text, width=150) 
                            st.caption(res['nombre']) 

                    except FileNotFoundError:
                        st.warning(f"No se encontró el archivo para mostrar: {res['nombre']}")

        # Muestra resultados para ambas métricas
        render_results("Euclidiana", resultado['euc'], radio_euc)
        st.markdown("---") 
        render_results("Coseno", resultado['cos'], radio_cos)