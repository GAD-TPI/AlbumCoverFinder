# -*- coding: utf-8 -*-
"""
Aplicación para Búsqueda de Carátulas de Álbumes por Similitud
Trabajo Práctico Integrador - Gestión Avanzada de Datos 2025
Grupo 7: Emilia Fernández, Salvador Tiguá

Funcionalidades principales:
1. Carga de un modelo pre-entrenado (ResNet50) para extracción de características.
2. Gestión de base de datos de imágenes (extracción incremental y caché).
3. Búsqueda mediante fuerza bruta o indexación eficiente (FAISS).
4. Interfaz gráfica (streamlit) para realizar consultas y visualizar resultados.
"""

import os
import sys
import logging
import datetime
import time
import glob

# --- Importaciones de Ciencia de Datos y ML ---
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances

# --- Configuración de TensorFlow/Keras ---
# Variables de entorno para evitar logging molesto
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import tensorflow as keras
from keras._tf_keras.keras.preprocessing import image
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.models import Model

# --- Importaciones de la Aplicación ---
import streamlit as st # para aplicación web
import faiss  # índice para búsquedas eficientes

# ==========================================
#       CONFIGURACIÓN Y CONSTANTES
# ==========================================

# Rutas base del proyecto
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
FEATURES_DIR = os.path.join(BASE_DIR, 'features')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

# Elegir dataset dentro de /data para utilizar
DATASET_NAME = 'dataset_80k' 
DATASET_PATH = os.path.join(DATA_DIR, DATASET_NAME)

# Archivos de caché para persistencia de vectores de características
FEATURES_FILE = os.path.join(FEATURES_DIR, f'{DATASET_NAME}_features.npy')
NAMES_FILE = os.path.join(FEATURES_DIR, f'{DATASET_NAME}_names.npy')

# ==========================================
#   FUNCIONES DE CARGA Y PREPROCESAMIENTO
# ==========================================

@st.cache_resource
def cargar_modelo():
    """
    Carga el modelo ResNet50 pre-entrenado en ImageNet sin la capa superior (top).
    Añade GlobalAveragePooling para obtener un vector de características de 2048 dimensiones.
    Usa @st.cache_resource para cargar el modelo en memoria una única vez.
    """
    base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    feature_extractor = Model(inputs=base_model.input, outputs=base_model.output)
    return feature_extractor

def cargar_imagenes_dataset(carpeta):
    """
    Escanea recursivamente la carpeta dada buscando archivos de imagen.
    Devuelve:
        - rutas_absolutas para cargar la imagen.
        - rutas_relativas para usar como ID único en la base de datos.
    """
    rutas_absolutas = []
    extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp')
    for ext in extensions:
        rutas_absolutas.extend(glob.glob(os.path.join(carpeta, '**', ext), recursive=True))
    rutas_relativas = [os.path.relpath(r, carpeta) for r in rutas_absolutas]
    return rutas_absolutas, rutas_relativas

@st.cache_data
def cargar_caracteristicas_dataset(_model):
    """
    Gestiona la extracción de características del dataset.
    1. Intenta cargar vectores pre-calculados desde el disco (archivos .npy).
    2. Si hay imágenes nuevas que no están en caché, las procesa en lotes.
    3. Actualiza el caché incrementalmente.
    """
    os.makedirs(FEATURES_DIR, exist_ok=True)
    rutas_dataset, nombres_actuales = cargar_imagenes_dataset(DATASET_PATH)
    
    if not rutas_dataset:
        st.error(f"No se encontraron imágenes en la carpeta: {DATASET_PATH}")
        return None, None

    features_dataset = np.empty((0, 2048))
    nombres_cacheados = []
    
    # --- Carga de Caché Existente ---
    if os.path.exists(FEATURES_FILE) and os.path.exists(NAMES_FILE):
        try:
            features_dataset = np.load(FEATURES_FILE)
            nombres_cacheados = np.load(NAMES_FILE, allow_pickle=True).tolist()
            
            # Verificación simple de integridad
            if len(nombres_cacheados) == len(rutas_dataset) and len(features_dataset) == len(rutas_dataset):
                st.info(f"Cargando {len(features_dataset)} características desde caché.")
                return features_dataset, nombres_cacheados
        except Exception: 
            # Si el caché está corrupto, reiniciamos
            features_dataset = np.empty((0, 2048))
            nombres_cacheados = []
            
    # --- Identificación de Imágenes Nuevas ---
    nombres_set = set(nombres_cacheados)
    rutas_a_procesar = [] 
    nombres_a_procesar = [] 

    for i, img_path in enumerate(rutas_dataset):
        img_name = nombres_actuales[i] 
        if img_name not in nombres_set:
            rutas_a_procesar.append(img_path)
            nombres_a_procesar.append(img_name)

    # --- Procesamiento por Lotes ---
    if rutas_a_procesar:
        BATCH_SIZE = 32 
        SAVE_INTERVAL = 15 # Intervalo para guardar progreso en disco
        total_lotes = (len(rutas_a_procesar) + BATCH_SIZE - 1) // BATCH_SIZE
        
        progress_text_template = "Procesando Lote {current_batch}/{total_lotes}"
        progress_slot = st.empty()
        progress_bar = progress_slot.progress(0, text=progress_text_template.format(current_batch=0, total_lotes=total_lotes))

        features_a_procesar = []
        
        for i in range(0, len(rutas_a_procesar), BATCH_SIZE):
            batch_num_actual = (i // BATCH_SIZE) + 1
            progress_bar.progress(batch_num_actual / total_lotes, 
                                text=progress_text_template.format(current_batch=batch_num_actual, total_lotes=total_lotes))

            # Preparar lote actual
            batch_paths = rutas_a_procesar[i : i + BATCH_SIZE]
            batch_names = nombres_a_procesar[i : i + BATCH_SIZE]
            batch_images_arrays = []
            valid_batch_names = []

            for j, img_path in enumerate(batch_paths):
                try:
                    img = image.load_img(img_path, target_size=(224, 224)) 
                    img_array = image.img_to_array(img)
                    batch_images_arrays.append(img_array)
                    valid_batch_names.append(batch_names[j]) 
                except Exception:
                    continue # Saltar imágenes corruptas
            
            if not batch_images_arrays: continue
            
            # Preprocesamiento y Predicción
            batch_array = np.array(batch_images_arrays)
            batch_preprocessed = preprocess_input(batch_array)
            batch_features = _model.predict(batch_preprocessed, verbose=0)
            
            features_a_procesar.append(batch_features)
            nombres_cacheados.extend(valid_batch_names)
            
            # --- Guardado Incremental ---
            if batch_num_actual % SAVE_INTERVAL == 0 or batch_num_actual == total_lotes:
                features_consolidadas = np.vstack([features_dataset] + features_a_procesar)
                
                np.save(FEATURES_FILE, features_consolidadas)
                np.save(NAMES_FILE, nombres_cacheados)

                features_dataset = features_consolidadas
                features_a_procesar = []
                st.info(f"Progreso guardado: {len(nombres_cacheados)}/{len(rutas_dataset)} imágenes procesadas.")

        progress_bar.empty()

    return features_dataset, nombres_cacheados

@st.cache_resource
def cargar_indice_faiss(features_dataset):
    """
    Construye los índices de búsqueda eficiente usando FAISS.
    Devuelve:
        - index_euc: índice para distancia Euclidiana.
        - index_cos: índice para distancia Coseno.
    """
    if features_dataset is None:
        return None, None
    
    D = features_dataset.shape[1] # Dimensión (2048)
    data = features_dataset.astype('float32')
    
    # Índice para euclidiana
    index_euc = faiss.IndexFlatL2(D)
    index_euc.add(data) 
    
    # Índice para coseno - requiere normalización L2 previa
    data_norm = data.copy()
    faiss.normalize_L2(data_norm)
    index_cos = faiss.IndexFlatIP(D)
    index_cos.add(data_norm)
    
    return index_euc, index_cos

def prepare_image(img_input, target_size=(224, 224)): 
    """Prepara una imagen individual para ser procesada por el modelo."""
    img = Image.open(img_input)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size) 
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)

def _extract_features(img_input, model): 
    """Función helper para extraer vector de características de una imagen."""
    try:
        img_preprocessed = prepare_image(img_input)
        features = model.predict(img_preprocessed, verbose=0)
        return features.flatten()
    except Exception as e:
        st.error(f"Error al procesar la imagen de consulta: {e}")
        return None

# ==========================================
#           MOTORES DE BÚSQUEDA
# ==========================================

def buscar_vecinos_brute_force(features_dataset, features_query, nombres_dataset, 
                                 radio_euc, radio_cos, top_k=10):
    """
    Calcula distancias comparando la consulta contra todo el dataset (Fuerza Bruta).
    """
    # Cálculo de todas las distancias
    dist_euc_all = euclidean_distances([features_query], features_dataset)[0]
    dist_cos_all = cosine_distances([features_query], features_dataset)[0]
    
    nombres_dataset = np.array(nombres_dataset)
    
    # Filtrado y ordenamiento para Euclidiana
    idx_euc_sorted = np.argsort(dist_euc_all)
    resultados_euc = []
    for idx in idx_euc_sorted:
        dist = dist_euc_all[idx]
        if dist > radio_euc: break # corte por radio
        resultados_euc.append({'nombre': nombres_dataset[idx], 'dist': dist})
        if len(resultados_euc) >= top_k: break

    # Filtrado y ordenamiento para Coseno
    idx_cos_sorted = np.argsort(dist_cos_all)
    resultados_cos = []
    for idx in idx_cos_sorted:
        dist = dist_cos_all[idx]
        if dist > radio_cos: break # corte por radio
        resultados_cos.append({'nombre': nombres_dataset[idx], 'dist': dist})
        if len(resultados_cos) >= top_k: break

    return {'euc': resultados_euc, 'cos': resultados_cos}

def buscar_vecinos_faiss(features_dataset, features_query, nombres_dataset, 
                          radio_euc, radio_cos, top_k=10, index_euc=None, index_cos=None):
    """
    Realiza la búsqueda utilizando índices optimizados FAISS.
    """
    if index_euc is None or index_cos is None:
        st.error("Error: Índices Faiss no cargados.")
        return {'euc': [], 'cos': []}
        
    query_vector = features_query.astype('float32').reshape(1, -1)
    K_MAX = len(nombres_dataset) # Buscamos en todos para poder filtrar por radio luego

    # --- Búsqueda Euclidiana ---
    D_euc, I_euc = index_euc.search(query_vector, K_MAX)
    
    resultados_euc = []
    for dist_squared, idx in zip(D_euc[0], I_euc[0]):
        dist = np.sqrt(dist_squared) # FAISS L2 devuelve distancia cuadrada
        if dist > radio_euc: break
        resultados_euc.append({'nombre': nombres_dataset[idx], 'dist': dist})
        if len(resultados_euc) >= top_k: break

    # --- Búsqueda Coseno ---
    faiss.normalize_L2(query_vector) # Normalizar consulta
    D_cos, I_cos = index_cos.search(query_vector, K_MAX)
    
    resultados_cos = []
    for sim, idx in zip(D_cos[0], I_cos[0]):
        dist = max(0, 1 - sim) # Convertir Similitud a Distancia
        if dist > radio_cos: break
        resultados_cos.append({'nombre': nombres_dataset[idx], 'dist': dist})
        if len(resultados_cos) >= top_k: break
        
    return {'euc': resultados_euc, 'cos': resultados_cos}

# ==========================================
#   GESTIÓN DE RESULTADOS Y PERSISTENCIA
# ==========================================

def guardar_consulta_y_resultados(query_image, query_name, resultados, subfolder_name, engine_text, log_data):
    """
    Guarda los artefactos de la consulta en la carpeta 'results/':
    1. Imagen de consulta.
    2. Archivo de log con metadatos.
    3. CSV con los resultados detallados.
    4. Mosaicos de imágenes visualizando la consulta, sus vecinos + información.
    """
    query_dir = os.path.join(RESULTS_DIR, subfolder_name)
    os.makedirs(query_dir, exist_ok=True)
    
    if query_image.mode != 'RGB':
        query_image = query_image.convert('RGB')
        
    query_image.save(os.path.join(query_dir, f"consulta_{query_name}.jpg"))
    
    # LOG DE METADATOS
    log_file = os.path.join(query_dir, "metadata_log.txt")
    with open(log_file, 'w', encoding='utf-8') as f: 
        f.write(f"--- LOG DE CONSULTA ---\n")
        f.write(f"Fecha: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Dataset: {DATASET_NAME}\n")
        f.write(f"Imagen: {query_name}\n")
        f.write(f"Motor: {engine_text}\n")
        f.write(f"Tiempo Ejecución: {log_data['Tiempo_Ejecucion_s']:.4f} s\n")
        f.write(f"Configuración Radio Euc: {log_data['Radio_Euclidiana']} | Hallados: {log_data['Total_Euc']}\n")
        f.write(f"Configuración Radio Cos: {log_data['Radio_Coseno']} | Hallados: {log_data['Total_Cos']}\n")
        
    # CSV DE RESULTADOS
    max_len = max(len(resultados['euc']), len(resultados['cos']))
    data = []
    for i in range(max_len):
        row = {'Puesto': i + 1}
        
        # Datos Euclidiana
        if i < len(resultados['euc']):
            nombre_euc = resultados['euc'][i]['nombre']
            row['Nombre_Archivo_Euc'] = nombre_euc
            row['Distancia_Euc'] = resultados['euc'][i]['dist']
        else:
            nombre_euc = 'N/A'
            row['Nombre_Archivo_Euc'] = 'N/A'
            row['Distancia_Euc'] = 'N/A'
            
        # Datos Coseno
        if i < len(resultados['cos']):
            nombre_cos = resultados['cos'][i]['nombre']
            row['Nombre_Archivo_Cos'] = nombre_cos
            row['Distancia_Cos'] = resultados['cos'][i]['dist']
        else:
            nombre_cos = 'N/A'
            row['Nombre_Archivo_Cos'] = 'N/A'
            row['Distancia_Cos'] = 'N/A'
        
        # Coincidencia Unánime
        row['Unanime'] = 1 if (nombre_euc != 'N/A' and nombre_euc == nombre_cos) else 0
        data.append(row)
        
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(query_dir, f"resultados_unificados.csv"), index=False)
    
    # GENERACIÓN DE IMAGEN CONSOLIDADA (MOSAICO)
    def generar_imagen_consolidada(metric_key, metric_results, metric_name):
        IMG_SIZE = 150
        IMG_SPACING = 10
        INFO_HEIGHT = 52
        TITLE_HEIGHT = 90
        rows, cols = 3, 4 # Grid 4x3

        ancho_final = cols * IMG_SIZE + (cols + 1) * IMG_SPACING
        alto_final = TITLE_HEIGHT + rows * IMG_SIZE + rows * INFO_HEIGHT + (rows + 1) * IMG_SPACING

        img_compuesta = Image.new('RGB', (ancho_final, alto_final), color='white')
        draw = ImageDraw.Draw(img_compuesta)

        # Cargar fuentes o usar default
        try:
            font_title = ImageFont.truetype("arial.ttf", 20)
            font_subtitle = ImageFont.truetype("arial.ttf", 16)
            font_info = ImageFont.truetype("arial.ttf", 14)
        except IOError:
            font_title = font_subtitle = font_info = ImageFont.load_default()

        # Encabezados
        main_title = f"10 imágenes más similares a {query_name}"
        sub_title = f"(Motor: {engine_text} | Distancia: {metric_name})"
        
        w_main = draw.textbbox((0, 0), main_title, font=font_title)[2]
        w_sub = draw.textbbox((0, 0), sub_title, font=font_subtitle)[2]
        
        draw.text(((ancho_final - w_main) / 2, IMG_SPACING), main_title, fill='black', font=font_title)
        draw.text(((ancho_final - w_sub) / 2, IMG_SPACING + 30), sub_title, fill='gray', font=font_subtitle)
        draw.line([(0, TITLE_HEIGHT - 10), (ancho_final, TITLE_HEIGHT - 10)], fill='gray', width=1)

        # Dibujar Imagen de Consulta (Posición 0,0)
        y_offset = TITLE_HEIGHT
        q_img_resized = query_image.resize((IMG_SIZE, IMG_SIZE))
        img_compuesta.paste(q_img_resized, (IMG_SPACING, y_offset + IMG_SPACING))
        
        caption_y = y_offset + IMG_SIZE + IMG_SPACING
        draw.text((IMG_SPACING, caption_y + 5), "CONSULTA", fill='black', font=font_info)
        draw.text((IMG_SPACING, caption_y + 20), query_name[:20], fill='black', font=font_info)

        # Dibujar Resultados
        if not metric_results:
            draw.text((ancho_final/2 - 50, y_offset + 50), "Sin resultados", fill='black', font=font_info)
        else:
            for i, res in enumerate(metric_results):
                # Calcular fila/columna en el grid (saltando la primera posición reservada para consulta)
                if i < 3: row_idx, col_idx = 0, i + 1
                elif i < 7: row_idx, col_idx = 1, i - 3
                else: row_idx, col_idx = 2, i - 7

                x = col_idx * IMG_SIZE + (col_idx + 1) * IMG_SPACING
                y = y_offset + row_idx * (IMG_SIZE + INFO_HEIGHT) + IMG_SPACING

                try:
                    res_img = Image.open(os.path.join(DATASET_PATH, res['nombre'])).convert('RGB')
                    res_img = res_img.resize((IMG_SIZE, IMG_SIZE))
                    img_compuesta.paste(res_img, (x, y))
                except Exception:
                    draw.rectangle([x, y, x + IMG_SIZE, y + IMG_SIZE], fill="lightgray")

                draw.text((x + 2, y + IMG_SIZE + 5), f"#{i+1} Dist: {res['dist']:.2f}", fill='black', font=font_info)
                draw.text((x + 2, y + IMG_SIZE + 20), res['nombre'][:20], fill='black', font=font_info)

        img_compuesta.save(os.path.join(query_dir, f"imagen_consolidada_{metric_key}.jpg"))
        
    generar_imagen_consolidada('euclidiana', resultados['euc'], 'Euclidiana')
    generar_imagen_consolidada('coseno', resultados['cos'], 'Coseno')
    
    return query_dir

# ==========================================
#      INTERFAZ DE USUARIO (STREAMLIT)
# ==========================================

# Estilos CSS Personalizados
STYLING_CSS = """
<style>
h1 { text-align: center; font-family: 'Serif'; color: #333; border-bottom: 2px solid #ddd; padding-bottom: 10px; }
div.stButton > button { display: block; margin: 0 auto; width: 100%; max-width: 250px; background-color: #4CAF50; border-radius: 8px; }
h2 { color: #555; border-left: 5px solid #007bff; padding-left: 10px; margin-top: 10px; }
</style>
"""
st.markdown(STYLING_CSS, unsafe_allow_html=True)
st.set_page_config(page_title="Buscador de Carátulas", layout="wide") 

st.title("Buscador de Carátulas de Álbumes Similares")

# Inicialización de Recursos
feature_extractor = cargar_modelo()

with st.spinner(f'Cargando e indexando base de datos ({DATASET_NAME})...'):
    features_dataset, nombres_dataset = cargar_caracteristicas_dataset(feature_extractor)
    
    # Inicializar índices FAISS solo si hay datos
    if features_dataset is not None and len(features_dataset) > 0:
        index_euc_faiss, index_cos_faiss = cargar_indice_faiss(features_dataset)
    else:
        index_euc_faiss, index_cos_faiss = None, None

# Lógica Principal de la UI
if features_dataset is None or not nombres_dataset:
    st.error("Error crítico: No se pudo cargar la base de datos.")
else:
    st.success(f"Sistema listo. {len(nombres_dataset)} imágenes indexadas.")
    st.markdown("---")
    
    # Variables de estado
    query_features = None
    query_image = None
    query_name = ""

    # Selector de Motor
    search_engine = st.selectbox(
        "Selecciona el Motor de Búsqueda:",
        ("Fuerza Bruta (ResNet)", "Faiss (Indexado)"),
        key="search_engine_select"
    )
    st.markdown("---")
    
    # Carga de Archivo de Consulta
    uploaded_file = st.file_uploader("Sube una imagen de consulta:", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Resetear resultados si cambia la imagen
        if 'uploaded_file_name' not in st.session_state or st.session_state.uploaded_file_name != uploaded_file.name:
            if 'resultado' in st.session_state: del st.session_state['resultado']
            st.session_state.uploaded_file_name = uploaded_file.name

        try:
            query_image = Image.open(uploaded_file)
            query_name = uploaded_file.name
            
            # Extraer features de la consulta
            uploaded_file.seek(0)
            query_features = _extract_features(uploaded_file, feature_extractor)
        except Exception as e:
            st.error(f"Error procesando imagen: {e}")
            st.stop()

        if query_features is not None:
            # Layout: Imagen a la izquierda, Controles a la derecha
            col_img, col_controls = st.columns([1, 1.5])
            
            with col_img:
                st.image(query_image, caption='Imagen de consulta', width=250)
            
            with col_controls:
                radio_euc = st.number_input("Radio de Búsqueda (Euclidiana):", min_value=0.0, value=30.0, step=0.5)
                radio_cos = st.number_input("Radio de Búsqueda (Coseno):", min_value=0.0, max_value=2.0, value=0.45, step=0.01)
                
                st.markdown('<div style="height: 15px;"></div>', unsafe_allow_html=True)
                
                if st.button('Buscar imágenes similares', type="primary"):
                    start_time = time.perf_counter() 
                    current_engine = st.session_state.search_engine_select

                    with st.spinner(f'Buscando con {current_engine}...'):
                        if current_engine == "Faiss (Indexado)":
                            resultado = buscar_vecinos_faiss(
                                features_dataset, query_features, nombres_dataset,
                                radio_euc, radio_cos, index_euc=index_euc_faiss, index_cos=index_cos_faiss
                            )
                        else:
                            resultado = buscar_vecinos_brute_force(
                                features_dataset, query_features, nombres_dataset,
                                radio_euc, radio_cos
                            )
                    
                    # Guardar resultados en sesión
                    st.session_state['elapsed_time'] = time.perf_counter() - start_time
                    st.session_state['resultado'] = resultado
                    st.session_state['query_info'] = (query_image, query_name, radio_euc, radio_cos)
                    st.session_state['search_engine'] = current_engine
                    st.rerun()

    # Visualización de Resultados
    if 'resultado' in st.session_state:
        resultado = st.session_state['resultado']
        query_image, query_name, radio_euc, radio_cos = st.session_state['query_info']
        used_engine = st.session_state['search_engine']
        elapsed_time = st.session_state['elapsed_time']
        
        # Guardar en disco
        engine_suffix = "indexado" if "Faiss" in used_engine else "fuerzabruta"
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        clean_name = query_name.split('.')[0]
        subfolder_name = f"consulta_{clean_name}_{engine_suffix}_{timestamp}"
        
        log_data = {
            'Tiempo_Ejecucion_s': elapsed_time,
            'Radio_Euclidiana': radio_euc,
            'Radio_Coseno': radio_cos,
            'Total_Euc': len(resultado['euc']),
            'Total_Cos': len(resultado['cos'])
        }
        
        query_dir = guardar_consulta_y_resultados(query_image, query_name, resultado, subfolder_name, used_engine, log_data) 
        
        st.markdown("---")
        st.success(f"Resultados guardados en: **{query_dir}**")
        st.info(f"Tiempo: **{elapsed_time:.4f} s** | Motor: {used_engine}")
        st.subheader('Resultados de la Búsqueda')

        def render_results_section(metric_name, results, radio):
            with st.expander(f"**{metric_name}** | {len(results)} resultados (Radio ≤ {radio})", expanded=True):
                if not results:
                    st.info("No se encontraron resultados en este radio.")
                    return
                cols = st.columns(4)
                for i, res in enumerate(results):
                    with cols[i % 4]:
                        try:
                            img = Image.open(os.path.join(DATASET_PATH, res['nombre']))
                            st.image(img, caption=f"#{i+1} Dist: {res['dist']:.2f}", width=150)
                            st.caption(res['nombre'])
                        except FileNotFoundError:
                            st.warning(f"Archivo no encontrado: {res['nombre']}")

        render_results_section("Euclidiana", resultado['euc'], radio_euc)
        st.markdown("---") 
        render_results_section("Coseno", resultado['cos'], radio_cos)