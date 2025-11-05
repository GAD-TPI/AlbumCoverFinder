# -*- coding: utf-8 -*-
"""
Aplicación Streamlit para la Búsqueda de Imágenes Similares (CBIR).

Esta app permite al usuario subir una imagen de carátula de álbum y
encuentra las imágenes más similares en un dataset local usando ResNet50.
"""

import os
import logging
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import streamlit as st
import glob
import numpy as np
from PIL import Image  # Usamos PIL para manejar imágenes en Streamlit
from tensorflow import keras
from keras._tf_keras.keras.preprocessing import image
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.models import Model
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances

# --- Configuración de Rutas ---
# (Igual que en search.py, asume que 'data' y 'features' están en el mismo nivel)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
FEATURES_DIR = os.path.join(BASE_DIR, 'features')
DATASET_PATH = os.path.join(DATA_DIR, 'dataset')
FEATURES_FILE = os.path.join(FEATURES_DIR, 'dataset_features.npy')
NAMES_FILE = os.path.join(FEATURES_DIR, 'dataset_names.npy')

# --- Carga de Modelo (con Caché) ---

@st.cache_resource  # Decorador mágico: solo carga el modelo UNA VEZ.
def cargar_modelo():
    """Carga el modelo ResNet50 pre-entrenado."""
    print("Iniciando carga de modelo ResNet50...") # Log para la consola
    base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    feature_extractor = Model(inputs=base_model.input, outputs=base_model.output)
    print("Modelo cargado exitosamente.")
    return feature_extractor

# --- Carga de Características del Dataset (con Caché) ---

def cargar_imagenes_dataset(carpeta):
    """Carga rutas de imágenes y sus nombres de una carpeta."""
    rutas = []
    for ext in ('*.jpg', '*.jpeg', '*.png', '*.bmp'):
        rutas.extend(glob.glob(os.path.join(carpeta, ext)))
    nombres = [os.path.basename(r) for r in rutas]
    return rutas, nombres

@st.cache_data  # Decorador mágico: solo carga y procesa los datos UNA VEZ.
def cargar_caracteristicas_dataset():
    """
    Carga las características del dataset desde el caché (.npy).
    Si hay imágenes nuevas en la carpeta /data/dataset, las procesa,
    actualiza el caché y devuelve el set completo.
    """
    os.makedirs(FEATURES_DIR, exist_ok=True)
    
    # 1. Cargar rutas de imágenes actuales
    rutas_dataset, nombres_actuales = cargar_imagenes_dataset(DATASET_PATH)
    if not rutas_dataset:
        st.error(f"No se encontraron imágenes en la carpeta: {DATASET_PATH}")
        return None, None

    # 2. Cargar características cacheadas si existen
    features_dataset = np.empty((0, 2048))
    nombres_cacheados = []
    if os.path.exists(FEATURES_FILE) and os.path.exists(NAMES_FILE):
        print("Cargando características desde caché...")
        features_dataset = np.load(FEATURES_FILE)
        nombres_cacheados = np.load(NAMES_FILE).tolist()
    else:
        print("No se encontró caché. Se procesará todo el dataset.")

    # 3. Encontrar imágenes nuevas que no están en caché
    nombres_set = set(nombres_cacheados)
    rutas_a_procesar = []
    nombres_a_procesar = []

    for i, img_path in enumerate(rutas_dataset):
        img_name = nombres_actuales[i]
        if img_name not in nombres_set:
            rutas_a_procesar.append(img_path)
            nombres_a_procesar.append(img_name)

    # 4. Procesar imágenes nuevas (si las hay)
    if rutas_a_procesar:
        print(f"Procesando {len(rutas_a_procesar)} imágenes nuevas...")
        
        # Barra de progreso en Streamlit
        progress_text = f"Actualizando base de datos. Procesando {len(rutas_a_procesar)} imágenes nuevas..."
        progress_bar = st.progress(0, text=progress_text)
        
        new_features = []
        for i, img_path in enumerate(rutas_a_procesar):
            # _feature_extractor es el modelo cargado con @st.cache_resource
            features = _extract_features(img_path, _feature_extractor)
            new_features.append(features)
            nombres_cacheados.append(nombres_a_procesar[i])
            progress_bar.progress((i + 1) / len(rutas_a_procesar), text=f"{progress_text} ({i+1}/{len(rutas_a_procesar)})")
        
        progress_bar.empty() # Limpiar barra de progreso
        
        # Unir y guardar
        features_dataset = np.vstack([features_dataset] + new_features)
        try:
            np.save(FEATURES_FILE, features_dataset)
            np.save(NAMES_FILE, nombres_cacheados)
            print("Caché actualizado y guardado.")
        except Exception as e:
            st.error(f"Error al guardar archivos de caché: {e}")
    else:
        print("La base de datos está al día.")

    return features_dataset, nombres_cacheados

# --- Funciones de Procesamiento ---

def prepare_image(img_input, target_size=(224, 224)):
    """
    Carga y pre-procesa una imagen para ResNet50.
    'img_input' puede ser una ruta de archivo o un objeto de archivo (de Streamlit).
    """
    img = image.load_img(img_input, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)

def _extract_features(img_input, model):
    """Función interna para extraer características."""
    img_preprocessed = prepare_image(img_input)
    features = model.predict(img_preprocessed, verbose=0)
    return features.flatten()

def buscar_vecino_mas_cercano(features_dataset, features_query, nombres_dataset):
    """
    Busca los vecinos más cercanos (Euclidiano y Coseno) para una
    única consulta.
    """
    dist_euc = euclidean_distances([features_query], features_dataset)[0]
    dist_cos = cosine_distances([features_query], features_dataset)[0]

    idx_euc = np.argmin(dist_euc)
    idx_cos = np.argmin(dist_cos)

    resultado = {
        'vecino_euc': nombres_dataset[idx_euc],
        'vecino_cos': nombres_dataset[idx_cos],
        'dist_euc': dist_euc[idx_euc],
        'dist_cos': dist_cos[idx_cos]
    }
    return resultado

# --- Interfaz Principal de la Aplicación ---

st.set_page_config(page_title="Buscador de Carátulas", layout="wide")
st.title("🖼️ Buscador de Carátulas de Álbumes Similares")

# 1. Cargar el modelo (se usará caché si ya está cargado)
feature_extractor = cargar_modelo()

# 2. Cargar/Actualizar el dataset (se usará caché si está al día)
with st.spinner('Cargando y verificando base de datos de imágenes...'):
    features_dataset, nombres_dataset = cargar_caracteristicas_dataset()

if features_dataset is None or not nombres_dataset:
    st.error("No se pudo cargar la base de datos. Revisa la consola y la carpeta 'data/dataset'.")
else:
    st.success(f"¡Base de datos lista! Se cargaron {len(nombres_dataset)} imágenes.")
    st.markdown("---")

    # 3. Widget para subir la imagen de consulta
    uploaded_file = st.file_uploader(
        "Sube una imagen de consulta aquí:",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        # Mostrar la imagen subida
        query_image = Image.open(uploaded_file)
        
        col_izq, col_der = st.columns(2)
        with col_izq:
            st.image(query_image, caption='Tu imagen de consulta', use_container_width=True)

        with col_der:
            # 4. Botón para iniciar la búsqueda
            if st.button('Buscar imágenes similares', type="primary"):
                
                # 5. Procesar la búsqueda
                with st.spinner('Buscando... 🕵️'):
                    
                    # Extraer características de la imagen subida
                    # (El file uploader se "rebobina" con seek(0) por si acaso)
                    uploaded_file.seek(0)
                    features_query = _extract_features(uploaded_file, feature_extractor)
                    
                    # Realizar la búsqueda
                    resultado = buscar_vecino_mas_cercano(
                        features_dataset, 
                        features_query, 
                        nombres_dataset
                    )
                    
                    # 6. Cargar imágenes de resultados
                    try:
                        euc_path = os.path.join(DATASET_PATH, resultado['vecino_euc'])
                        cos_path = os.path.join(DATASET_PATH, resultado['vecino_cos'])
                        
                        img_euc = Image.open(euc_path)
                        img_cos = Image.open(cos_path)
                        
                        # 7. Mostrar resultados
                        st.subheader('Resultados de la Búsqueda')
                        
                        col_res1, col_res2 = st.columns(2)
                        with col_res1:
                            st.image(img_euc, caption=f"Vecino Euclidiano (Dist: {resultado['dist_euc']:.2f})")
                            st.write(f"Archivo: `{resultado['vecino_euc']}`")

                        with col_res2:
                            st.image(img_cos, caption=f"Vecino Coseno (Dist: {resultado['dist_cos']:.2f})")
                            st.write(f"Archivo: `{resultado['vecino_cos']}`")

                    except FileNotFoundError as e:
                        st.error(f"Error: No se pudo encontrar el archivo de imagen resultado: {e}")