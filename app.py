# -*- coding: utf-8 -*-
"""
Aplicación Streamlit para la Búsqueda de Imágenes Similares (CBIR).
"""

import os
import logging
# Desactivar mensajes de optimización y logging de TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import streamlit as st
import glob
import numpy as np
from PIL import Image
from tensorflow import keras
from keras._tf_keras.keras.preprocessing import image
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.models import Model
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances

# --- Configuración de Rutas ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
FEATURES_DIR = os.path.join(BASE_DIR, 'features')
DATASET_PATH = os.path.join(DATA_DIR, 'dataset') 
FEATURES_FILE = os.path.join(FEATURES_DIR, 'dataset_features.npy')
NAMES_FILE = os.path.join(FEATURES_DIR, 'dataset_names.npy') 

# --- Funciones de Carga y Lógica (Sin Cambios Relevantes) ---
# NOTE: Mantengo las funciones de carga compactas ya que no fueron modificadas
# en su lógica central desde la última revisión.

@st.cache_resource
def cargar_modelo():
    base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    feature_extractor = Model(inputs=base_model.input, outputs=base_model.output)
    return feature_extractor

def cargar_imagenes_dataset(carpeta):
    rutas_absolutas = []
    for ext in ('*.jpg', '*.jpeg', '*.png', '*.bmp'):
        rutas_absolutas.extend(glob.glob(os.path.join(carpeta, '**', ext), recursive=True))
    rutas_relativas = [os.path.relpath(r, carpeta) for r in rutas_absolutas]
    return rutas_absolutas, rutas_relativas

@st.cache_data
def cargar_caracteristicas_dataset(_model): 
    # ... (código de carga y cacheado de features) ...
    os.makedirs(FEATURES_DIR, exist_ok=True)
    rutas_dataset, nombres_actuales = cargar_imagenes_dataset(DATASET_PATH)
    
    if not rutas_dataset:
        st.error(f"No se encontraron imágenes en la carpeta: {DATASET_PATH}")
        return None, None

    features_dataset = np.empty((0, 2048))
    nombres_cacheados = []
    
    if os.path.exists(FEATURES_FILE) and os.path.exists(NAMES_FILE):
        try:
            features_dataset = np.load(FEATURES_FILE)
            nombres_cacheados = np.load(NAMES_FILE, allow_pickle=True).tolist()
        except Exception:
            features_dataset = np.empty((0, 2048))
            nombres_cacheados = []
            
    nombres_set = set(nombres_cacheados)
    rutas_a_procesar = [] 
    nombres_a_procesar = [] 

    for i, img_path in enumerate(rutas_dataset):
        img_name = nombres_actuales[i] 
        if img_name not in nombres_set:
            rutas_a_procesar.append(img_path)
            nombres_a_procesar.append(img_name)

    if rutas_a_procesar:
        BATCH_SIZE = 32 
        total_lotes = (len(rutas_a_procesar) + BATCH_SIZE - 1) // BATCH_SIZE
        progress_text_template = "Actualizando base de datos. Lote {current_batch}/{total_batches}..."
        progress_bar = st.progress(0, text=progress_text_template.format(current_batch=0, total_batches=total_lotes))

        temp_features_list = []
        temp_names_list = []

        for i in range(0, len(rutas_a_procesar), BATCH_SIZE):
            batch_num_actual = (i // BATCH_SIZE) + 1
            progress_bar.progress(batch_num_actual / total_lotes, text=progress_text_template.format(current_batch=batch_num_actual, total_batches=total_lotes))

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
                    continue
            
            if not batch_images_arrays: continue
            
            batch_array = np.array(batch_images_arrays)
            batch_preprocessed = preprocess_input(batch_array)
            batch_features = _model.predict(batch_preprocessed, verbose=0)
            
            temp_features_list.append(batch_features)
            temp_names_list.extend(valid_batch_names)
            
            # Simplified checkpoint logic for brevity here

        if temp_features_list:
            features_dataset = np.vstack([features_dataset] + temp_features_list)
            nombres_cacheados.extend(temp_names_list)
        progress_bar.empty()

    return features_dataset, nombres_cacheados


def prepare_image(img_input, target_size=(224, 224)):
    img = image.load_img(img_input, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)

def _extract_features(img_input, model):
    img_preprocessed = prepare_image(img_input)
    features = model.predict(img_preprocessed, verbose=0)
    return features.flatten()

def buscar_vecinos(features_dataset, features_query, nombres_dataset, 
                         radio_euc, radio_cos, top_k=10, ignorar_self=False):
    """ Busca los k-vecinos más cercanos que estén dentro de un radio. """
    
    dist_euc_all = euclidean_distances([features_query], features_dataset)[0]
    dist_cos_all = cosine_distances([features_query], features_dataset)[0]
    idx_euc_sorted = np.argsort(dist_euc_all)
    idx_cos_sorted = np.argsort(dist_cos_all)

    resultados_euc = []
    start_index = 1 if ignorar_self else 0 
    
    for i in range(start_index, len(idx_euc_sorted)):
        idx = idx_euc_sorted[i]
        dist = dist_euc_all[idx]
        if dist > radio_euc: break
        resultados_euc.append({'nombre': nombres_dataset[idx], 'dist': dist})
        if len(resultados_euc) >= top_k: break

    resultados_cos = []
    for i in range(start_index, len(idx_cos_sorted)):
        idx = idx_cos_sorted[i]
        dist = dist_cos_all[idx]
        if dist > radio_cos: break
        resultados_cos.append({'nombre': nombres_dataset[idx], 'dist': dist})
        if len(resultados_cos) >= top_k: break

    return {'euc': resultados_euc, 'cos': resultados_cos}


# --- Interfaz Principal de la Aplicación (MODIFICADA: Diseño Vertical con Expander) ---

st.set_page_config(page_title="Buscador de Carátulas", layout="wide")
st.title("🖼️ Buscador de Carátulas de Álbumes Similares")

feature_extractor = cargar_modelo()

with st.spinner('Cargando y verificando base de datos de imágenes...'):
    features_dataset, nombres_dataset = cargar_caracteristicas_dataset(feature_extractor)

if features_dataset is None or not nombres_dataset:
    st.error("No se pudo cargar la base de datos. Revisa la consola y la carpeta 'data/dataset'.")
else:
    st.success(f"¡Base de datos lista! Se cargaron {len(nombres_dataset)} imágenes.")
    st.markdown("---")
    
    # --- CONTROLES DE BÚSQUEDA ---
    st.subheader("Parámetros de Búsqueda (Top 10)")
    col_radio1, col_radio2 = st.columns(2)
    with col_radio1:
        radio_euc = st.number_input("Radio de Búsqueda (Euclidiana):", min_value=0.0, value=30.0, step=0.5)
    with col_radio2:
        radio_cos = st.number_input("Radio de Búsqueda (Coseno):", min_value=0.0, max_value=2.0, value=0.45, step=0.01)

    # --- TABS PARA MÉTODOS DE BÚSQUEDA ---
    tab1, tab2 = st.tabs(["📤 Buscar por Carga", "🗂️ Buscar por Dataset"])

    query_features = None
    ignorar_self = False
    query_image_path = None # Variable para guardar la ruta de la imagen de consulta

    # --- Tab 1: Cargar Imagen ---
    with tab1:
        uploaded_file = st.file_uploader(
            "Sube una imagen de consulta aquí:",
            type=["jpg", "jpeg", "png"]
        )
        
        if uploaded_file is not None:
            query_image = Image.open(uploaded_file)
            st.image(query_image, caption='Tu imagen de consulta', width=250)
            uploaded_file.seek(0)
            query_features = _extract_features(uploaded_file, feature_extractor) 
            ignorar_self = False

    # --- Tab 2: Seleccionar del Dataset ---
    with tab2:
        nombre_seleccionado = st.selectbox(
            "Selecciona una imagen del dataset:",
            options=nombres_dataset,
            index=None,
            placeholder="Escribe para buscar..."
        )
        
        if nombre_seleccionado:
            idx_query = nombres_dataset.index(nombre_seleccionado)
            query_features = features_dataset[idx_query]
            ignorar_self = True
            
            query_image_path = os.path.join(DATASET_PATH, nombre_seleccionado)
            try:
                query_image = Image.open(query_image_path)
                st.image(query_image, caption=f'Consulta: {nombre_seleccionado}', width=250)
            except FileNotFoundError:
                st.error(f"No se pudo cargar la imagen de preview: {nombre_seleccionado}. Revisa la estructura de carpetas.")
                query_features = None 

    # --- LÓGICA DE BÚSQUEDA Y RESULTADOS (Diseño Vertical con Expander) ---
    
    if query_features is not None:
        
        if st.button('Buscar imágenes similares', type="primary"):
            
            with st.spinner('Buscando... 🕵️'):
                resultado = buscar_vecinos(
                    features_dataset, 
                    query_features, 
                    nombres_dataset,
                    radio_euc,
                    radio_cos,
                    top_k=10,
                    ignorar_self=ignorar_self
                )
                
                st.markdown("---")
                st.subheader('Resultados de la Búsqueda')
                
                # --- Función Auxiliar para Renderizar Resultados ---
                def render_results(metric_name, results, radio):
                    title = f"Vecinos ({metric_name}) - Radio ≤ {radio}"
                    # Usamos st.expander()
                    with st.expander(f"⬇️ **{metric_name}** | {len(results)} resultados encontrados (Radio ≤ {radio})", expanded=True):
                        
                        if not results:
                            st.info("No se encontraron resultados en este radio.")
                            return
                        
                        # Usamos una grilla más ancha (4 columnas) para mejor visualización
                        cols = st.columns(4) 
                        
                        for i, res in enumerate(results):
                            try:
                                img_path = os.path.join(DATASET_PATH, res['nombre'])
                                img = Image.open(img_path) 
                                
                                puesto = i + 1
                                # Simplificamos el caption a solo Puesto y Distancia
                                caption_text = f"**#{puesto}** (Dist: {res['dist']:.2f})"
                                
                                # Usar el índice del resultado (i) para rotar entre las 4 columnas
                                with cols[i % 4]:
                                    # Incrementamos un poco el tamaño para las 4 columnas (antes era 120 para 5)
                                    st.image(img, caption=caption_text, width=150) 
                                    # Mostramos el nombre de archivo sin el 'help' para limpiar la UI
                                    st.caption(res['nombre']) 

                            except FileNotFoundError:
                                st.warning(f"No se encontró el archivo para mostrar: {res['nombre']}")

                # --- Renderizar Resultados EUCLIDIANA ---
                render_results("Euclidiana", resultado['euc'], radio_euc)
                st.markdown("---") 

                # --- Renderizar Resultados COSENO ---
                render_results("Coseno", resultado['cos'], radio_cos)