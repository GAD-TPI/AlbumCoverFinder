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

# --- Carga de Modelo (con Caché) ---

@st.cache_resource
def cargar_modelo():
    """Carga el modelo ResNet50 pre-entrenado."""
    print("Iniciando carga de modelo ResNet50...")
    base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    feature_extractor = Model(inputs=base_model.input, outputs=base_model.output)
    print("Modelo cargado exitosamente.")
    return feature_extractor

# --- Carga de Características del Dataset (con Caché) ---

def cargar_imagenes_dataset(carpeta):
    rutas = []
    # <--- MODIFICADO: Busca también en subcarpetas (recursive=True) ---
    for ext in ('*.jpg', '*.jpeg', '*.png', '*.bmp'):
        rutas.extend(glob.glob(os.path.join(carpeta, '**', ext), recursive=True))
    nombres = [os.path.basename(r) for r in rutas]
    return rutas, nombres

# <--- MODIFICADO: La función ahora acepta el modelo como argumento ---
@st.cache_data
def cargar_caracteristicas_dataset(_model): # _model es el feature_extractor
    os.makedirs(FEATURES_DIR, exist_ok=True)
    rutas_dataset, nombres_actuales = cargar_imagenes_dataset(DATASET_PATH)
    if not rutas_dataset:
        st.error(f"No se encontraron imágenes en la carpeta: {DATASET_PATH}")
        return None, None

    features_dataset = np.empty((0, 2048))
    nombres_cacheados = []
    if os.path.exists(FEATURES_FILE) and os.path.exists(NAMES_FILE):
        print("Cargando características desde caché...")
        features_dataset = np.load(FEATURES_FILE)
        nombres_cacheados = np.load(NAMES_FILE).tolist()
    else:
        print("No se encontró caché. Se procesará todo el dataset.")

    nombres_set = set(nombres_cacheados)
    rutas_a_procesar = []
    nombres_a_procesar = []

    for i, img_path in enumerate(rutas_dataset):
        img_name = nombres_actuales[i]
        if img_name not in nombres_set:
            rutas_a_procesar.append(img_path)
            nombres_a_procesar.append(img_name)

    # <--- INICIO: BLOQUE DE PROCESAMIENTO POR LOTES (COMO COLAB) ---
    if rutas_a_procesar:
        print(f"Procesando {len(rutas_a_procesar)} imágenes nuevas...")
        
        # --- CONFIGURACIÓN DE PROCESAMIENTO ---
        # BATCH_SIZE más bajo para CPU local
        BATCH_SIZE = 32 
        CHECKPOINT_EVERY_N_BATCHES = 5 # Guardar cada 5 lotes
        # ------------------------------------

        total_lotes = (len(rutas_a_procesar) + BATCH_SIZE - 1) // BATCH_SIZE
        progress_text_template = "Actualizando base de datos. Lote {current_batch}/{total_batches}..."
        progress_bar = st.progress(0, text=progress_text_template.format(current_batch=0, total_batches=total_lotes))

        temp_features_list = []
        temp_names_list = []

        for i in range(0, len(rutas_a_procesar), BATCH_SIZE):
            batch_num_actual = (i // BATCH_SIZE) + 1
            
            # Actualizar barra de progreso de Streamlit
            progress_percentage = batch_num_actual / total_lotes
            progress_text = progress_text_template.format(current_batch=batch_num_actual, total_batches=total_lotes)
            progress_bar.progress(progress_percentage, text=progress_text)

            # Preparar el lote
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
                except Exception as e:
                    print(f"\nError al cargar {img_path}: {e}. Omitiendo.")
            
            if not batch_images_arrays:
                print(f"Lote {batch_num_actual} vacío, saltando.")
                continue
            
            batch_array = np.array(batch_images_arrays)
            batch_preprocessed = preprocess_input(batch_array)
            
            # Usar el modelo pasado como argumento
            batch_features = _model.predict(batch_preprocessed, verbose=0)
            
            # Guardar resultados temporales en memoria
            temp_features_list.append(batch_features)
            temp_names_list.extend(valid_batch_names)

            # --- Lógica de Checkpoint ---
            if batch_num_actual % CHECKPOINT_EVERY_N_BATCHES == 0:
                print(f"Guardando checkpoint (Lote {batch_num_actual})...")
                
                # Unir los features y nombres temporales a los principales
                features_dataset = np.vstack([features_dataset] + temp_features_list)
                nombres_cacheados.extend(temp_names_list)
                
                # Guardar en disco
                try:
                    np.save(FEATURES_FILE, features_dataset)
                    np.save(NAMES_FILE, nombres_cacheados)
                    # Limpiar las listas temporales
                    temp_features_list = []
                    temp_names_list = []
                    print("Checkpoint guardado.")
                except Exception as e:
                    print(f"Error al guardar checkpoint: {e}")
        
        # --- Guardado Final ---
        if temp_features_list:
            print("Guardando últimos lotes restantes...")
            features_dataset = np.vstack([features_dataset] + temp_features_list)
            nombres_cacheados.extend(temp_names_list)
            try:
                np.save(FEATURES_FILE, features_dataset)
                np.save(NAMES_FILE, nombres_cacheados)
                print("Guardado final completado.")
            except Exception as e:
                print(f"Error en guardado final: {e}")

        progress_bar.empty()
        print("Caché actualizado y guardado.")
    # <--- FIN: BLOQUE DE PROCESAMIENTO POR LOTES ---
    else:
        print("La base de datos está al día.")

    return features_dataset, nombres_cacheados

# --- Funciones de Procesamiento ---

def prepare_image(img_input, target_size=(224, 224)):
    img = image.load_img(img_input, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)

def _extract_features(img_input, model):
    img_preprocessed = prepare_image(img_input)
    features = model.predict(img_preprocessed, verbose=0)
    return features.flatten()

# <--- INICIO: FUNCIÓN DE BÚSQUEDA MODIFICADA ---
def buscar_vecinos(features_dataset, features_query, nombres_dataset, 
                       radio_euc, radio_cos, top_k=10, ignorar_self=False):
    """
    Busca los k-vecinos más cercanos que estén dentro de un radio.
    """
    
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
# <--- FIN: FUNCIÓN DE BÚSQUEDA MODIFICADA ---


# --- Interfaz Principal de la Aplicación ---

st.set_page_config(page_title="Buscador de Carátulas", layout="wide")
st.title("🖼️ Buscador de Carátulas de Álbumes Similares")

# 1. Cargar el modelo
feature_extractor = cargar_modelo()

# 2. Cargar/Actualizar el dataset
with st.spinner('Cargando y verificando base de datos de imágenes...'):
    # <--- MODIFICADO: Pasamos el modelo a la función ---
    features_dataset, nombres_dataset = cargar_caracteristicas_dataset(feature_extractor)

if features_dataset is None or not nombres_dataset:
    st.error("No se pudo cargar la base de datos. Revisa la consola y la carpeta 'data/dataset'.")
else:
    st.success(f"¡Base de datos lista! Se cargaron {len(nombres_dataset)} imágenes.")
    st.markdown("---")
    
    # <--- INICIO: UI MODIFICADA CON TABS ---

    # --- CONTROLES DE BÚSQUEDA ---
    st.subheader("Parámetros de Búsqueda (Top 10)")
    col_radio1, col_radio2 = st.columns(2)
    with col_radio1:
        radio_euc = st.number_input("Radio de Búsqueda (Euclidiana):", min_value=0.0, value=20.0, step=0.5)
    with col_radio2:
        radio_cos = st.number_input("Radio de Búsqueda (Coseno):", min_value=0.0, max_value=2.0, value=0.4, step=0.01)

    # --- TABS PARA MÉTODOS DE BÚSQUEDA ---
    tab1, tab2 = st.tabs(["📤 Buscar por Carga", "🗂️ Buscar por Dataset"])

    query_features = None
    query_image = None
    ignorar_self = False

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
            # <--- MODIFICADO: Usamos el modelo global aquí ---
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
                st.error(f"No se pudo cargar la imagen de preview: {nombre_seleccionado}")
                query_features = None # Evitar que se pueda buscar

    # --- LÓGICA DE BÚSQUEDA Y RESULTADOS (COMÚN A AMBOS TABS) ---
    
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
                
                col_res1, col_res2 = st.columns(2)
                
                # --- Columna de Resultados EUCLIDIANOS ---
                with col_res1:
                    st.write(f"**Vecinos (Euclidiana)** (Radio <= {radio_euc})")
                    if not resultado['euc']:
                        st.info("No se encontraron resultados en este radio.")
                    
                    for res in resultado['euc']:
                        try:
                            img_path = os.path.join(DATASET_PATH, res['nombre'])
                            img = Image.open(img_path)
                            st.image(img, caption=f"Dist: {res['dist']:.2f}")
                            st.caption(f"Archivo: {res['nombre']}", help="Nombre del archivo en el dataset")
                        except FileNotFoundError:
                            st.error(f"No se encontró el archivo: {res['nombre']}")

                # --- Columna de Resultados COSENO ---
                with col_res2:
                    st.write(f"**Vecinos (Coseno)** (Radio <= {radio_cos})")
                    if not resultado['cos']:
                        st.info("No se encontraron resultados en este radio.")
                        
                    for res in resultado['cos']:
                        try:
                            img_path = os.path.join(DATASET_PATH, res['nombre'])
                            img = Image.open(img_path)
                            st.image(img, caption=f"Dist: {res['dist']:.2f}")
                            st.caption(f"Archivo: {res['nombre']}", help="Nombre del archivo en el dataset")
                        except FileNotFoundError:
                            st.error(f"No se encontró el archivo: {res['nombre']}")
    
    # <--- FIN: UI MODIFICADA CON TABS ---