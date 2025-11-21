# -*- coding: utf-8 -*-

import os
import logging
import datetime
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import time # Importado para medir el tiempo de ejecución

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import streamlit as st
import glob
import numpy as np
import faiss 
from tensorflow import keras
from keras._tf_keras.keras.preprocessing import image
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.models import Model
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances

# --- Configuración de Rutas ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
FEATURES_DIR = os.path.join(BASE_DIR, 'features')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

# --- SELECCIÓN DE DATASET (Modificar solo esta sección) ---
# Descomenta la línea de la base de datos que deseas usar.
# DATASET_NAME = 'dataset_20k' # Base de datos actual (20k)
DATASET_NAME = 'dataset_80k' # Nueva base de datos (80k)
# ---------------------------------------------------------

# Rutas dependientes del dataset seleccionado
DATASET_PATH = os.path.join(DATA_DIR, DATASET_NAME) 
# Los archivos de caché ahora incluyen el nombre del dataset para evitar conflictos
FEATURES_FILE = os.path.join(FEATURES_DIR, f'{DATASET_NAME}_features.npy')
NAMES_FILE = os.path.join(FEATURES_DIR, f'{DATASET_NAME}_names.npy') 

# --- Funciones de Carga y Preprocesamiento ---

@st.cache_resource
def cargar_modelo(): # carga y configura el modelo ResNet50
    base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    feature_extractor = Model(inputs=base_model.input, outputs=base_model.output)
    return feature_extractor

def cargar_imagenes_dataset(carpeta): # busca todas las imágenes válidas en /data
    rutas_absolutas = []
    for ext in ('*.jpg', '*.jpeg', '*.png', '*.bmp'):
        rutas_absolutas.extend(glob.glob(os.path.join(carpeta, '**', ext), recursive=True))
    rutas_relativas = [os.path.relpath(r, carpeta) for r in rutas_absolutas]
    return rutas_absolutas, rutas_relativas

@st.cache_data
def cargar_caracteristicas_dataset(_model): # carga características precalculadas o las calcula
    os.makedirs(FEATURES_DIR, exist_ok=True)
    rutas_dataset, nombres_actuales = cargar_imagenes_dataset(DATASET_PATH)
    
    if not rutas_dataset:
        st.error(f"No se encontraron imágenes en la carpeta: {DATASET_PATH}")
        return None, None

    features_dataset = np.empty((0, 2048))
    nombres_cacheados = []
    
    # 1. Intenta cargar datos existentes (cache)
    if os.path.exists(FEATURES_FILE) and os.path.exists(NAMES_FILE):
        try:
            features_dataset = np.load(FEATURES_FILE)
            nombres_cacheados = np.load(NAMES_FILE, allow_pickle=True).tolist()
            # Si se cargan todos los nombres y el número de features coincide, asumimos que está completo.
            if len(nombres_cacheados) == len(rutas_dataset) and len(features_dataset) == len(rutas_dataset):
                st.info(f"Cargando {len(features_dataset)} características desde caché.")
                return features_dataset, nombres_cacheados
        except Exception:
            # Si falla la carga del cache (archivo corrupto), reinicia la base de datos de características
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
        SAVE_INTERVAL = 15 # Guardar el progreso cada 25 lotes
        total_lotes = (len(rutas_a_procesar) + BATCH_SIZE - 1) // BATCH_SIZE
        
        # 🚀 MODIFICACIÓN: Texto simplificado para el progreso del lote.
        progress_text_template = "Procesando Lote {current_batch}/{total_lotes} (Extrayendo Features...)"
        
        # NOTA: Usamos st.empty() para poner la barra sin el texto "Running..." de Streamlit.
        progress_slot = st.empty()
        progress_bar = progress_slot.progress(0, text=progress_text_template.format(current_batch=0, total_lotes=total_lotes))


        # Lista temporal para acumular features procesadas en este RUN
        features_a_procesar = []
        
        for i in range(0, len(rutas_a_procesar), BATCH_SIZE):
            batch_num_actual = (i // BATCH_SIZE) + 1
            
            # Actualizamos el texto de la barra de progreso
            progress_bar.progress(batch_num_actual / total_lotes, text=progress_text_template.format(current_batch=batch_num_actual, total_lotes=total_lotes))

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
            
            # Acumulamos las features procesadas
            features_a_procesar.append(batch_features)
            nombres_cacheados.extend(valid_batch_names)
            
            # 💾 GUARDADO INCREMENTAL
            if batch_num_actual % SAVE_INTERVAL == 0 or batch_num_actual == total_lotes:
                
                # Consolidar las nuevas features procesadas con las ya cargadas (si las hay)
                features_consolidadas = np.vstack([features_dataset] + features_a_procesar)
                
                # Guardar el estado actual en disco
                np.save(FEATURES_FILE, features_consolidadas)
                np.save(NAMES_FILE, nombres_cacheados)

                # Actualizar features_dataset con la versión completa y limpiar lista temporal
                features_dataset = features_consolidadas
                features_a_procesar = []
                st.info(f"💾 Progreso guardado: {len(nombres_cacheados)}/{len(rutas_dataset)} imágenes procesadas.")

        progress_bar.empty()

    return features_dataset, nombres_cacheados

@st.cache_data
def cargar_indice_faiss(features_dataset):
    """
    Crea y entrena el índice Faiss para la búsqueda de vecinos más cercanos.
    """
    if features_dataset is None:
        return None, None
    
    D = features_dataset.shape[1] # Dimensión del vector (2048)
    data = features_dataset.astype('float32')
    
    # --- Índice Euclidiana (L2) ---
    index_euc = faiss.IndexFlatL2(D)
    index_euc.add(data) 
    
    # --- Índice Coseno (Producto Interno - IP) ---
    faiss.normalize_L2(data)
    index_cos = faiss.IndexFlatIP(D)
    index_cos.add(data)
    
    return index_euc, index_cos

def prepare_image(img_input, target_size=(224, 224)): 
    # Abre la imagen usando PIL (funciona con uploaded_file)
    img = Image.open(img_input)
    # Asegúrate de que el objeto PIL esté en el tamaño objetivo
    img = img.resize(target_size) 
    
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)

def _extract_features(img_input, model): # extrae características de una imagen
    try:
        img_preprocessed = prepare_image(img_input)
        features = model.predict(img_preprocessed, verbose=0)
        return features.flatten()
    except Exception as e:
        st.error(f"Error al procesar la imagen de consulta. Asegúrate de que es un archivo de imagen válido: {e}")
        return None

# --- Funciones de Búsqueda ---

def buscar_vecinos_brute_force(features_dataset, features_query, nombres_dataset, 
                                 radio_euc, radio_cos, top_k=10):
    """
    Método de búsqueda original (Fuerza Bruta/ResNet).
    """
    
    dist_euc_all = euclidean_distances([features_query], features_dataset)[0]
    dist_cos_all = cosine_distances([features_query], features_dataset)[0]
    
    nombres_dataset = np.array(nombres_dataset)
    
    dist_euc_filtered = dist_euc_all
    nombres_euc_filtered = nombres_dataset
    dist_cos_filtered = dist_cos_all
    nombres_cos_filtered = nombres_dataset
    
    # Procesar Resultados (Euclidiana)
    idx_euc_sorted = np.argsort(dist_euc_filtered)
    resultados_euc = []
    for i in range(len(idx_euc_sorted)):
        idx = idx_euc_sorted[i]
        dist = dist_euc_filtered[idx]
        if dist > radio_euc: break
        resultados_euc.append({'nombre': nombres_euc_filtered[idx], 'dist': dist})
        if len(resultados_euc) >= top_k: break

    # Procesar Resultados (Coseno)
    idx_cos_sorted = np.argsort(dist_cos_filtered)
    resultados_cos = []
    for i in range(len(idx_cos_sorted)):
        idx = idx_cos_sorted[i]
        dist = dist_cos_filtered[idx]
        if dist > radio_cos: break
        resultados_cos.append({'nombre': nombres_cos_filtered[idx], 'dist': dist})
        if len(resultados_cos) >= top_k: break

    return {'euc': resultados_euc, 'cos': resultados_cos}


def buscar_vecinos_faiss(features_dataset, features_query, nombres_dataset, 
                          radio_euc, radio_cos, top_k=10, index_euc=None, index_cos=None):
    """
    Método de búsqueda indexada Faiss.
    """
    if index_euc is None or index_cos is None:
        st.error("Error: Índices Faiss no cargados. Intenta recargar la aplicación.")
        return {'euc': [], 'cos': []}
        
    query_vector = features_query.astype('float32').reshape(1, -1)
    K_MAX = len(nombres_dataset) 

    # --- Búsqueda Euclidiana (L2 Index) ---
    D_euc, I_euc = index_euc.search(query_vector, K_MAX)
    
    resultados_euc = []
    for dist_squared, idx in zip(D_euc[0], I_euc[0]):
        # Se aplica la raíz cuadrada (sqrt) para obtener la distancia L2 
        dist = np.sqrt(dist_squared)
        
        if dist > radio_euc: break
        resultados_euc.append({'nombre': nombres_dataset[idx], 'dist': dist})
        if len(resultados_euc) >= top_k: break

    # --- Búsqueda Coseno (IP Index) ---
    faiss.normalize_L2(query_vector)
    
    D_cos, I_cos = index_cos.search(query_vector, K_MAX)
    
    resultados_cos = []
    for sim, idx in zip(D_cos[0], I_cos[0]):
        dist = max(0, 1 - sim) 

        if dist > radio_cos: break
        resultados_cos.append({'nombre': nombres_dataset[idx], 'dist': dist})
        if len(resultados_cos) >= top_k: break
        
    return {'euc': resultados_euc, 'cos': resultados_cos}


# Se modifica la función para aceptar los datos de log y el nombre del motor
def guardar_consulta_y_resultados(query_image, query_name, resultados, subfolder_name, engine_text, log_data):
    """
    Función de persistencia: genera la carpeta de la consulta, guarda el CSV con resultados
    unificados (Euclidiana y Coseno) y los mosaicos de imágenes, además de un log.
    """
    
    query_dir = os.path.join(RESULTS_DIR, subfolder_name)
    os.makedirs(query_dir, exist_ok=True)
    
    query_image.save(os.path.join(query_dir, f"consulta_{query_name}.jpg"))
    
    # --- LOGGING (Fix de codificación y contenido) ---
    log_file = os.path.join(query_dir, "metadata_log.txt")
    
    # FIX DE CODIFICACIÓN: Se añade encoding='utf-8' al abrir el archivo.
    with open(log_file, 'w', encoding='utf-8') as f: 
        f.write(f"--- LOG DE CONSULTA ---\n")
        f.write(f"Fecha y Hora: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Dataset Utilizado: {DATASET_NAME}\n")
        f.write(f"Imagen Consultada: {query_name}\n")
        f.write(f"Motor de Búsqueda: {engine_text}\n")
        f.write(f"Tiempo de Ejecución: {log_data['Tiempo_Ejecucion_s']:.4f} segundos\n")
        f.write(f"Radio Euclidiana Usado: {log_data['Radio_Euclidiana']}\n")
        f.write(f"Resultados Euc Encontrados: {log_data['Total_Euc']}/10\n")
        f.write(f"Radio Coseno Usado: {log_data['Radio_Coseno']}\n")
        f.write(f"Resultados Cos Encontrados: {log_data['Total_Cos']}/10\n")
        
    max_len = max(len(resultados['euc']), len(resultados['cos']))
    
    data = []
    for i in range(max_len):
        row = {'Puesto': i + 1}
        
        nombre_euc = 'N/A'
        nombre_cos = 'N/A'
        
        if i < len(resultados['euc']):
            nombre_euc = resultados['euc'][i]['nombre']
            row['Nombre_Archivo_Euc'] = nombre_euc
            row['Distancia_Euc'] = resultados['euc'][i]['dist']
        else:
            row['Nombre_Archivo_Euc'] = 'N/A'
            row['Distancia_Euc'] = 'N/A'
            
        if i < len(resultados['cos']):
            nombre_cos = resultados['cos'][i]['nombre']
            row['Nombre_Archivo_Cos'] = nombre_cos
            row['Distancia_Cos'] = resultados['cos'][i]['dist']
        else:
            row['Nombre_Archivo_Cos'] = 'N/A'
            row['Distancia_Cos'] = 'N/A'
        
        if nombre_euc != 'N/A' and nombre_cos != 'N/A':
            row['Unanime'] = 1 if nombre_euc == nombre_cos else 0
        else:
            row['Unanime'] = 0 
            
        data.append(row)
        
    df = pd.DataFrame(data)
    
    csv_file = os.path.join(query_dir, f"resultados_unificados.csv")
    df.to_csv(csv_file, index=False)
    
    # --- Mejora de Imágenes Generadas (Punto 3) ---
    def generar_imagen_consolidada(metric_key, metric_results, metric_name):
        if not metric_results: return
        
        IMG_SIZE = 150
        IMG_SPACING = 10 
        INFO_HEIGHT = 45 
        TITLE_HEIGHT = 80 
        
        rows = 3
        cols = 4
        
        ancho_final = 4 * IMG_SIZE + (cols + 1) * IMG_SPACING
        alto_final = TITLE_HEIGHT + rows * IMG_SIZE + rows * INFO_HEIGHT + (rows + 1) * IMG_SPACING
        
        img_compuesta = Image.new('RGB', (ancho_final, alto_final), color='white')
        draw = ImageDraw.Draw(img_compuesta)
        
        try:
            # Fuentes de archivo de sistema (deberían estar disponibles)
            font_title = ImageFont.truetype("arial.ttf", 20)
            font_subtitle = ImageFont.truetype("arial.ttf", 18)
            font_info = ImageFont.truetype("arial.ttf", 15)
        except IOError:
            # Fallback a fuentes por defecto si las de sistema no se encuentran
            font_title = ImageFont.load_default()
            font_subtitle = ImageFont.load_default()
            font_info = ImageFont.load_default()
            
        # Título principal y subtítulo (motor)
        main_title = f"10 imágenes más similares a {query_name}"
        sub_title = f"(Motor: {engine_text} | Distancia: {metric_name})"
        
        text_w_main, _ = draw.textbbox((0, 0), main_title, font=font_title)[2:]
        text_w_sub, _ = draw.textbbox((0, 0), sub_title, font=font_subtitle)[2:]
        
        draw.text(((ancho_final - text_w_main) / 2, IMG_SPACING), main_title, fill='black', font=font_title)
        draw.text(((ancho_final - text_w_sub) / 2, IMG_SPACING + 25), sub_title, fill='gray', font=font_subtitle)

        # Línea de separación
        draw.line([(0, TITLE_HEIGHT - IMG_SPACING/2), (ancho_final, TITLE_HEIGHT - IMG_SPACING/2)], fill='gray', width=1)

        y_offset = TITLE_HEIGHT
        
        # Dibuja la Consulta (Posición 0,0)
        q_img_resized = query_image.resize((IMG_SIZE, IMG_SIZE))
        img_compuesta.paste(q_img_resized, (IMG_SPACING, y_offset + IMG_SPACING))
        draw.text((IMG_SPACING, y_offset + IMG_SIZE + IMG_SPACING), f"CONSULTA:", fill='black', font=font_info)
        draw.text((IMG_SPACING, y_offset + IMG_SIZE + 18 + IMG_SPACING), f"{query_name}", fill='black', font=font_info)
        
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
                
                text_line_1 = f"#{puesto} (Dist: {res['dist']:.2f})"
                
                # Usar dos líneas de texto para evitar overflow del nombre de archivo.
                draw.text((x_start, y_start + IMG_SIZE), text_line_1, fill='black', font=font_info)
                draw.text((x_start, y_start + IMG_SIZE + 18), f"Archivo: {res['nombre']}", fill='black', font=font_info)
                
            except FileNotFoundError:
                draw.rectangle([x_start, y_start, x_start + IMG_SIZE, y_start + IMG_SIZE], fill="gray")
                draw.text((x_start, y_start + IMG_SIZE), f"NO ENCONTRADO", fill='black', font=font_info)
        
        img_compuesta.save(os.path.join(query_dir, f"imagen_consolidada_{metric_key}.jpg"))
        
    generar_imagen_consolidada('euclidiana', resultados['euc'], 'Euclidiana')
    generar_imagen_consolidada('coseno', resultados['cos'], 'Coseno')
    
    return query_dir

# --- INTERFAZ PRINCIPAL ---

STYLING_CSS = """
<style>
h1 {
    text-align: center;
    font-family: 'Times New Roman', Times, serif; 
    font-weight: 700;
    color: #333333;
    border-bottom: 2px solid #DDDDDD;
    padding-bottom: 10px;
}

div.stButton > button {
    display: block;
    margin: 0 auto;
    width: 100%; 
    max-width: 250px; 
    background-color: #4CAF50; 
    border-radius: 8px;
}

h2 {
    color: #555555;
    border-left: 5px solid #007bff;
    padding-left: 10px;
    margin-top: 10px;
    margin-bottom: 15px;
}

.stImage {
    margin-bottom: -15px;
}

.stNumberInput {
    margin-bottom: 15px;
}
</style>
"""
st.markdown(STYLING_CSS, unsafe_allow_html=True)
st.set_page_config(page_title="Buscador de Carátulas", layout="wide") 

st.title("Buscador de Carátulas de Álbumes Similares")

feature_extractor = cargar_modelo()

with st.spinner(f'Cargando base de datos ({DATASET_NAME})...'):
    features_dataset, nombres_dataset = cargar_caracteristicas_dataset(feature_extractor)
    
    # Es vital verificar que features_dataset no esté vacío antes de cargar FAISS
    if features_dataset is not None and len(features_dataset) > 0:
        index_euc_faiss, index_cos_faiss = cargar_indice_faiss(features_dataset)
    else:
        index_euc_faiss, index_cos_faiss = None, None


if features_dataset is None or not nombres_dataset or len(features_dataset) == 0:
    st.error("No se pudo cargar la base de datos. Revisa la consola o borra los archivos .npy si están corruptos.")
else:
    st.success(f"Base de datos lista: {len(nombres_dataset)} imágenes cargadas desde {DATASET_NAME}.")
    st.markdown("---")
    
    query_features = None
    query_image = None
    query_name = ""

    # --- Selector de Motor de Búsqueda ---
    search_engine = st.selectbox(
        "Selecciona el Motor de Búsqueda:",
        ("Fuerza Bruta (ResNet)", "Faiss (Indexado)"),
        key="search_engine_select"
    )
    st.markdown("---")
    
    # Lógica de carga de archivo 
    uploaded_file = st.file_uploader(
        "Sube una imagen de consulta aquí:",
        type=["jpg", "jpeg", "png"]
    )
    
    if uploaded_file is not None:
        
        # Lógica de resetear la sesión si se sube un nuevo archivo.
        if 'uploaded_file_name' not in st.session_state or st.session_state.uploaded_file_name != uploaded_file.name:
            # Si el archivo es nuevo, limpiamos los resultados de la búsqueda anterior.
            if 'resultado' in st.session_state:
                del st.session_state['resultado']
            st.session_state.uploaded_file_name = uploaded_file.name

        # 1. Cargar imagen PIL y nombre 
        try:
            query_image = Image.open(uploaded_file)
            query_name = uploaded_file.name
        except Exception as e:
            st.error(f"Error al abrir la imagen: {e}")
            st.stop() 

        # 2. Extraer Features
        uploaded_file.seek(0)
        query_features = _extract_features(uploaded_file, feature_extractor)
        
        # --- DETENER EJECUCIÓN SI FALLA LA EXTRACCIÓN ---
        if query_features is None:
            st.stop()
        
        # 3. Dibujar Interfaz de Controles
        col_img, col_controls = st.columns([1, 1.5])
        
        with col_img:
            st.image(query_image, caption='Imagen de consulta', width=250)
        
        with col_controls:
            radio_euc = st.number_input("Radio de Búsqueda (Euclidiana):", min_value=0.0, value=30.0, step=0.5, key="euc_carga")
            radio_cos = st.number_input("Radio de Búsqueda (Coseno):", min_value=0.0, max_value=2.0, value=0.45, step=0.01, key="cos_carga")
            
            st.markdown('<div style="height: 15px;"></div>', unsafe_allow_html=True)
            
            # --- Lógica de Búsqueda (Se ejecuta al presionar el botón) ---
            if st.button('Buscar imágenes similares', type="primary", key="btn_carga"):
                
                start_time = time.perf_counter() 
                
                # Usar el valor actual del selectbox
                current_engine = st.session_state.search_engine_select

                with st.spinner(f'Buscando con {current_engine}...'):
                    # Lógica de despacho de la búsqueda
                    if current_engine == "Faiss (Indexado)":
                        resultado = buscar_vecinos_faiss(
                            features_dataset, 
                            query_features, 
                            nombres_dataset,
                            radio_euc,
                            radio_cos,
                            index_euc=index_euc_faiss,
                            index_cos=index_cos_faiss
                        )
                    else: # Fuerza Bruta (ResNet)
                        resultado = buscar_vecinos_brute_force(
                            features_dataset, 
                            query_features, 
                            nombres_dataset,
                            radio_euc,
                            radio_cos
                        )
                    
                end_time = time.perf_counter()
                elapsed_time = end_time - start_time
                
                # Se almacena el resultado en la sesión
                st.session_state['elapsed_time'] = elapsed_time
                st.session_state['resultado'] = resultado
                st.session_state['query_info'] = (query_image, query_name, radio_euc, radio_cos)
                st.session_state['search_engine'] = current_engine # Usar el motor actual
                
                # Forzar el renderizado para mostrar los resultados abajo
                st.rerun()

        # --- LÓGICA DE RESULTADOS (Se ejecuta si ya se hizo una búsqueda) ---
        if 'resultado' in st.session_state:
            
            resultado = st.session_state['resultado']
            query_image, query_name, radio_euc, radio_cos = st.session_state['query_info']
            used_engine = st.session_state['search_engine']
            elapsed_time = st.session_state['elapsed_time']
            
            
            # --- NOMENCLATURA (Orden: Imagen_Metodo_Timestamp) ---
            if used_engine == "Faiss (Indexado)":
                engine_suffix = "indexado" 
                engine_text = "Faiss (Indexado)"
            else:
                engine_suffix = "fuerzabruta" 
                engine_text = "Fuerza Bruta"

            # Se genera el nombre de la carpeta en el orden deseado
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            clean_query_name = query_name.split('.')[0] 
            # Formato: consulta_NOMBRE_METODO_FECHA
            subfolder_name = f"consulta_{clean_query_name}_{engine_suffix}_{timestamp}"
            
            # Preparar datos para el log
            log_data = {
                'Tiempo_Ejecucion_s': elapsed_time,
                'Motor_Busqueda': used_engine,
                'Radio_Euclidiana': radio_euc,
                'Radio_Coseno': radio_cos,
                'Total_Euc': len(resultado['euc']),
                'Total_Cos': len(resultado['cos'])
            }
            
            query_dir = guardar_consulta_y_resultados(query_image, query_name, resultado, subfolder_name, engine_text, log_data) 
            
            st.markdown("---")
            st.success(f"Resultados guardados en: **{query_dir}** (Motor: {engine_text})")
            st.info(f"Tiempo de Búsqueda: **{elapsed_time:.4f} segundos**")
            st.subheader('Resultados de la Búsqueda')
            
            
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

            render_results("Euclidiana", resultado['euc'], radio_euc)
            st.markdown("---") 

            render_results("Coseno", resultado['cos'], radio_cos)