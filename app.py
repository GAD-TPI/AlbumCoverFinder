# CÓDIGO FINAL CON CORRECCIONES DE TEXTO Y LÓGICA FAISS

import os
import logging
import datetime
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

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
DATASET_PATH = os.path.join(DATA_DIR, 'dataset') 
FEATURES_FILE = os.path.join(FEATURES_DIR, 'dataset_features.npy')
NAMES_FILE = os.path.join(FEATURES_DIR, 'dataset_names.npy') 

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
            
        if temp_features_list:
            features_dataset = np.vstack([features_dataset] + temp_features_list)
            nombres_cacheados.extend(temp_names_list)
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
    for dist, idx in zip(D_euc[0], I_euc[0]):
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
        
    # --- CORRECCIÓN DEL BUG ---
    return {'euc': resultados_euc, 'cos': resultados_cos}


# --- Funciones de Persistencia y Renderizado (Se mantienen iguales) ---

def guardar_consulta_y_resultados(query_image, query_name, resultados, subfolder_name):
    # Guarda la imagen de consulta, los resultados en un CSV unificado, y las imágenes consolidadas
    
    query_dir = os.path.join(RESULTS_DIR, subfolder_name)
    os.makedirs(query_dir, exist_ok=True)
    
    query_image.save(os.path.join(query_dir, f"consulta_{query_name}.jpg"))
    
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
    
    def generar_imagen_consolidada(metric_key, metric_results, metric_name):
        if not metric_results: return
        
        IMG_SIZE = 150
        IMG_SPACING = 5
        INFO_HEIGHT = 40
        TITLE_HEIGHT = 60 
        
        rows = 3
        cols = 4
        
        ancho_final = 4 * IMG_SIZE + (cols + 1) * IMG_SPACING
        alto_final = TITLE_HEIGHT + rows * IMG_SIZE + rows * INFO_HEIGHT + (rows + 1) * IMG_SPACING
        
        img_compuesta = Image.new('RGB', (ancho_final, alto_final), color='white')
        draw = ImageDraw.Draw(img_compuesta)
        
        try:
            font_title = ImageFont.truetype("arial.ttf", 24) # Aumentado un poco el tamaño del título
            font_info = ImageFont.truetype("arial.ttf", 16)
        except IOError:
            font_title = ImageFont.load_default()
            font_info = ImageFont.load_default()
            
        title_text = f"10 imágenes más similares a {query_name} (Distancia {metric_name})"
        text_w, text_h = draw.textbbox((0, 0), title_text, font=font_title)[2:]
        draw.text(((ancho_final - text_w) / 2, IMG_SPACING), title_text, fill='black', font=font_title)
        draw.line([(0, TITLE_HEIGHT - IMG_SPACING), (ancho_final, TITLE_HEIGHT - IMG_SPACING)], fill='gray', width=1)

        y_offset = TITLE_HEIGHT
        
        q_img_resized = query_image.resize((IMG_SIZE, IMG_SIZE))
        img_compuesta.paste(q_img_resized, (IMG_SPACING, y_offset + IMG_SPACING))
        draw.text((IMG_SPACING, y_offset + IMG_SIZE + IMG_SPACING), f"CONSULTA: {query_name}", fill='black', font=font_info)
        
        for i, res in enumerate(metric_results):
            puesto = i + 1
            
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

with st.spinner('Cargando base de datos...'):
    features_dataset, nombres_dataset = cargar_caracteristicas_dataset(feature_extractor)
    
    # --- Cargar Índices FAISS (Solo si el dataset está cargado) ---
    index_euc_faiss, index_cos_faiss = cargar_indice_faiss(features_dataset)


if features_dataset is None or not nombres_dataset:
    st.error("No se pudo cargar la base de datos. Revisa la consola.")
else:
    st.success(f"Base de datos lista: {len(nombres_dataset)} imágenes cargadas.")
    st.markdown("---")
    
    query_features = None
    query_image = None
    query_name = ""

    # --- Selector de Motor de Búsqueda ---
    search_engine = st.selectbox(
        "Selecciona el Motor de Búsqueda:",
        # TEXTO CORREGIDO
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
            if st.button('Buscar imágenes similares', type="primary", key="btn_carga"):
                
                with st.spinner(f'Buscando con {search_engine}...'):
                    # Lógica de despacho de la búsqueda
                    if search_engine == "Faiss (Indexado)":
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
                    
                    st.session_state['resultado'] = resultado
                    st.session_state['query_info'] = (query_image, query_name, radio_euc, radio_cos)
                    st.session_state['search_engine'] = search_engine 
                    st.session_state['run_search'] = True
                
                st.rerun()


    # --- LÓGICA DE RESULTADOS (MOVIMIENTO HACIA ABAJO) ---

    if 'run_search' not in st.session_state:
        st.session_state['run_search'] = False

    if st.session_state['run_search']:
        
        resultado = st.session_state['resultado']
        query_image, query_name, radio_euc, radio_cos = st.session_state['query_info']
        used_engine = st.session_state['search_engine']
        
        # --- NOMENCLATURA: Asignar prefijo R o I ---
        if used_engine == "Faiss (Indexado)":
            engine_prefix = "I" # I = Indexado
            engine_text = "Faiss"
        else:
            engine_prefix = "R" # R = Referencia/ResNet (Brute Force)
            engine_text = "ResNet"

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        clean_query_name = query_name.split('.')[0] 
        # Formato: consulta_R/I_NOMBRE_FECHA
        subfolder_name = f"consulta_{engine_prefix}_{clean_query_name}_{timestamp}"
        
        query_dir = guardar_consulta_y_resultados(query_image, query_name, resultado, subfolder_name)
        
        st.markdown("---")
        st.success(f"Resultados guardados en: {query_dir} (Motor: {engine_text})")
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