# -*- coding: utf-8 -*-
"""
Script para la Búsqueda de Imágenes Similares (CBIR) usando ResNet50.

Este script realiza las siguientes operaciones:
1. Carga imágenes de un dataset local y de una carpeta de consulta.
2. Utiliza un modelo ResNet50 pre-entrenado para extraer vectores de características.
3. Almacena en caché (guarda en disco) las características del dataset para
   evitar re-procesarlas.
4. Compara las imágenes de consulta con el dataset usando distancias
   Euclidiana y Coseno.
5. Guarda los resultados (un gráfico de comparación por consulta) en una
   carpeta única por ejecución.
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from keras._tf_keras.keras.preprocessing import image
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.models import Model
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
from datetime import datetime
# Se eliminó shutil, ya que no se copiarán las imágenes, solo se guardará el plot.

# --- Configuración de Rutas ---
# ./search.py
# ./data/dataset/
# ./data/query/
# ./features/
# ./results/ (se creará para guardar las carpetas de ejecución)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
FEATURES_DIR = os.path.join(BASE_DIR, 'features')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

DATASET_PATH = os.path.join(DATA_DIR, 'dataset')
QUERY_PATH = os.path.join(DATA_DIR, 'query')
FEATURES_FILE = os.path.join(FEATURES_DIR, 'dataset_features.npy')
NAMES_FILE = os.path.join(FEATURES_DIR, 'dataset_names.npy')

# --- Modelo Global ---
try:
    print("Cargando modelo ResNet50...")
    base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    feature_extractor = Model(inputs=base_model.input, outputs=base_model.output)
    print("Modelo ResNet50 cargado exitosamente.")
except Exception as e:
    print(f"Error fatal al cargar el modelo: {e}")
    print("Asegúrese de tener tensorflow instalado y conexión a internet la primera vez.")
    exit()

# --- Definición de Funciones ---

def cargar_imagenes(carpeta):
    """Carga rutas de imágenes y sus nombres de una carpeta."""
    rutas = []
    for ext in ('*.jpg', '*.jpeg', '*.png', '*.bmp'):
        rutas.extend(glob.glob(os.path.join(carpeta, ext)))
    nombres = [os.path.basename(r) for r in rutas]
    return rutas, nombres

def prepare_image(image_path, target_size=(224, 224)):
    """Carga y pre-procesa una imagen para ResNet50."""
    img = image.load_img(image_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)

def extract_features(image_path):
    """Extrae el vector de características de una imagen."""
    img_preprocessed = prepare_image(image_path)
    features = feature_extractor.predict(img_preprocessed, verbose=0)
    return features.flatten()

def buscar_vecino_mas_cercano(features_dataset, features_query, nombres_dataset):
    """Busca los vecinos más cercanos para un set de consultas."""
    resultados = []
    for i, f_query in enumerate(features_query):
        dist_euc = euclidean_distances([f_query], features_dataset)[0]
        dist_cos = cosine_distances([f_query], features_dataset)[0]
        idx_euc = np.argmin(dist_euc)
        idx_cos = np.argmin(dist_cos)

        resultados.append({
            'query_idx': i,
            'vecino_euc': nombres_dataset[idx_euc],
            'vecino_cos': nombres_dataset[idx_cos],
            'dist_euc': dist_euc[idx_euc],
            'dist_cos': dist_cos[idx_cos]
        })
    return resultados

# <--- INICIO BLOQUE MODIFICADO ---
def guardar_resultados(resultados, imagenes_query_rutas, nombres_query, run_dir_path):
    """
    Guarda un gráfico de comparación por cada consulta dentro de la
    carpeta de ejecución única.
    """
    print("\n--- Guardando Resultados ---")
    for r in resultados:
        query_name = nombres_query[r['query_idx']]
        query_path = imagenes_query_rutas[r['query_idx']]

        try:
            # 1. Cargar imágenes para el gráfico
            euc_path = os.path.join(DATASET_PATH, r['vecino_euc'])
            cos_path = os.path.join(DATASET_PATH, r['vecino_cos'])
            
            img_query = image.load_img(query_path)
            img_euc = image.load_img(euc_path)
            img_cos = image.load_img(cos_path)

        except FileNotFoundError as e:
            print(f"Advertencia: No se pudo cargar una imagen para visualización de {query_name}: {e}")
            continue
        except Exception as e:
            print(f"Error procesando {query_name}: {e}")
            continue

        # 2. Crear y guardar el gráfico de comparación
        plt.figure(figsize=(12, 5))
        plt.suptitle(f"Consulta: {query_name}", fontsize=14)

        # Subplot 1: Consulta
        plt.subplot(1, 3, 1)
        plt.imshow(img_query)
        plt.title("Consulta")
        plt.axis('off')

        # Subplot 2: Vecino Euclidiano
        plt.subplot(1, 3, 2)
        plt.imshow(img_euc)
        plt.title(f"Euclidiano ({r['dist_euc']:.2f})\n{r['vecino_euc']}")
        plt.axis('off')

        # Subplot 3: Vecino Coseno
        plt.subplot(1, 3, 3)
        plt.imshow(img_cos)
        plt.title(f"Coseno ({r['dist_cos']:.2f})\n{r['vecino_cos']}")
        plt.axis('off')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # 3. Guardar el gráfico en la carpeta de ejecución
        query_name_base = os.path.splitext(query_name)[0]
        save_filename = f"comparacion_{query_name_base}.png"
        save_path = os.path.join(run_dir_path, save_filename)
        
        plt.savefig(save_path)
        plt.close() # <-- Importante: cierra la figura para liberar memoria

        print(f"Resultado para '{query_name}' guardado en: {save_path}")
# <--- FIN BLOQUE MODIFICADO ---

# --- Lógica Principal ---

def main():
    # 1. Asegurar que existan los directorios base
    os.makedirs(FEATURES_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 1.b. Crear carpeta única para esta EJECUCIÓN <--- NUEVO
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    run_folder_name = f"consulta_{timestamp}"
    run_dir_path = os.path.join(RESULTS_DIR, run_folder_name)
    os.makedirs(run_dir_path, exist_ok=True)
    print(f"\nGuardando todos los resultados de esta ejecución en: {run_dir_path}")

    # 2. Cargar rutas de imágenes
    imagenes_dataset_rutas, nombres_dataset_actuales = cargar_imagenes(DATASET_PATH)
    imagenes_query_rutas, nombres_query = cargar_imagenes(QUERY_PATH)

    # Validar que se encontraron imágenes
    if not imagenes_dataset_rutas:
        print(f"Error: No se encontraron imágenes en {DATASET_PATH}. Saliendo.")
        return
    if not imagenes_query_rutas:
        print(f"Error: No se encontraron imágenes de consulta en {QUERY_PATH}. Saliendo.")
        return

    print(f"Imágenes en dataset: {len(imagenes_dataset_rutas)}")
    print(f"Imágenes de consulta: {len(imagenes_query_rutas)}")

    # 3. Lógica de extracción y caché de características del dataset
    features_dataset = np.empty((0, 2048))
    nombres_dataset_cacheados = []

    if os.path.exists(FEATURES_FILE) and os.path.exists(NAMES_FILE):
        print("Cargando características y nombres desde caché...")
        try:
            features_dataset = np.load(FEATURES_FILE)
            nombres_dataset_cacheados = np.load(NAMES_FILE).tolist()
            print(f"Se cargaron {len(nombres_dataset_cacheados)} características cacheadas.")
        except Exception as e:
            print(f"Advertencia: Error al cargar archivos de caché, se re-procesará todo: {e}")
            features_dataset = np.empty((0, 2048))
            nombres_dataset_cacheados = []
    else:
        print("No se encontraron archivos de caché. Se iniciará una nueva extracción.")

    nombres_dataset_set = set(nombres_dataset_cacheados)
    new_features = []
    new_names = []

    print("Verificando imágenes nuevas y extrayendo características...")
    for i, img_path in enumerate(imagenes_dataset_rutas):
        img_name = nombres_dataset_actuales[i]
        if img_name not in nombres_dataset_set:
            print(f"Extrayendo características para: {img_name}...")
            try:
                features = extract_features(img_path)
                new_features.append(features)
                new_names.append(img_name)
            except Exception as e:
                print(f"Error al procesar {img_name}: {e}")
    
    if new_features:
        features_dataset = np.vstack([features_dataset] + new_features)
        nombres_dataset_final = nombres_dataset_cacheados + new_names
        print(f"Se extrajeron características de {len(new_features)} imágenes nuevas.")
        
        try:
            np.save(FEATURES_FILE, features_dataset)
            np.save(NAMES_FILE, nombres_dataset_final)
            print("Se guardaron las características y nombres actualizados en caché.")
        except Exception as e:
            print(f"Error al guardar archivos de caché: {e}")
    else:
        nombres_dataset_final = nombres_dataset_cacheados
        print("No se encontraron imágenes nuevas para procesar.")

    # 4. Extracción de características de consulta
    print("Extrayendo características de las imágenes de consulta...")
    try:
        features_query = np.vstack([extract_features(p) for p in imagenes_query_rutas])
    except Exception as e:
        print(f"Error al extraer características de consulta: {e}")
        return

    # 5. Búsqueda del vecino más cercano
    print("Buscando vecinos más cercanos...")
    resultados = buscar_vecino_mas_cercano(features_dataset, features_query, nombres_dataset_final)

    # 6. Guardar resultados <--- MODIFICADO
    guardar_resultados(resultados, imagenes_query_rutas, nombres_query, run_dir_path)
    print("\nProceso de búsqueda finalizado.")


if __name__ == "__main__":
    main()