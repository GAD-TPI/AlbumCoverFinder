# Script para ejecutar las consultas de `data/consulta`
import os
import sys
import traceback
import argparse
import datetime

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

print(f"Working dir: {ROOT}")

try:
    import app
except Exception as e:
    print("Error importando app.py:", e)
    traceback.print_exc()
    raise

# Inicializar recursos si no existen como globals en app
try:
    if not hasattr(app, 'feature_extractor') or app.feature_extractor is None:
        print('Cargando modelo...')
        app.feature_extractor = app.cargar_modelo()
    else:
        print('Modelo ya cargado en app.feature_extractor')

    if not hasattr(app, 'features_dataset') or app.features_dataset is None or len(getattr(app, 'features_dataset')) == 0:
        print('Cargando/obteniendo características del dataset...')
        features_dataset, nombres_dataset = app.cargar_caracteristicas_dataset(app.feature_extractor)
        app.features_dataset = features_dataset
        app.nombres_dataset = nombres_dataset
    else:
        print('Features ya presentes en app.features_dataset')

    if not (hasattr(app, 'index_euc_faiss') and app.index_euc_faiss is not None):
        print('Construyendo índices Faiss...')
        app.index_euc_faiss, app.index_cos_faiss = app.cargar_indice_faiss(app.features_dataset)
    else:
        print('Índices Faiss ya presentes')
        
    # --- Implementación de ejecutar_consultas_batch MOVIDA a este script ---
    def ejecutar_consultas(carpeta_consulta=None, subfolder_root='consulta_50k', engine='Faiss',
                                top_k=10, radio_euc=100.0, radio_cos=0.45, overwrite=False, max_images=None):
        """
        Ejecuta consultas para cada imagen encontrada en `carpeta_consulta` (recursivo)
        y guarda los resultados dentro de `results/<subfolder_root>/...` usando
        la función `guardar_consulta_y_resultados` definida en `app`.
        Esta versión referencia objetos y funciones en el módulo `app`.
        """
        if carpeta_consulta is None:
            carpeta_consulta = os.path.join(app.DATA_DIR, 'consulta')

        os.makedirs(app.RESULTS_DIR, exist_ok=True)
        target_root = os.path.join(app.RESULTS_DIR, subfolder_root)
        os.makedirs(target_root, exist_ok=True)

        rutas_absolutas, _ = app.cargar_imagenes_dataset(carpeta_consulta)
        saved_dirs = []

        if not rutas_absolutas:
            print(f"No se encontraron imágenes en la carpeta de consultas: {carpeta_consulta}")
            return saved_dirs

        contador = 0
        for img_path in rutas_absolutas:
            if max_images is not None and contador >= max_images:
                break

            try:
                query_name = os.path.basename(img_path)
                query_image = app.Image.open(img_path)

                start_t = app.time.perf_counter()
                features_query = app._extract_features(img_path, app.feature_extractor)
                if features_query is None:
                    continue

                # Ejecutar ambos motores: Faiss (si está disponible) y Fuerza Bruta
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                clean_query_name = os.path.splitext(query_name)[0]

                # 1) Faiss (Indexado) - si está disponible
                if hasattr(app, 'index_euc_faiss') and app.index_euc_faiss is not None:
                    try:
                        start_f = app.time.perf_counter()
                        resultados_f = app.buscar_vecinos_faiss(
                            app.features_dataset,
                            features_query,
                            app.nombres_dataset,
                            radio_euc,
                            radio_cos,
                            top_k=top_k,
                            index_euc=app.index_euc_faiss,
                            index_cos=app.index_cos_faiss
                        )
                        end_f = app.time.perf_counter()
                        elapsed_f = end_f - start_f

                        log_data_f = {
                            'Tiempo_Ejecucion_s': elapsed_f,
                            'Motor_Busqueda': 'Faiss (Indexado)',
                            'Radio_Euclidiana': radio_euc,
                            'Radio_Coseno': radio_cos,
                            'Total_Euc': len(resultados_f.get('euc', [])),
                            'Total_Cos': len(resultados_f.get('cos', []))
                        }

                        subfolder_name_f = os.path.join(subfolder_root, f"consulta_{clean_query_name}_indexado_{timestamp}")
                        saved_dir_f = app.guardar_consulta_y_resultados(query_image, query_name, resultados_f, subfolder_name_f, 'Faiss (Indexado)', log_data_f)
                        saved_dirs.append(saved_dir_f)
                        print(f"Guardada consulta Faiss: {query_name} -> {saved_dir_f}")
                    except Exception as e:
                        print(f"Error Faiss procesando {img_path}: {e}")

                # 2) Fuerza Bruta
                try:
                    start_b = app.time.perf_counter()
                    resultados_b = app.buscar_vecinos_brute_force(
                        app.features_dataset,
                        features_query,
                        app.nombres_dataset,
                        radio_euc,
                        radio_cos,
                        top_k=top_k
                    )
                    end_b = app.time.perf_counter()
                    elapsed_b = end_b - start_b

                    log_data_b = {
                        'Tiempo_Ejecucion_s': elapsed_b,
                        'Motor_Busqueda': 'Fuerza Bruta',
                        'Radio_Euclidiana': radio_euc,
                        'Radio_Coseno': radio_cos,
                        'Total_Euc': len(resultados_b.get('euc', [])),
                        'Total_Cos': len(resultados_b.get('cos', []))
                    }

                    subfolder_name_b = os.path.join(subfolder_root, f"consulta_{clean_query_name}_fuerzabruta_{timestamp}")
                    saved_dir_b = app.guardar_consulta_y_resultados(query_image, query_name, resultados_b, subfolder_name_b, 'Fuerza Bruta', log_data_b)
                    saved_dirs.append(saved_dir_b)
                    print(f"Guardada consulta Fuerza Bruta: {query_name} -> {saved_dir_b}")
                except Exception as e:
                    print(f"Error Fuerza Bruta procesando {img_path}: {e}")

                contador += 1

            except Exception as e:
                print(f"Error procesando {img_path}: {e}")
                continue

        return saved_dirs

    # Parsear argumentos CLI para mayor flexibilidad
    parser = argparse.ArgumentParser(description='Ejecutar batch de consultas sobre data/consulta')
    parser.add_argument('--radio_euc', type=float, default=100.0, help='Radio Euclidiana (default: 100.0)')
    parser.add_argument('--radio_cos', type=float, default=0.45, help='Radio Coseno (default: 0.45)')
    parser.add_argument('--top_k', type=int, default=10, help='Número máximo de resultados por métrica (default: 10)')
    parser.add_argument('--max_images', type=int, default=25, help='Máximo de imágenes a procesar (default: 25)')
    parser.add_argument('--engine', type=str, default='Faiss', help="Motor: 'Faiss' o 'Brute' (default: Faiss)")
    parser.add_argument('--subfolder_root', type=str, default='consulta_50k', help='Carpeta raíz dentro de results (default: consulta_50k)')

    args, _ = parser.parse_known_args()

    print(f"Lanzando procesamiento batch (radio_euc={args.radio_euc}, radio_cos={args.radio_cos}, engine={args.engine})...")
    resultado_dirs = ejecutar_consultas(carpeta_consulta=os.path.join(app.DATA_DIR, 'consulta'),
                                              subfolder_root=args.subfolder_root,
                                              engine=args.engine,
                                              top_k=args.top_k,
                                              radio_euc=args.radio_euc,
                                              radio_cos=args.radio_cos,
                                              overwrite=False,
                                              max_images=args.max_images)

    print('Procesamiento finalizado. Directorios creados:')
    for d in resultado_dirs:
        print(d)

except Exception as e:
    print('Error durante el procesamiento batch:')
    traceback.print_exc()
    raise
