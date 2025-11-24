# Script para ejecutar en batch las consultas de `data/consulta`
import os
import sys
import traceback
import argparse

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

        # Parsear argumentos CLI para mayor flexibilidad
        parser = argparse.ArgumentParser(description='Ejecutar batch de consultas sobre data/consulta')
        parser.add_argument('--radio_euc', type=float, default=100.0, help='Radio Euclidiana (default: 100.0)')
        parser.add_argument('--radio_cos', type=float, default=0.45, help='Radio Coseno (default: 0.45)')
        parser.add_argument('--top_k', type=int, default=10, help='Número máximo de resultados por métrica (default: 10)')
        parser.add_argument('--max_images', type=int, default=None, help='Máximo de imágenes a procesar (default: todas)')
        parser.add_argument('--engine', type=str, default='Faiss', help="Motor: 'Faiss' o 'Brute' (default: Faiss)")
        parser.add_argument('--subfolder_root', type=str, default='consulta_50k', help='Carpeta raíz dentro de results (default: consulta_50k)')

        args, _ = parser.parse_known_args()

        print(f"Lanzando procesamiento batch (radio_euc={args.radio_euc}, radio_cos={args.radio_cos}, engine={args.engine})...")
        resultado_dirs = app.ejecutar_consultas_batch(carpeta_consulta=os.path.join(app.DATA_DIR, 'consulta'),
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
