# Album Cover Finder - Búsqueda de Imágenes por Similitud

Este repositorio contiene el código fuente del **Trabajo Práctico Integrador** para la cátedra de **Gestión Avanzada de Datos**. El sistema implementa un motor de búsqueda de imágenes basado en contenido (CBIR) utilizando Redes Neuronales Convolucionales (ResNet50) y búsqueda vectorial optimizada (FAISS).

La aplicación permite cargar imágenes de portadas de álbumes y buscar otras similares en una base de datos determinada. 

## Información del Proyecto

**Institución:** Universidad Tecnológica Nacional - Facultad Regional Concepción del Uruguay  
**Carrera:** Ingeniería en Sistemas de Información  
**Materia:** Gestión Avanzada de Datos (2025)  

**Integrantes:**
* María Emilia Fernandez
* Salvador Tiguá

**Docentes:**
* Mg. Ing. Andrés J. Pascal
* Ing. Adrián Planas

---

## Instalación y Ejecución

Pasos a seguir para poner en marcha la aplicación en un entorno local.

### 1. Prerrequisitos
* Python 3.8 o superior.

### 2. Configuración del Entorno
Abrir una terminal en la carpeta raíz del proyecto y ejecutar los siguientes comandos:

#### 2.1 Crear el entorno virtual
```
python -m venv .venv
```

#### 2.2 Activar el entorno
En Windows:
```
.\.venv\Scripts\activate
```
En Mac/Linux:
```
source .venv/bin/activate
```

#### 2.3 Instalar las dependencias
```
pip install -r requirements.txt
```

#### 2.4 Ejecutar la Aplicación Web
```
streamlit run app.py
```

## Configuración del Dataset

Para que el sistema funcione, se deben cargar las imágenes sobre las cuales se realizará la búsqueda.

### Ubicación
El dataset debe ser una subcarpeta de /data. 
Se puede cambiar el nombre de la base de datos editando la variable DATASET_NAME en el archivo app.py. Por defecto es "dataset_80k".

### Formatos Soportados
Las imágenes pueden ser .jpg, .jpeg, .png o .bmp.

### Procesamiento Inicial
La primera vez que se ejecuta la aplicación el sistema escaneará la carpeta, procesará todas las imágenes y generará los vectores de características.

Este proceso puede tardar varios minutos dependiendo de la cantidad de imágenes y del hardware. 

Se generarán archivos .npy en la carpeta /features. Las ejecuciones subsiguientes serán instantáneas (a menos que se agreguen imágenes), pues leerán directamente estos archivos.

## Uso y Resultados

### Interfaz Web

1. Subir una imagen de consulta cargando un archivo.
2. Seleccionar el motor de búsqueda:
   * Fuerza Bruta: Comparación exacta contra todo el dataset.
   * Faiss (Indexado): Búsqueda aproximada de alta velocidad (ideal para grandes volúmenes).
4. Ajustar los radios de similitud (Euclidiana y Coseno) según la tolerancia deseada.
5. Hacer clic en "Buscar imágenes similares".

### Visualización de Resultados

Cada consulta realizada genera un registro automático en la carpeta results/. El sistema crea una subcarpeta con el formato consulta_NOMBRE_FECHA que contiene:

* 📄 metadata_log.txt: detalles técnicos de la ejecución (tiempos, parámetros utilizados).

* 📊 resultados_unificados.csv: tabla con los vecinos más cercanos encontrados y sus distancias.

* 🖼️ Mosaicos .jpg: imagen consolidada que muestra tu consulta junto a los 10 resultados más similares visualmente.

### Ejecución por Lotes (Avanzado)

Si se desean procesar múltiples consultas automáticamente sin usar la interfaz web, se puede utilizar este script:
```
python run_consultas.py --radio_cos 0.45 --engine Faiss
```

Se procesarán todas las imágenes colocadas en data/consulta.

## Tecnologías Utilizadas

* TensorFlow/Keras: implementación de ResNet50 (pre-entrenada en ImageNet) para extracción de embeddings.
* FAISS (Facebook AI Similarity Search): indexación y búsqueda eficiente de vectores.
* Streamlit: frontend interactivo.
* Pandas/Numpy: manipulación de datos y operaciones matriciales
