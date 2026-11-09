# EA2 · Modelos de predicción y clasificación

Un notebook ejecutable que descarga un dataset de Hugging Face, lo prepara, entrena una **regresión lineal** y un **clasificador**, y deja un resumen `.txt` en `reportes/`. GitHub Actions lo ejecuta automáticamente con cada `push` a `main`.

## Integrante

| Nombre | Rol |
|---|---|
| _[Escribe aquí tu nombre completo]_ | Desarrollo completo (grupo de una persona) |

## Dataset

| | |
|---|---|
| **Identificador** | `jdavit/colombian-conflict-SQA` |
| **Enlace** | https://huggingface.co/datasets/jdavit/colombian-conflict-SQA |
| **Configuración** | `default` |
| **Splits** | `train` (~2.895 filas) y `test` (30 filas). Se cargan los dos y se unen; luego se hace una división propia 80/20, porque 30 filas no alcanzan para evaluar. |
| **Licencia** | CC-BY-4.0 |

**Contenido:** unas 2.925 preguntas y respuestas en español sobre informes del conflicto armado colombiano (masacres, desplazamiento, desaparición forzada, memoria histórica…). Las respuestas fueron escritas por modelos de inteligencia artificial.

| Columna | Qué tiene |
|---|---|
| `question` | Pregunta, por ejemplo "¿Cuáles son los hallazgos principales del informe 058-CI-00233…?" |
| `context` | Fragmento del informe que recibió la IA como apoyo (a veces vacío) |
| `answer` | Respuesta escrita por la IA |
| `generated_by` | Qué IA escribió la respuesta (casi siempre `gpt-4o`) |

## Estructura

```
src/clases_po/                 Clases reutilizables (basadas en las del repositorio del curso)
  hf_loader.py                 HFDatasetLoader: conexión automática a Hugging Face
  preparador_conflicto.py      PreparadorConflicto: validar, limpiar y convertir textos en números
  regresion_lineal.py          RegresionLineal: preparar, dividir, entrenar (.pkl), evaluar
  clasificador_text.py         ClasificadorTexto: TF-IDF + clasificador, métricas, palabras clave
  reportes.py                  Reportes: genera el resumen .txt de los dos modelos
src/actividades/
  EA2_Modelos.ipynb            Notebook que orquesta todo el flujo (con resultados visibles)
  reportes/EA2_resumen_modelos.txt   Resumen generado automáticamente
  figuras/                     Gráficas generadas (balance, regresión, matriz de confusión)
  datos/, entrenamiento/       CSV del dataset y modelos .pkl (se regeneran; van en el artefacto)
.github/workflows/ejecutar-ea2.yml  Ejecución automática en GitHub Actions
requirements.txt
```

## Flujo

```
Hugging Face → DataFrame → preprocesamiento
                         → regresión lineal + evaluación
                         → clasificación + evaluación
                         → resumen automático .txt
GitHub Actions ejecuta el flujo completo.
```

## Uso

```bash
cd EA2_modelos
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install pyarrow openpyxl  # opcional: parquet y xlsx
jupyter lab                   # abrir src/actividades/EA2_Modelos.ipynb y ejecutar todo
```

O sin abrir Jupyter, igual que lo hace GitHub Actions:

```bash
cd src/actividades
jupyter nbconvert --to notebook --execute --inplace EA2_Modelos.ipynb
```

## Ejecución automática (GitHub Actions)

El workflow `.github/workflows/ejecutar-ea2.yml` se activa **automáticamente con cada `push` a `main`**. Hace lo siguiente:

1. Crea un `.venv` con Python 3.12 e instala las dependencias.
2. Limpia los resultados anteriores.
3. Ejecuta `EA2_Modelos.ipynb` con `nbconvert`: obtiene el dataset, construye el DataFrame, preprocesa, entrena y evalúa los dos modelos y genera el `.txt`.
4. Publica el artefacto descargable **`resumen-ea2`**, con el `.txt`, las figuras, los modelos `.pkl` y el notebook ejecutado.
5. Hace commit del notebook con resultados, del `.txt` y de las figuras con el mensaje `[skip ci]`, así el repositorio siempre tiene una copia del resumen generado.

**Ejecución exitosa:** _[pega aquí el enlace a la ejecución de Actions]_

## Dónde están las evidencias

| Evidencia | Ubicación |
|---|---|
| Carga y estructura del DataFrame | Notebook, Etapa 1 (forma, `info()`, muestra, diccionario de columnas) |
| Preprocesamiento antes y después | Notebook, Etapa 2 (tabla de calidad, registro de filas, tabla antes/después) y sección 2 del `.txt` |
| Regresión lineal | Notebook, Etapa 3; sección 3 del `.txt`; `figuras/regresion_real_vs_predicho.png` |
| Clasificación | Notebook, Etapa 4; sección 4 del `.txt`; `figuras/matriz_confusion.png` |
| Resumen automático | `src/actividades/reportes/EA2_resumen_modelos.txt` y artefacto `resumen-ea2` |
| Control de versiones | Historial de commits del repositorio |

---

## Explicación en palabras sencillas

### Preprocesamiento

El dataset solo trae **texto**, y un modelo solo entiende **números**. Por eso, además de limpiar los datos, se "miden" los textos.

| Paso | Qué problema resuelve |
|---|---|
| Validar (calidad: métrica + decisión) | Saber cuántos datos vacíos, duplicados o mal formateados hay antes de confiar en ellos. |
| Normalizar espacios | Los espacios dobles cambian el conteo de palabras y esconden duplicados. |
| Quitar filas sin pregunta o respuesta | Sin ellas no hay nada que aprender. |
| Quitar duplicados | Una fila repetida que cae en entrenamiento y en prueba es como ver el examen antes de presentarlo. |
| Contexto vacío → `tiene_contexto = 0` | No se borran esas filas; se marcan. |
| Crear variables numéricas | Largo de la pregunta, del contexto y de la respuesta, y columnas 0/1 (*one-hot*) para las categorías. |
| Crear la etiqueta `tipo_pregunta` | Regla fija sobre la pregunta (ver Clasificación). |

**Fuga de información:** todos estos pasos son reglas fijas que no aprenden de los datos. Lo único que aprende de los datos es **TF-IDF**, que construye un vocabulario. Está dentro del `Pipeline` del clasificador, así que se ajusta **solo con entrenamiento**. La regresión no usa escalado, por lo que sus coeficientes se leen directamente en caracteres.

### Regresión lineal

- **Pregunta:** ¿cuántos caracteres tendrá la respuesta de la IA? Sirve para estimar costos y tiempos de un sistema de preguntas y respuestas.
- **y:** `largo_respuesta_car`.
- **X:**
  - `largo_pregunta_palabras` y `largo_contexto_palabras`
  - `tiene_contexto`
  - `tipo_hallazgos` y `tipo_hechos_especificos`; la referencia es `causas_y_recomendaciones`
  - `generado_gpt4o`
- **Descartada:** `menciona_codigo_informe`. Casi siempre coincide con el tipo de pregunta, y dos variables que dicen lo mismo (colinealidad) vuelven inestables los coeficientes.
- **Modelo:** `LinearRegression()` con valores predeterminados, `test_size=0.2` y `random_state=42`.
- **Cómo leerla:** `y = w₁·x₁ + … + b`. Cada **w** son caracteres que suma o resta su variable, y **b** es el punto de partida.
- **Métricas:**
  - **MAE:** error promedio en caracteres.
  - **RMSE:** como el MAE, pero castiga más los errores grandes.
  - **R²:** qué parte de las diferencias de largo se explica.
  - Todo se compara con **Pere**, el modelo perezoso que siempre predice el promedio.

### Clasificación

- **Pregunta:** leyendo solo la respuesta, ¿qué tipo de pregunta se hizo?
- **Clases:**
  - `causas_y_recomendaciones`: la pregunta menciona "causas estructurales" o "no repetición".
  - `hallazgos`: menciona "hallazgos".
  - `hechos_especificos`: cualquier otra pregunta (eventos, actores, lugares).
- **X:** el texto de `answer` convertido con TF-IDF. **La pregunta no se usa**, porque de ella sale la etiqueta.
- **Algoritmo:** regresión logística. Funciona bien con texto, es rápida y deja ver qué palabras usa para decidir.
  - TF-IDF: `ngram_range=(1,2)`, `max_features=3000`, `min_df=2` y palabras vacías en español.
  - Modelo: `class_weight="balanced"`, `max_iter=2000` y `C=1.0` (predeterminado).
- **Métricas:**
  - **Exactitud.**
  - **Precisión:** de lo que dijo que era de una clase, cuánto lo era.
  - **Recall:** de lo que era de una clase, cuánto encontró.
  - **F1 macro:** cada clase pesa igual, así que es justo aunque las clases tengan tamaños distintos.
  - **Matriz de confusión.**
  - Comparación con Pere, que siempre dice la clase más común.

Los valores obtenidos, su interpretación y las limitaciones de cada modelo están en el `.txt` y en el notebook.
