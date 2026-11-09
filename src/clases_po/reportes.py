"""Generación de reportes de DataFrames."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


class Reportes:
    """Imprime y guarda un resumen legible de un DataFrame."""

    def generar_reporte(self, df: pd.DataFrame) -> str:
        """Genera el contenido textual del reporte."""
        self._validar_dataframe(df)
        numericas = df.select_dtypes(include="number")
        lineas = [
            "REPORTE DEL DATAFRAME",
            "=" * 60,
            f"Filas: {len(df)}",
            f"Columnas: {len(df.columns)}",
            f"Duplicados: {int(df.duplicated().sum())}",
            "",
            "Columnas y tipos:",
            *[f"- {columna}: {tipo}" for columna, tipo in df.dtypes.items()],
            "",
            "Valores nulos por columna:",
            *[f"- {columna}: {cantidad}" for columna, cantidad in df.isna().sum().items()],
        ]

        if not numericas.empty:
            lineas += [
                "",
                "Estadísticas numéricas:",
                numericas.describe().round(3).to_string(),
            ]

        return "\n".join(lineas) + "\n"

    def imprimir_reporte(self, df: pd.DataFrame) -> None:
        """Imprime en la consola el reporte del DataFrame."""
        print(self.generar_reporte(df), end="")

    def guardar_reporte(self, df: pd.DataFrame, ruta: str | Path) -> Path:
        """Guarda el reporte en un archivo de texto y devuelve su ruta."""
        path = Path(ruta)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.generar_reporte(df), encoding="utf-8")
        return path
        
    def reporte_regresion_lineal(
        self,
        df: pd.DataFrame,
        regresion: object,
        ruta: str | Path = "entrenamiento/reporte_regresion_lineal.txt",
    ) -> Path:
        """Genera un TXT con el DataFrame y los resultados de una regresión.

        La regresión debe haber sido ejecutada previamente con
        ``RegresionLineal.ejecutar(df)``. El reporte no vuelve a entrenar,
        predecir ni evaluar el modelo: utiliza los resultados que la instancia
        conserva en ``modelo``, ``parametros``, ``metricas`` y
        ``ruta_modelo``.

        El archivo incluye la configuración de columnas, las dimensiones de
        los conjuntos de datos, los parámetros ``w`` y ``b``, las métricas
        MAE/RMSE/R², la ruta del modelo persistido, el resumen del DataFrame
        y sus primeras diez filas.

        Args:
            df: DataFrame utilizado para preparar la regresión.
            regresion: Instancia ejecutada de ``RegresionLineal``.
            ruta: Ruta del archivo TXT que se creará.

        Returns:
            Ruta del archivo TXT generado.

        Raises:
            TypeError: Si ``df`` no es un DataFrame, ``regresion`` no es una
                instancia de ``RegresionLineal`` o ``ruta`` no es una ruta.
            ValueError: Si la instancia todavía no tiene resultados de una
                ejecución completa.
            OSError: Si no se puede crear o escribir el archivo.
        """
        self._validar_dataframe(df)
        if not isinstance(ruta, (str, Path)):
            raise TypeError("ruta debe ser una cadena o pathlib.Path.")

        from .regresion_lineal import RegresionLineal

        if not isinstance(regresion, RegresionLineal):
            raise TypeError("regresion debe ser una instancia de RegresionLineal.")
        resultados_requeridos = {
            "modelo": regresion.modelo,
            "parametros": regresion.parametros,
            "metricas": regresion.metricas,
            "ruta_modelo": regresion.ruta_modelo,
        }
        faltantes = [
            nombre for nombre, valor in resultados_requeridos.items()
            if valor is None
        ]
        if faltantes:
            raise ValueError(
                "La regresión debe ejecutarse antes de generar el reporte. "
                f"Faltan resultados: {', '.join(faltantes)}."
            )

        lineas = [
            "REPORTE DE REGRESIÓN LINEAL",
            "=" * 70,
            f"Columnas predictoras (X): {regresion.feature_column}",
            f"Columna objetivo (y): {regresion.target_column}",
            f"Test size: {regresion.test_size}",
            f"Random state: {regresion.random_state}",
            "",
            "MODELO",
            "-" * 70,
            f"Ruta del modelo: {regresion.ruta_modelo}",
            f"Parámetros: {regresion.parametros}",
            "",
            "MÉTRICAS DE EVALUACIÓN",
            "-" * 70,
            *[
                f"{nombre.upper()}: {valor:.6f}"
                for nombre, valor in regresion.metricas.items()
            ],
            "",
            "DIMENSIONES DE LOS DATOS",
            "-" * 70,
            f"DataFrame original: {df.shape}",
            f"X: {regresion.X.shape if regresion.X is not None else 'no disponible'}",
            f"y: {regresion.y.shape if regresion.y is not None else 'no disponible'}",
            (
                "X_train: "
                f"{regresion.X_train.shape if regresion.X_train is not None else 'no disponible'}"
            ),
            (
                "X_test: "
                f"{regresion.X_test.shape if regresion.X_test is not None else 'no disponible'}"
            ),
            (
                "y_train: "
                f"{regresion.y_train.shape if regresion.y_train is not None else 'no disponible'}"
            ),
            (
                "y_test: "
                f"{regresion.y_test.shape if regresion.y_test is not None else 'no disponible'}"
            ),
            "",
            "REPORTE DEL DATAFRAME",
            "-" * 70,
            self.generar_reporte(df).rstrip(),
            "",
            "PRIMERAS 10 FILAS",
            "-" * 70,
            df.head(10).to_string(index=False),
            "",
        ]
        path = Path(ruta)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lineas), encoding="utf-8")
        print(f"[reporte_regresion_lineal] reporte guardado en: {path}")
        return path


    @staticmethod
    def _validar_dataframe(df: pd.DataFrame) -> None:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df debe ser un pandas.DataFrame.")

    # ================================================================== #
    # EA2: resumen automático de los dos modelos
    # ================================================================== #
    LECTURA_VARIABLES = {
        "largo_pregunta_palabras": "por cada palabra más en la pregunta",
        "largo_contexto_palabras": "por cada palabra más de contexto",
        "tiene_contexto": "si la pregunta trae contexto (1) frente a no traerlo (0)",
        "tipo_hallazgos": "si la pregunta pide hallazgos (frente a causas y recomendaciones)",
        "tipo_hechos_especificos": "si la pregunta es de hechos específicos (frente a causas y recomendaciones)",
        "generado_gpt4o": "si la respuesta la escribió gpt-4o (frente a otra IA)",
    }

    @classmethod
    def coeficientes_regresion(cls, regresion: object) -> pd.DataFrame:
        """Tabla w por variable con su lectura en palabras."""
        w = regresion.parametros["w"]
        w = [w] if not isinstance(w, list) else w
        filas = []
        for variable, peso in zip(regresion.feature_column, w):
            contexto = cls.LECTURA_VARIABLES.get(variable, f"si {variable} sube 1 unidad")
            filas.append({
                "variable": variable,
                "w": round(float(peso), 2),
                "lectura": f"{contexto}, la respuesta cambia {peso:+.1f} caracteres",
            })
        tabla = pd.DataFrame(filas)
        return tabla.reindex(tabla["w"].abs().sort_values(ascending=False).index).reset_index(drop=True)

    @classmethod
    def interpretar_regresion(cls, regresion: object, mae_perezoso: float) -> list[str]:
        m = regresion.metricas
        media = float(regresion.y_test.mean())
        r2_train = float(regresion.modelo.score(regresion.X_train, regresion.y_train))
        mejora = 100 * (1 - m["mae"] / mae_perezoso) if mae_perezoso else 0.0
        principal = cls.coeficientes_regresion(regresion).iloc[0]
        return [
            f"MAE = {m['mae']:.0f}: en promedio el modelo se equivoca en {m['mae']:.0f} caracteres, "
            f"frente a respuestas que miden en promedio {media:.0f} (error típico del {100 * m['mae'] / media:.0f} %).",
            f"RMSE = {m['rmse']:.0f}: es mayor que el MAE porque castiga más los errores grandes; "
            "algunas respuestas se salen mucho de lo esperado.",
            f"R² = {m['r2']:.2f}: el modelo explica cerca del {100 * max(m['r2'], 0):.0f} % de las diferencias "
            "de largo entre respuestas; el resto depende del contenido del informe, que el modelo no lee.",
            f"Pere (predecir siempre el promedio) tiene MAE = {mae_perezoso:.0f}; el modelo reduce el error en {mejora:.0f} %.",
            f"R² entrenamiento = {r2_train:.2f} vs prueba = {m['r2']:.2f}: "
            + ("parecidos, no hay señales de sobreajuste (el modelo no memorizó)."
               if abs(r2_train - m["r2"]) < 0.1 else "hay diferencia: rinde peor con datos nuevos."),
            f"Variable con más peso: {principal['variable']} ({principal['lectura']}).",
            f"b = {regresion.parametros['b']:.0f}: largo esperado cuando todas las variables valen 0 "
            "(pregunta de causas y recomendaciones, sin contexto"
            + (", escrita por otra IA" if "generado_gpt4o" in regresion.feature_column else "")
            + "). Es el punto de partida de la recta; una pregunta de 0 palabras no existe en la realidad.",
        ]

    @staticmethod
    def interpretar_clasificacion(clasificador: object, exactitud_perezoso: float, clase_perezoso: str) -> list[str]:
        m = clasificador.metricas
        pc = m["por_clase"]
        mejor, peor = pc["f1"].idxmax(), pc["f1"].idxmin()
        matriz = m["matriz"].to_numpy().copy()
        np.fill_diagonal(matriz, 0)
        lineas = [
            f"Exactitud = {m['exactitud']:.1%}: de cada 100 respuestas de prueba acierta el tipo de pregunta en unas "
            f"{100 * m['exactitud']:.0f}.",
            f"Pere (decir siempre '{clase_perezoso}') acierta {exactitud_perezoso:.1%}; el modelo lo supera por "
            f"{100 * (m['exactitud'] - exactitud_perezoso):.1f} puntos.",
            f"Las clases no tienen el mismo tamaño, así que también se usa F1 macro = {m['f1_macro']:.2f} "
            "(promedio de F1 donde cada clase pesa igual) para que una clase pequeña no quede escondida.",
            (f"Clase que mejor reconoce: {mejor} (F1 = {pc.loc[mejor, 'f1']:.2f}); "
             f"la más difícil: {peor} (F1 = {pc.loc[peor, 'f1']:.2f}).")
            if pc["f1"].max() - pc["f1"].min() > 0.005
            else f"Las tres clases se reconocen igual de bien (F1 = {pc['f1'].mean():.2f} en cada una).",
        ]
        if m["exactitud"] >= 0.97:
            lineas.append(
                "¿Por qué tan alto? La IA repite en la respuesta palabras de la pregunta (por ejemplo "
                "'causas estructurales' o 'hallazgos principales'). No es fuga de información, porque el modelo "
                "nunca ve la pregunta, pero muestra que con estas plantillas la tarea es fácil."
            )
        if matriz.sum():
            i, j = np.unravel_index(matriz.argmax(), matriz.shape)
            clases = list(pc.index)
            lineas.append(
                f"Error más frecuente: {matriz[i, j]} respuestas de {clases[i]} se clasificaron como {clases[j]}; "
                "ocurre cuando una respuesta mezcla temas (hallazgos que también narran hechos concretos)."
            )
        else:
            lineas.append("No hubo errores en el conjunto de prueba.")
        return lineas

    LIMITACIONES_REGRESION = [
        "Supone relaciones en línea recta; el efecto real del contexto puede no serlo.",
        "Mide los textos pero no entiende su significado.",
        "Las respuestas las escribió una IA: el modelo describe el estilo de esa IA.",
        "Varias preguntas tratan del mismo informe; preguntas parecidas pueden quedar en entrenamiento y prueba.",
    ]
    LIMITACIONES_CLASIFICACION = [
        "La etiqueta sale de una regla; si la regla se equivoca, el modelo aprende ese error.",
        "Las respuestas siguen plantillas, lo que facilita la tarea; con textos libres el resultado bajaría.",
        "TF-IDF no entiende el orden ni el significado de las palabras, solo cuáles aparecen.",
    ]

    def reporte_modelos(
        self,
        info_dataset: dict,
        df_crudo: pd.DataFrame,
        df_preparado: pd.DataFrame,
        calidad: pd.DataFrame,
        registro: pd.DataFrame,
        regresion: object,
        mae_perezoso: float,
        variable_descartada: tuple[str, float],
        clasificador: object,
        exactitud_perezoso: float,
        clase_perezoso: str,
        ruta: str | Path = "reportes/EA2_resumen_modelos.txt",
    ) -> Path:
        """Escribe un TXT con el dataset, el preprocesamiento y los dos modelos.

        Todo sale de objetos ya ejecutados (``RegresionLineal.ejecutar`` y
        ``ClasificadorTexto.ejecutar``): el reporte no vuelve a entrenar.
        """
        from datetime import datetime
        from zoneinfo import ZoneInfo

        import sklearn

        self._validar_dataframe(df_crudo)
        self._validar_dataframe(df_preparado)
        if regresion.metricas is None or clasificador.metricas is None:
            raise ValueError("Ejecuta los dos modelos antes de generar el reporte.")

        L, S = "=" * 78, "-" * 78
        vinetas = lambda items: [f"  - {x}" for x in items]
        m_r, m_c = regresion.metricas, clasificador.metricas
        pc = m_c["por_clase"].copy()
        clf_modelo = clasificador.pipeline.named_steps["modelo"]
        tfidf = clasificador.pipeline.named_steps["tfidf"]
        ahora = datetime.now(ZoneInfo("America/Bogota"))

        lineas = [
            L,
            "RESUMEN AUTOMÁTICO · EA2 Modelos de predicción y clasificación",
            f"Fecha de ejecución: {ahora:%Y-%m-%d %H:%M:%S} (hora de Colombia) · scikit-learn {sklearn.__version__}",
            L,
            "",
            "1. DATASET",
            S,
            f"Identificador: {info_dataset['dataset']}",
            f"Enlace: https://huggingface.co/datasets/{info_dataset['dataset']}",
            f"Configuración: {info_dataset.get('config') or 'default'} · splits: {info_dataset['splits']}",
            "Contenido: preguntas y respuestas en español sobre informes del conflicto armado colombiano.",
            f"DataFrame crudo: {df_crudo.shape[0]} filas x {df_crudo.shape[1]} columnas "
            f"({', '.join(map(str, df_crudo.columns))})",
            "",
            "2. PREPROCESAMIENTO",
            S,
            "Validación (métrica y decisión):",
            calidad.to_string(index=False),
            "",
            "Registro de limpieza (filas por paso):",
            registro.to_string(index=False),
            "",
            f"Enriquecimiento: de {df_crudo.shape[1]} a {df_preparado.shape[1]} columnas. Se midieron los textos "
            "(palabras y caracteres), se marcó si hay contexto, se creó la etiqueta tipo_pregunta y "
            "columnas 0/1 (one-hot) para tipo_pregunta y generated_by.",
            "Fuga de información: las reglas anteriores no aprenden de los datos. TF-IDF, que sí aprende un "
            "vocabulario, se ajusta solo con entrenamiento dentro del Pipeline del clasificador.",
            "",
            "3. MODELO DE REGRESIÓN LINEAL",
            S,
            "Objetivo: predecir cuántos caracteres tendrá la respuesta escrita por la IA.",
            f"Variable objetivo (y): {regresion.target_column} (numérica continua)",
            f"Variables predictoras (X): {regresion.feature_column}",
            f"Variable descartada: {variable_descartada[0]} (correlación {variable_descartada[1]:.2f} con "
            "tipo_hechos_especificos: dice casi lo mismo y volvería inestables los coeficientes).",
            "Algoritmo: LinearRegression (mínimos cuadrados) con parámetros predeterminados.",
            f"División: test_size={regresion.test_size}, random_state={regresion.random_state} · "
            f"entrenamiento={len(regresion.X_train)} filas, prueba={len(regresion.X_test)} filas.",
            f"Modelo guardado: {regresion.ruta_modelo}",
            "",
            f"Métricas en prueba: MAE={m_r['mae']:.2f} · RMSE={m_r['rmse']:.2f} · R²={m_r['r2']:.4f} · "
            f"MAE de Pere={mae_perezoso:.2f}",
            f"Intercepto b = {regresion.parametros['b']:.2f}",
            "Coeficientes w:",
            self.coeficientes_regresion(regresion).to_string(index=False),
            "",
            "Interpretación:",
            *vinetas(self.interpretar_regresion(regresion, mae_perezoso)),
            "Limitaciones:",
            *vinetas(self.LIMITACIONES_REGRESION),
            "",
            "4. MODELO DE CLASIFICACIÓN",
            S,
            "Objetivo: a partir de la respuesta, identificar qué tipo de pregunta se hizo.",
            f"Variable objetivo (y): {clasificador.columna_objetivo} · clases: {list(pc.index)}",
            "Regla de la etiqueta: 'causas estructurales' o 'no repetición' -> causas_y_recomendaciones; "
            "'hallazgos' -> hallazgos; cualquier otra -> hechos_especificos.",
            f"Variable predictora (X): {clasificador.columna_texto} convertida en números con TF-IDF. "
            "No se usa la pregunta: de ella sale la etiqueta (sería darle la respuesta del examen).",
            f"Algoritmo: TfidfVectorizer(ngram_range={tfidf.ngram_range}, max_features={tfidf.max_features}, "
            f"min_df={tfidf.min_df}, stop_words=español) -> {type(clf_modelo).__name__}"
            f"(class_weight={getattr(clf_modelo, 'class_weight', None)}, max_iter={getattr(clf_modelo, 'max_iter', None)}, "
            f"C={getattr(clf_modelo, 'C', None)})",
            "Por qué: la regresión logística funciona bien con texto convertido en números, es rápida y deja ver "
            "qué palabras usa para decidir. class_weight='balanced' compensa las clases con menos ejemplos.",
            f"División estratificada: test_size={clasificador.test_size}, random_state={clasificador.random_state} · "
            f"entrenamiento={len(clasificador.X_train)}, prueba={len(clasificador.X_test)}.",
            f"Distribución de clases en prueba: {clasificador.y_test.value_counts().to_dict()}",
            f"Modelo guardado: {clasificador.ruta_modelo}",
            "",
            f"Métricas en prueba: exactitud={m_c['exactitud']:.4f} · F1 macro={m_c['f1_macro']:.4f} · "
            f"F1 ponderado={m_c['f1_ponderado']:.4f} · exactitud de Pere={exactitud_perezoso:.4f}",
            pc.round(4).to_string(),
            "",
            "Matriz de confusión (filas = real, columnas = predicho):",
            m_c["matriz"].to_string(),
            "",
            "Palabras que más empujan hacia cada clase:",
            clasificador.palabras_clave(8).to_string(index=False),
            "",
            "Interpretación:",
            *vinetas(self.interpretar_clasificacion(clasificador, exactitud_perezoso, clase_perezoso)),
            "Limitaciones:",
            *vinetas(self.LIMITACIONES_CLASIFICACION),
            "",
            "5. CONCLUSIONES",
            S,
            f"  - La regresión explica {100 * max(m_r['r2'], 0):.0f} % de la variación del largo de las respuestas, "
            f"con un error típico de {m_r['mae']:.0f} caracteres: sirve como estimación aproximada.",
            f"  - El clasificador reconoce el tipo de pregunta con {m_c['exactitud']:.0%} de exactitud y F1 macro "
            f"{m_c['f1_macro']:.2f} usando solo la respuesta.",
            "  - Ambos modelos funcionan porque la IA responde con patrones; describen su estilo más que el conflicto.",
            "  - Mejora posible: dividir por informe para que un mismo informe no quede en entrenamiento y prueba.",
            "",
        ]
        path = Path(ruta)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lineas), encoding="utf-8")
        print(f"[reporte_modelos] reporte guardado en: {path}")
        return path
