"""Preparación del dataset jdavit/colombian-conflict-SQA para los dos modelos.

Sigue el orden de las clases sincrónicas:

    validar   -> tabla de calidad: cada validación tiene una métrica y una decisión
    limpiar   -> reglas repetibles; el registro guarda cuántas filas entran y salen
    enriquecer-> convierte los textos en columnas numéricas y crea la etiqueta

Ninguno de estos pasos "aprende" de los datos (no calcula promedios ni
vocabularios): son reglas fijas. Por eso pueden aplicarse antes de dividir en
entrenamiento y prueba sin causar fuga de información. Lo que sí aprende
(TF-IDF) queda dentro del Pipeline del clasificador y se ajusta solo con
entrenamiento.
"""
from __future__ import annotations

import re
import unicodedata

import pandas as pd

COLUMNAS_TEXTO = ["question", "context", "answer"]
PATRON_CODIGO_INFORME = re.compile(r"\b\d{3}-[A-Z]{2}-\d{5}\b")
CATEGORIA_REFERENCIA = "causas_y_recomendaciones"

# Palabras muy comunes en español que no ayudan a distinguir clases
STOPWORDS_ES = """a al algo algunas algunos ante antes como con contra cual cuales
cuando de del desde donde durante e el ella ellas ellos en entre era es esa esas
ese eso esos esta estaba estas este esto estos fue fueron ha han hasta hay la las
le les lo los mas me mi muy no nos o otra otras otro otros para pero por porque
que quien se segun ser si sin sobre su sus tambien tanto te tiene tienen todo
todos tras un una uno unos y ya informe""".split()


def normalizar(texto: str) -> str:
    """Minúsculas y sin tildes, para comparar textos sin importar cómo se escribieron."""
    texto = unicodedata.normalize("NFKD", str(texto).lower())
    return "".join(c for c in texto if not unicodedata.combining(c))


def tipo_de_pregunta(pregunta: str) -> str:
    """Regla fija que asigna la etiqueta de clasificación a partir de la pregunta."""
    p = normalizar(pregunta)
    if "causas estructurales" in p or "no repeticion" in p:
        return "causas_y_recomendaciones"
    if "hallazgos" in p:
        return "hallazgos"
    return "hechos_especificos"


class PreparadorConflicto:
    """Valida, limpia y enriquece el DataFrame crudo del dataset."""

    def __init__(self) -> None:
        self.calidad: pd.DataFrame | None = None
        self.registro: list[tuple[str, int, str]] = []

    # ------------------------------------------------------------------ #
    # 1. validar antes de confiar
    # ------------------------------------------------------------------ #
    def validar(self, df: pd.DataFrame) -> pd.DataFrame:
        """Mide la calidad del DataFrame crudo y propone una decisión por problema."""
        textos = df[COLUMNAS_TEXTO].astype("string")
        vacios = textos.fillna("").apply(lambda s: (s.str.strip() == "").sum())
        espacios = textos.fillna("").apply(
            lambda s: (s.str.contains(r"\s{2,}|^\s|\s$", regex=True)).sum()
        )
        filas = [
            ("Completitud · nulos (todas las columnas)", int(df.isna().sum().sum()), "convertir a vacío y revisar"),
            ("Completitud · question vacía", int(vacios["question"]), "eliminar fila"),
            ("Completitud · answer vacía", int(vacios["answer"]), "eliminar fila"),
            ("Completitud · context vacío", int(vacios["context"]), "conservar y marcar tiene_contexto = 0"),
            ("Unicidad · filas duplicadas (question+context+answer)",
             int(df.duplicated(subset=COLUMNAS_TEXTO).sum()), "eliminar"),
            ("Formato · textos con espacios dobles o en los bordes", int(espacios.sum()), "normalizar espacios"),
            ("Dominio · valores distintos de generated_by",
             int(df["generated_by"].nunique()), "convertir a categoría"),
        ]
        self.calidad = pd.DataFrame(filas, columns=["validación", "métrica", "decisión"])
        return self.calidad

    # ------------------------------------------------------------------ #
    # 2. limpiar con reglas que se puedan repetir
    # ------------------------------------------------------------------ #
    def limpiar(self, df: pd.DataFrame) -> pd.DataFrame:
        datos = df.copy()
        self.registro = [("datos crudos", len(datos), "train + test de Hugging Face")]

        for col in COLUMNAS_TEXTO:
            datos[col] = (datos[col].astype("string").fillna("")
                          .str.replace(r"\s+", " ", regex=True).str.strip())
        self.registro.append(("espacios normalizados", len(datos),
                              "los espacios extra cambian el conteo de palabras y esconden duplicados"))

        datos = datos[(datos["question"] != "") & (datos["answer"] != "")]
        self.registro.append(("con pregunta y respuesta", len(datos),
                              "sin pregunta o sin respuesta no hay nada que aprender"))

        datos = datos.drop_duplicates(subset=COLUMNAS_TEXTO).reset_index(drop=True)
        self.registro.append(("sin duplicados", len(datos),
                              "una fila repetida en entrenamiento y prueba sería ver el examen antes"))

        datos["generated_by"] = datos["generated_by"].astype("string").fillna("desconocido").astype("category")
        self.registro.append(("generated_by como categoría", len(datos),
                              "es una etiqueta (qué IA respondió), no texto libre"))
        return datos

    @property
    def tabla_registro(self) -> pd.DataFrame:
        tabla = pd.DataFrame(self.registro, columns=["paso", "filas", "por qué"])
        tabla.insert(2, "cambio", tabla["filas"].diff().fillna(0).astype(int))
        return tabla

    # ------------------------------------------------------------------ #
    # 3. enriquecer: de texto a números
    # ------------------------------------------------------------------ #
    @staticmethod
    def enriquecer(df: pd.DataFrame) -> pd.DataFrame:
        datos = df.copy()
        palabras = lambda s: s.str.split().str.len().fillna(0).astype(int)

        datos["largo_pregunta_palabras"] = palabras(datos["question"])
        datos["largo_contexto_palabras"] = palabras(datos["context"])
        datos["largo_respuesta_palabras"] = palabras(datos["answer"])
        datos["largo_respuesta_car"] = datos["answer"].str.len().astype(float)
        datos["tiene_contexto"] = (datos["context"] != "").astype(int)
        datos["menciona_codigo_informe"] = datos["question"].str.contains(PATRON_CODIGO_INFORME).astype(int)

        # Etiqueta de clasificación (regla fija sobre la pregunta)
        datos["tipo_pregunta"] = datos["question"].map(tipo_de_pregunta)

        # One-hot de tipo_pregunta para la regresión. Se deja fuera la
        # categoría de referencia para no repetir información.
        for tipo in ["hallazgos", "hechos_especificos"]:
            datos[f"tipo_{tipo}"] = (datos["tipo_pregunta"] == tipo).astype(int)

        # One-hot de generated_by: 1 si la respondió gpt-4o (la IA más común)
        datos["generado_gpt4o"] = (datos["generated_by"].astype(str) == "gpt-4o").astype(int)
        return datos

    def preparar(self, df: pd.DataFrame) -> pd.DataFrame:
        """Atajo: validar, limpiar y enriquecer en orden."""
        self.validar(df)
        return self.enriquecer(self.limpiar(df))
