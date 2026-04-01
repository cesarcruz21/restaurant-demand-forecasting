"""
pipeline/preprocessing.py
Limpieza de datos y feature engineering para el Credit Risk Scoring System.
Autor: Cesar Cruz
"""

import numpy as np
import pandas as pd


def limpiar_datos(df: pd.DataFrame, mapeo: dict) -> pd.DataFrame:
    """
    Renombra columnas, imputa nulos y corrige valores fuera de rango.

    Parameters
    ----------
    df     : DataFrame original con los nombres de columnas del usuario.
    mapeo  : Diccionario {nombre_interno: nombre_usuario}.

    Returns
    -------
    DataFrame limpio con nombres internos estandarizados.
    """
    # Renombrar a nombres internos
    df_clean = df.rename(
        columns={v: k for k, v in mapeo.items() if v in df.columns}
    ).copy()

    # Convertir columnas texto a numérico si es necesario
    for col in df_clean.select_dtypes(include='object').columns:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

    # Imputar mediana en ingreso y dependientes
    for col in ['ingreso', 'dependientes']:
        if col in df_clean.columns and df_clean[col].isnull().any():
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())

    # Corregir edad cero (imposible)
    if 'edad' in df_clean.columns:
        df_clean.loc[df_clean['edad'] == 0, 'edad'] = df_clean['edad'].median()

    # Clipear utilización a [0, 1]
    if 'utilizacion' in df_clean.columns:
        df_clean['utilizacion'] = df_clean['utilizacion'].clip(lower=0.0, upper=1.0)

    # Clipear deuda_ratio en percentil 99 (winsorización)
    if 'deuda_ratio' in df_clean.columns:
        p99 = df_clean['deuda_ratio'].quantile(0.99)
        df_clean['deuda_ratio'] = df_clean['deuda_ratio'].clip(upper=p99)

    # Imputar cualquier nulo numérico restante con mediana
    for col in df_clean.select_dtypes(include='number').columns:
        if df_clean[col].isnull().any():
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())

    return df_clean


def crear_features(df_clean: pd.DataFrame) -> pd.DataFrame:
    """
    Feature engineering sobre el dataset limpio.

    Features creadas
    ----------------
    - total_past_due      : suma de atrasos en todos los buckets
    - clean_history       : 1 si nunca tuvo atraso
    - income_per_dependent: ingreso por dependiente (capacidad real de pago)
    - high_utilization    : flag si utilización > 80%
    - debt_to_income      : ratio deuda/ingreso normalizado
    - age_segment         : segmento etario (0=joven … 3=senior)
    """
    df_feat = df_clean.copy()

    # Eliminar columna de binning preexistente si existe
    if 'age_bin' in df_feat.columns:
        df_feat = df_feat.drop(columns=['age_bin'])

    # Atrasos acumulados
    cols_atraso = [c for c in ['atraso_30', 'atraso_60', 'atraso_90'] if c in df_feat.columns]
    if cols_atraso:
        df_feat['total_past_due'] = df_feat[cols_atraso].sum(axis=1)
        df_feat['clean_history']  = (df_feat['total_past_due'] == 0).astype(int)
    else:
        df_feat['total_past_due'] = 0
        df_feat['clean_history']  = 1

    # Ingreso por dependiente (evita división por cero sumando 1)
    if 'ingreso' in df_feat.columns and 'dependientes' in df_feat.columns:
        df_feat['income_per_dependent'] = df_feat['ingreso'] / (df_feat['dependientes'] + 1)

    # Flag de alta utilización
    if 'utilizacion' in df_feat.columns:
        df_feat['high_utilization'] = (df_feat['utilizacion'] > 0.8).astype(int)

    # Ratio deuda / ingreso normalizado
    if 'deuda_ratio' in df_feat.columns and 'ingreso' in df_feat.columns:
        df_feat['debt_to_income'] = df_feat['deuda_ratio'] / (df_feat['ingreso'] / 1_000 + 1)

    # Segmento etario
    if 'edad' in df_feat.columns:
        df_feat['age_segment'] = np.where(
            df_feat['edad'] <= 30, 0,
            np.where(df_feat['edad'] <= 45, 1,
            np.where(df_feat['edad'] <= 60, 2, 3))
        )

    return df_feat
