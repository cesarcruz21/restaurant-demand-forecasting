"""
pipeline/validation.py
Validación de datos de entrada para el Credit Risk Scoring System.
Autor: Cesar Cruz
"""

import pandas as pd
import streamlit as st


def validar_dataset(df: pd.DataFrame, mapeo: dict) -> tuple:
    """
    Valida que el dataset cumpla todos los requisitos antes de modelar.

    Returns
    -------
    (es_valido, errores, advertencias, info)
    """
    errores      = []
    advertencias = []
    info         = []

    # ── 1. Columnas mapeadas existen ─────────────────────────────────────────
    for clave, col_usuario in mapeo.items():
        if col_usuario not in df.columns:
            errores.append(
                f"Columna '{col_usuario}' (mapeada como '{clave}') no encontrada en el dataset."
            )

    if errores:
        return False, errores, advertencias, info

    # ── 2. Conversión implícita texto → numérico ──────────────────────────────
    for clave, col in mapeo.items():
        if df[col].dtype == object:
            converted = pd.to_numeric(df[col], errors='coerce')
            pct_failed = converted.isna().mean() * 100
            if pct_failed < 50:
                advertencias.append(
                    f"'{col}' es texto — se convertirá a numérico "
                    f"({pct_failed:.1f}% de valores no convertibles → NaN)."
                )
            else:
                errores.append(
                    f"'{col}' es texto y {pct_failed:.0f}% de valores no son numéricos. "
                    f"Verifica la columna."
                )

    # ── 3. Tipos de datos ─────────────────────────────────────────────────────
    col_target  = mapeo['target']
    col_ingreso = mapeo['ingreso']
    col_edad    = mapeo['edad']

    if not pd.api.types.is_numeric_dtype(df[col_target]):
        errores.append(f"La columna target '{col_target}' debe ser numérica (0/1).")

    valores_target  = df[col_target].dropna().unique()
    valores_validos = set(valores_target).issubset({0, 1, 0.0, 1.0})
    if not valores_validos:
        errores.append(
            f"'{col_target}' solo debe contener 0 y 1. "
            f"Valores encontrados: {sorted(valores_target)[:5]}"
        )

    if not pd.api.types.is_numeric_dtype(df[col_ingreso]):
        errores.append(f"La columna ingreso '{col_ingreso}' debe ser numérica.")

    if not pd.api.types.is_numeric_dtype(df[col_edad]):
        errores.append(f"La columna edad '{col_edad}' debe ser numérica.")

    # ── 4. Tamaño mínimo ──────────────────────────────────────────────────────
    if len(df) < 500:
        errores.append(
            f"El dataset tiene solo {len(df):,} filas. Se necesitan al menos 500 para modelar."
        )
    elif len(df) < 2_000:
        advertencias.append(
            f"El dataset tiene {len(df):,} filas. Con más de 2,000 los modelos son más estables."
        )

    # ── 5. Desbalance del target ──────────────────────────────────────────────
    if col_target in df.columns and pd.api.types.is_numeric_dtype(df[col_target]):
        tasa_default = df[col_target].mean()
        if tasa_default < 0.01:
            advertencias.append(
                f"Tasa de default muy baja ({tasa_default*100:.2f}%). "
                f"El modelo puede tener dificultades para detectar defaults."
            )
        elif tasa_default > 0.5:
            advertencias.append(
                f"Tasa de default alta ({tasa_default*100:.1f}%). "
                f"Verifica que el target esté codificado correctamente (1=default)."
            )
        else:
            info.append(f"Tasa de default: {tasa_default*100:.2f}% — rango aceptable.")

    # ── 6. Nulos excesivos ────────────────────────────────────────────────────
    for clave, col_usuario in mapeo.items():
        if col_usuario not in df.columns:
            continue
        pct_nulos = df[col_usuario].isnull().mean() * 100
        if pct_nulos > 50:
            errores.append(
                f"'{col_usuario}' tiene {pct_nulos:.1f}% de valores nulos — demasiado para imputar."
            )
        elif pct_nulos > 20:
            advertencias.append(
                f"'{col_usuario}' tiene {pct_nulos:.1f}% de nulos. Se imputará con mediana."
            )
        elif pct_nulos > 0:
            info.append(f"'{col_usuario}': {pct_nulos:.1f}% de nulos — se imputarán con mediana.")

    # ── 7. Rangos razonables ──────────────────────────────────────────────────
    if col_edad in df.columns:
        edad_min = df[col_edad].min()
        edad_max = df[col_edad].max()
        if edad_min < 0 or edad_max > 120:
            advertencias.append(
                f"Rango de edad fuera de lo esperado: [{edad_min}, {edad_max}]. "
                f"Se corregirán valores imposibles."
            )

    if col_ingreso in df.columns:
        ingresos_negativos = (df[col_ingreso] < 0).sum()
        if ingresos_negativos > 0:
            advertencias.append(
                f"{ingresos_negativos:,} registros con ingreso negativo — "
                f"podrían ser errores de captura."
            )

    es_valido = len(errores) == 0
    return es_valido, errores, advertencias, info


def mostrar_validacion(es_valido: bool, errores: list,
                       advertencias: list, info: list) -> bool:
    """Renderiza los resultados de validación en la UI de Streamlit."""
    st.markdown('<div class="section-header">Validación de datos</div>',
                unsafe_allow_html=True)

    for msg in errores:
        st.markdown(f'<div class="validation-error">✗ {msg}</div>',
                    unsafe_allow_html=True)
    for msg in advertencias:
        st.markdown(f'<div class="validation-warning">⚠ {msg}</div>',
                    unsafe_allow_html=True)
    for msg in info:
        st.markdown(f'<div class="validation-ok">✓ {msg}</div>',
                    unsafe_allow_html=True)

    if es_valido:
        st.success("Dataset validado — listo para modelar.")
    else:
        st.error("Corrige los errores antes de continuar.")

    return es_valido
