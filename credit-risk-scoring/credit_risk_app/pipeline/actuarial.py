"""
pipeline/actuarial.py
Análisis actuarial PD/LGD/EAD/EL y simulación Monte Carlo.
Metodología alineada con Basilea II — cartera de consumo no garantizada.
Autor: Cesar Cruz
"""

import io

import numpy as np
import pandas as pd


# ── Tabla LGD por segmento de riesgo (Basilea II, consumo no garantizado) ────
# Valores de referencia regulatorios para portafolios retail sin garantía.
# Rango típico: 35%–85% según historial y tipo de producto.
LGD_TABLE = {
    'Bajo':     0.35,   # PD < 10%   → recuperación alta, cliente bajo riesgo
    'Moderado': 0.50,   # PD 10–30%  → referencia estándar Basilea
    'Alto':     0.65,   # PD 30–60%  → recuperación deteriorada
    'Muy Alto': 0.80,   # PD > 60%   → riesgo de pérdida casi total
}

N_SIMULACIONES = 5_000


def _asignar_segmento(pd_val: float) -> str:
    if pd_val < 0.10:
        return 'Bajo'
    elif pd_val < 0.30:
        return 'Moderado'
    elif pd_val < 0.60:
        return 'Alto'
    else:
        return 'Muy Alto'


def analisis_actuarial(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    y_proba: np.ndarray,
    umbral: float,
    meses_exp: int,
    lgd_override: dict | None = None,
) -> tuple:
    """
    Calcula PD · LGD · EAD · EL y ejecuta simulación Monte Carlo vectorizada.

    Parameters
    ----------
    X_test       : Features del set de prueba.
    y_test       : Etiquetas reales (0/1).
    y_proba      : Probabilidades de default predichas por XGBoost.
    umbral       : Umbral de rechazo de crédito (PD mínima para rechazar).
    meses_exp    : Meses de exposición para calcular EAD = ingreso × meses.
    lgd_override : Diccionario opcional para sobrescribir LGD por segmento.
                   Ejemplo: {'Bajo': 0.40, 'Moderado': 0.55, ...}

    Returns
    -------
    df_score, aprobados, rechazados, mc_results
    """
    lgd_map = {**LGD_TABLE, **(lgd_override or {})}

    df_score           = X_test.copy()
    df_score['PD']     = y_proba
    df_score['target'] = y_test.values

    # Segmentación de riesgo
    df_score['segmento'] = df_score['PD'].apply(_asignar_segmento)

    # LGD por segmento (valores regulatorios Basilea II)
    df_score['LGD'] = df_score['segmento'].map(lgd_map)

    # EAD = ingreso mensual × meses de exposición
    col_ing = 'ingreso' if 'ingreso' in df_score.columns else df_score.columns[0]
    df_score['ingreso_filled'] = df_score[col_ing].fillna(df_score[col_ing].median())
    df_score['EAD'] = df_score['ingreso_filled'] * meses_exp

    # Expected Loss
    df_score['EL'] = df_score['PD'] * df_score['LGD'] * df_score['EAD']

    # Decisión de crédito
    df_score['decision'] = np.where(df_score['PD'] >= umbral, 'RECHAZAR', 'APROBAR')

    aprobados  = df_score[df_score['decision'] == 'APROBAR']
    rechazados = df_score[df_score['decision'] == 'RECHAZAR']

    # ── Monte Carlo vectorizado (5,000 escenarios) ────────────────────────────
    # Cada fila es un escenario; cada columna es un acreditado aprobado.
    # np.random.binomial vectorizado elimina el loop Python → ~50x más rápido.
    np.random.seed(42)
    pd_vals  = aprobados['PD'].values
    ead_vals = aprobados['EAD'].values
    lgd_vals = aprobados['LGD'].values

    defaults_matrix = np.random.binomial(
        n=1, p=pd_vals, size=(N_SIMULACIONES, len(aprobados))
    )
    perdidas = (defaults_matrix * ead_vals * lgd_vals).sum(axis=1)

    mc_results = {
        'EL':           perdidas.mean(),
        'VaR_95':       np.percentile(perdidas, 95),
        'VaR_99':       np.percentile(perdidas, 99),
        'CVaR_95':      perdidas[perdidas >= np.percentile(perdidas, 95)].mean(),
        'CVaR_99':      perdidas[perdidas >= np.percentile(perdidas, 99)].mean(),
        'capital':      np.percentile(perdidas, 99) - perdidas.mean(),
        'perdidas_sim': perdidas,
    }

    return df_score, aprobados, rechazados, mc_results


def generar_excel(
    df_score: pd.DataFrame,
    scores: dict,
    mc: dict,
    auc_final: float,
    umbral: float,
) -> io.BytesIO:
    """
    Genera un reporte Excel descargable con 4 hojas:
        1. Scoring completo
        2. Comparación de modelos
        3. Resumen actuarial
        4. Segmentación de portfolio
    """
    output = io.BytesIO()
    orden  = ['Bajo', 'Moderado', 'Alto', 'Muy Alto']

    with pd.ExcelWriter(output, engine='openpyxl') as writer:

        # Hoja 1 — Scoring
        df_score[['PD', 'LGD', 'EAD', 'EL', 'decision', 'segmento', 'target']].to_excel(
            writer, sheet_name='Scoring', index=False
        )

        # Hoja 2 — Modelos
        pd.DataFrame({
            'Modelo':       list(scores.keys()),
            'AUC-ROC CV':   [f"{v['mean']:.4f}" for v in scores.values()],
            'Std':          [f"±{v['std']:.4f}"  for v in scores.values()],
        }).to_excel(writer, sheet_name='Modelos', index=False)

        # Hoja 3 — Resumen actuarial
        pd.DataFrame({
            'Métrica': [
                'AUC-ROC Final', 'Umbral de rechazo',
                'EL Simulada (5k runs)', 'VaR 95%', 'VaR 99%',
                'CVaR 95%', 'CVaR 99%', 'Capital Económico',
            ],
            'Valor': [
                f"{auc_final:.4f}", f"{umbral:.2f}",
                f"${mc['EL']:,.0f}",      f"${mc['VaR_95']:,.0f}",
                f"${mc['VaR_99']:,.0f}",  f"${mc['CVaR_95']:,.0f}",
                f"${mc['CVaR_99']:,.0f}", f"${mc['capital']:,.0f}",
            ],
        }).to_excel(writer, sheet_name='Resumen Actuarial', index=False)

        # Hoja 4 — Segmentación
        seg = df_score.groupby('segmento', observed=True).agg(
            Clientes          = ('PD',     'count'),
            PD_Promedio       = ('PD',     'mean'),
            LGD_Promedio      = ('LGD',    'mean'),
            EL_Total          = ('EL',     'sum'),
            Tasa_Default_Real = ('target', 'mean'),
        ).reindex([s for s in orden if s in df_score['segmento'].unique()])
        seg.to_excel(writer, sheet_name='Segmentación')

    output.seek(0)
    return output
