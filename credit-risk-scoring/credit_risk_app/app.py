"""
app.py — Credit Risk Scoring System
Interfaz Streamlit: layout, sidebar y orquestación del pipeline.
Autor: Cesar Cruz
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve

from pipeline import (
    analisis_actuarial,
    crear_features,
    entrenar_modelos,
    generar_excel,
    get_feature_importance,
    limpiar_datos,
    mostrar_validacion,
    validar_dataset,
)

warnings.filterwarnings('ignore')

# ── Configuración de página ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Risk Scoring",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS personalizado ─────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
    .main { background-color: #0f1117; }

    .metric-card {
        background: #1a1d27; border: 1px solid #2d3147;
        border-radius: 8px; padding: 20px; text-align: center;
    }
    .metric-value {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 28px; font-weight: 500; color: #4fc3f7;
    }
    .metric-label {
        font-size: 12px; color: #8892b0;
        text-transform: uppercase; letter-spacing: 0.1em; margin-top: 4px;
    }
    .metric-card.danger  .metric-value { color: #ef5350; }
    .metric-card.success .metric-value { color: #66bb6a; }
    .metric-card.warning .metric-value { color: #ffa726; }

    .section-header {
        font-family: 'IBM Plex Mono', monospace; font-size: 11px;
        color: #4fc3f7; text-transform: uppercase; letter-spacing: 0.15em;
        border-bottom: 1px solid #2d3147; padding-bottom: 8px; margin: 24px 0 16px;
    }
    .validation-ok      { background:#0d2818; border-left:3px solid #66bb6a; padding:8px 14px; border-radius:0 4px 4px 0; font-size:13px; color:#66bb6a; margin:4px 0; }
    .validation-error   { background:#2d1010; border-left:3px solid #ef5350; padding:8px 14px; border-radius:0 4px 4px 0; font-size:13px; color:#ef5350; margin:4px 0; }
    .validation-warning { background:#2d2010; border-left:3px solid #ffa726; padding:8px 14px; border-radius:0 4px 4px 0; font-size:13px; color:#ffa726; margin:4px 0; }

    .stButton > button {
        background:#1565c0; color:white; border:none; border-radius:4px;
        font-family:'IBM Plex Sans',sans-serif; font-weight:500;
        width:100%; padding:12px;
    }
    .stButton > button:hover { background:#1976d2; }
</style>
""", unsafe_allow_html=True)


# ── Dataset de demo integrado ─────────────────────────────────────────────────
@st.cache_data
def generar_demo() -> pd.DataFrame:
    """
    Genera un dataset sintético de 3,000 créditos para demostración.
    Imita la estructura de Give Me Some Credit (Kaggle).
    Los parámetros reflejan distribuciones típicas de cartera retail.
    """
    rng = np.random.default_rng(42)
    n   = 3_000

    edad       = rng.integers(21, 75, n)
    ingreso    = np.clip(rng.lognormal(8.5, 0.6, n), 500, 50_000).round()
    utilizacion = np.clip(rng.beta(2, 5, n), 0, 1)
    deuda_ratio = np.clip(rng.lognormal(-0.5, 0.8, n), 0, 5)
    dependientes = rng.integers(0, 5, n)
    lineas      = rng.integers(1, 20, n)
    at30        = rng.integers(0, 5, n)
    at60        = rng.integers(0, 3, n)
    at90        = rng.integers(0, 3, n)

    # PD latente correlacionada con variables de comportamiento
    logit = (
        -3.5
        + 2.0 * utilizacion
        + 0.8 * (deuda_ratio / 5)
        + 0.5 * (at30 / 4)
        + 1.5 * (at90 / 3)
        - 0.3 * (np.log1p(ingreso) / 10)
    )
    pd_lat = 1 / (1 + np.exp(-logit))
    target = rng.binomial(1, pd_lat).astype(int)

    return pd.DataFrame({
        'SeriousDlqin2yrs':                          target,
        'RevolvingUtilizationOfUnsecuredLines':       utilizacion,
        'age':                                        edad,
        'NumberOfTime30-59DaysPastDueNotWorse':       at30,
        'DebtRatio':                                  deuda_ratio,
        'MonthlyIncome':                              ingreso,
        'NumberOfOpenCreditLinesAndLoans':            lineas,
        'NumberOfTimes90DaysLate':                    at90,
        'NumberOfTime60-89DaysPastDueNotWorse':       at60,
        'NumberOfDependents':                         dependientes,
    })


def _ax_style(ax):
    """Aplica estilo dark consistente a un eje matplotlib."""
    ax.set_facecolor('#1a1d27')
    ax.tick_params(colors='#8892b0')
    ax.xaxis.label.set_color('#8892b0')
    ax.yaxis.label.set_color('#8892b0')
    ax.title.set_color('#ccd6f6')
    for spine in ax.spines.values():
        spine.set_edgecolor('#2d3147')


def metric_card(value: str, label: str, cls: str = '') -> str:
    return (f'<div class="metric-card {cls}">'
            f'<div class="metric-value">{value}</div>'
            f'<div class="metric-label">{label}</div></div>')


# ════════════════════════════════════════════════════════════════════════════════
# LAYOUT
# ════════════════════════════════════════════════════════════════════════════════

st.markdown("## Credit Risk Scoring System")
st.markdown("*Análisis actuarial de riesgo crediticio · PD · LGD · EAD · Monte Carlo*")
st.divider()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙ Configuración")

    usar_demo = st.button("🎲 Usar dataset de demo", help="Carga 3,000 créditos sintéticos. No necesitas subir ningún archivo.")
    archivo   = st.file_uploader("O sube tu CSV de datos", type=['csv'])

    st.markdown("---")
    st.markdown("**Mapeo de columnas**")
    st.caption("Escribe el nombre exacto de cada columna en tu CSV")

    col_target  = st.text_input("Target (1=default, 0=no default)", "SeriousDlqin2yrs")
    col_ingreso = st.text_input("Ingreso mensual",                  "MonthlyIncome")
    col_deuda   = st.text_input("Ratio de deuda",                   "DebtRatio")
    col_edad    = st.text_input("Edad",                             "age")
    col_depend  = st.text_input("Dependientes",                     "NumberOfDependents")
    col_util    = st.text_input("Utilización crédito (0-1)",        "RevolvingUtilizationOfUnsecuredLines")
    col_at30    = st.text_input("Atrasos 30-59 días",               "NumberOfTime30-59DaysPastDueNotWorse")
    col_at60    = st.text_input("Atrasos 60-89 días",               "NumberOfTime60-89DaysPastDueNotWorse")
    col_at90    = st.text_input("Atrasos 90+ días",                 "NumberOfTimes90DaysLate")
    col_lineas  = st.text_input("Líneas de crédito abiertas",       "NumberOfOpenCreditLinesAndLoans")

    st.markdown("---")
    st.markdown("**Parámetros de negocio**")
    umbral    = st.slider("Umbral de rechazo (PD)",      0.05, 0.50, 0.15, 0.01,
                          help="PD mínima para rechazar un crédito")
    meses_exp = st.slider("Meses de exposición (EAD)",  1, 12, 3,
                          help="EAD = ingreso mensual × este número")

    st.markdown("---")
    st.markdown("**LGD por segmento** *(Basilea II, consumo no garantizado)*")
    lgd_bajo     = st.slider("LGD — Bajo riesgo",     0.20, 0.60, 0.35, 0.05)
    lgd_moderado = st.slider("LGD — Moderado",         0.30, 0.70, 0.50, 0.05)
    lgd_alto     = st.slider("LGD — Alto",             0.40, 0.80, 0.65, 0.05)
    lgd_muy_alto = st.slider("LGD — Muy Alto",         0.50, 0.95, 0.80, 0.05)

    correr = st.button("▶ Ejecutar análisis", type="primary")

lgd_override = {
    'Bajo': lgd_bajo, 'Moderado': lgd_moderado,
    'Alto': lgd_alto, 'Muy Alto': lgd_muy_alto,
}

mapeo = {
    'target':          col_target,
    'ingreso':         col_ingreso,
    'deuda_ratio':     col_deuda,
    'edad':            col_edad,
    'dependientes':    col_depend,
    'utilizacion':     col_util,
    'atraso_30':       col_at30,
    'atraso_60':       col_at60,
    'atraso_90':       col_at90,
    'lineas_abiertas': col_lineas,
}

# ── Estado del dataset ────────────────────────────────────────────────────────
if usar_demo:
    st.session_state['df_raw']    = generar_demo()
    st.session_state['demo_mode'] = True

if archivo:
    st.session_state['df_raw']    = pd.read_csv(archivo, index_col=0)
    st.session_state['demo_mode'] = False

df_raw = st.session_state.get('df_raw', None)

# ── Pantalla de bienvenida ────────────────────────────────────────────────────
if df_raw is None:
    st.info("← Sube tu CSV o usa el dataset de demo para comenzar.")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**1. Carga datos**")
        st.caption("Sube cualquier CSV de crédito o usa el demo integrado de 3,000 créditos sintéticos.")
    with c2:
        st.markdown("**2. Mapea columnas**")
        st.caption("Escribe el nombre exacto de cada columna o deja los valores predeterminados para el demo.")
    with c3:
        st.markdown("**3. Ejecuta**")
        st.caption("El sistema valida, entrena 3 modelos y genera reporte actuarial con Monte Carlo.")
    st.markdown("---")
    st.markdown("**Dataset real:** [Give Me Some Credit — Kaggle](https://www.kaggle.com/c/GiveMeSomeCredit)")

elif correr:
    # ── Progreso visible ──────────────────────────────────────────────────────
    progress = st.progress(0, text="Iniciando pipeline…")

    # Paso 1 — Validación
    progress.progress(10, text="Validando datos…")
    with st.expander("📋 Validación de datos", expanded=True):
        es_valido, errores, advertencias, info_msgs = validar_dataset(df_raw, mapeo)
        ok = mostrar_validacion(es_valido, errores, advertencias, info_msgs)

    if not ok:
        progress.empty()
        st.stop()

    # Paso 2 — Limpieza
    progress.progress(25, text="Limpiando datos…")
    df_clean = limpiar_datos(df_raw, mapeo)

    # Paso 3 — Features
    progress.progress(40, text="Creando features…")
    df_feat = crear_features(df_clean)

    # Paso 4 — Modelos
    progress.progress(55, text="Entrenando modelos (puede tomar 1-2 min)…")
    scores, modelo, X_test, y_test, y_proba, auc_final, features = entrenar_modelos(df_feat, umbral)

    # Paso 5 — Actuarial
    progress.progress(80, text="Ejecutando análisis actuarial y Monte Carlo…")
    df_score, aprobados, rechazados, mc = analisis_actuarial(
        X_test, y_test, y_proba, umbral, meses_exp, lgd_override
    )

    progress.progress(100, text="¡Análisis completado!")
    progress.empty()

    # ── MÉTRICAS PRINCIPALES ──────────────────────────────────────────────────
    st.markdown('<div class="section-header">Resultados del modelo</div>',
                unsafe_allow_html=True)

    defaults_capturados = ((df_score['decision'] == 'RECHAZAR') & (df_score['target'] == 1)).sum()
    total_defaults      = (df_score['target'] == 1).sum()
    recall_pct          = defaults_capturados / total_defaults * 100 if total_defaults > 0 else 0
    perdida_evitada     = rechazados[rechazados['target'] == 1]['EL'].sum()

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: st.markdown(metric_card(f"{auc_final:.4f}",         "AUC-ROC"),          unsafe_allow_html=True)
    with c2: st.markdown(metric_card(f"{len(aprobados):,}",      "Aprobados",   "success"), unsafe_allow_html=True)
    with c3: st.markdown(metric_card(f"{len(rechazados):,}",     "Rechazados",  "danger"),  unsafe_allow_html=True)
    with c4: st.markdown(metric_card(f"{recall_pct:.1f}%",       "Defaults capturados", "warning"), unsafe_allow_html=True)
    with c5: st.markdown(metric_card(f"${perdida_evitada/1e6:.1f}M", "Pérdida evitada", "success"), unsafe_allow_html=True)

    # ── COMPARACIÓN DE MODELOS ────────────────────────────────────────────────
    st.markdown('<div class="section-header">Comparación de modelos (AUC-ROC, CV 5 folds)</div>',
                unsafe_allow_html=True)
    df_scores_cmp = pd.DataFrame({
        'Modelo':   list(scores.keys()),
        'AUC-ROC':  [v['mean'] for v in scores.values()],
        'Std':      [v['std']  for v in scores.values()],
    }).sort_values('AUC-ROC', ascending=False)
    st.dataframe(
        df_scores_cmp.style.format({'AUC-ROC': '{:.4f}', 'Std': '±{:.4f}'}),
        use_container_width=True, hide_index=True,
    )

    # ── EVALUACIÓN DEL MODELO ─────────────────────────────────────────────────
    st.markdown('<div class="section-header">Evaluación del modelo XGBoost</div>',
                unsafe_allow_html=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.patch.set_facecolor('#0f1117')
    for ax in axes:
        _ax_style(ax)

    # ROC
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    axes[0].plot(fpr, tpr, color='#4fc3f7', linewidth=2, label=f'XGBoost (AUC={auc_final:.4f})')
    axes[0].plot([0, 1], [0, 1], '--', color='#2d3147', alpha=0.6)
    axes[0].fill_between(fpr, tpr, alpha=0.1, color='#4fc3f7')
    axes[0].set_xlabel('False Positive Rate'); axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title('Curva ROC')
    axes[0].legend(fontsize=9, labelcolor='#8892b0', facecolor='#1a1d27', edgecolor='#2d3147')

    # Precision-Recall
    prec, rec, _ = precision_recall_curve(y_test, y_proba)
    baseline = y_test.mean()
    axes[1].plot(rec, prec, color='#ef5350', linewidth=2)
    axes[1].axhline(baseline, color='#2d3147', linestyle='--', alpha=0.8,
                    label=f'Baseline ({baseline:.3f})')
    axes[1].set_xlabel('Recall'); axes[1].set_ylabel('Precision')
    axes[1].set_title('Curva Precision-Recall')
    axes[1].legend(fontsize=9, labelcolor='#8892b0', facecolor='#1a1d27', edgecolor='#2d3147')

    # Distribución de PD
    axes[2].hist(y_proba[y_test == 0], bins=50, alpha=0.6, color='#4fc3f7', label='No Default', density=True)
    axes[2].hist(y_proba[y_test == 1], bins=50, alpha=0.6, color='#ef5350', label='Default',    density=True)
    axes[2].axvline(umbral, color='#ffa726', linewidth=2, linestyle='--', label=f'Umbral ({umbral})')
    axes[2].set_xlabel('Probabilidad de Default (PD)'); axes[2].set_ylabel('Densidad')
    axes[2].set_title('Distribución de PD')
    axes[2].legend(fontsize=9, labelcolor='#8892b0', facecolor='#1a1d27', edgecolor='#2d3147')

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # ── FEATURE IMPORTANCE ────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Importancia de variables — XGBoost</div>',
                unsafe_allow_html=True)

    df_imp = get_feature_importance(modelo, features)
    top_n  = min(10, len(df_imp))

    fig_imp, ax_imp = plt.subplots(figsize=(10, 4))
    fig_imp.patch.set_facecolor('#0f1117')
    _ax_style(ax_imp)

    top = df_imp.head(top_n).iloc[::-1]  # invertir para barh ascendente
    bars = ax_imp.barh(top['Feature'], top['Importancia'], color='#4fc3f7', alpha=0.85)
    ax_imp.bar_label(bars, fmt='%.3f', padding=4, color='#8892b0', fontsize=9)
    ax_imp.set_xlabel('Importancia (gain)')
    ax_imp.set_title(f'Top {top_n} variables más importantes')
    plt.tight_layout()
    st.pyplot(fig_imp)
    plt.close()

    # ── ANÁLISIS ACTUARIAL / MONTE CARLO ──────────────────────────────────────
    st.markdown('<div class="section-header">Análisis actuarial — Monte Carlo (5,000 simulaciones)</div>',
                unsafe_allow_html=True)

    a1, a2, a3, a4 = st.columns(4)
    with a1: st.markdown(metric_card(f"${mc['EL']/1e6:.2f}M",      "Expected Loss"),          unsafe_allow_html=True)
    with a2: st.markdown(metric_card(f"${mc['VaR_99']/1e6:.2f}M",  "VaR 99%",    "warning"),  unsafe_allow_html=True)
    with a3: st.markdown(metric_card(f"${mc['CVaR_99']/1e6:.2f}M", "CVaR 99%",   "danger"),   unsafe_allow_html=True)
    with a4: st.markdown(metric_card(f"${mc['capital']/1e6:.2f}M",  "Capital económico"),      unsafe_allow_html=True)

    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 4))
    fig2.patch.set_facecolor('#0f1117')
    for ax in axes2:
        _ax_style(ax)

    perdidas = mc['perdidas_sim']
    axes2[0].hist(perdidas / 1e6, bins=60, color='#4fc3f7', alpha=0.7, edgecolor='#0f1117', linewidth=0.3)
    axes2[0].axvline(mc['EL']     / 1e6, color='#66bb6a', linewidth=2, linestyle='--', label=f"EL=${mc['EL']/1e6:.1f}M")
    axes2[0].axvline(mc['VaR_99'] / 1e6, color='#ef5350', linewidth=2, linestyle='--', label=f"VaR99%=${mc['VaR_99']/1e6:.1f}M")
    axes2[0].set_xlabel('Pérdida ($M)'); axes2[0].set_ylabel('Frecuencia')
    axes2[0].set_title('Distribución de pérdidas simuladas')
    axes2[0].legend(fontsize=9, labelcolor='#8892b0', facecolor='#1a1d27', edgecolor='#2d3147')

    sorted_p = np.sort(perdidas) / 1e6
    cdf = np.arange(1, len(sorted_p) + 1) / len(sorted_p)
    axes2[1].plot(sorted_p, cdf, color='#4fc3f7', linewidth=2)
    axes2[1].axhline(0.99, color='#ef5350', linestyle='--', alpha=0.7, label='99%')
    axes2[1].axhline(0.95, color='#ffa726', linestyle='--', alpha=0.7, label='95%')
    axes2[1].set_xlabel('Pérdida ($M)'); axes2[1].set_ylabel('Probabilidad acumulada')
    axes2[1].set_title('Función de distribución acumulada (CDF)')
    axes2[1].legend(fontsize=9, labelcolor='#8892b0', facecolor='#1a1d27', edgecolor='#2d3247')

    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

    # ── SEGMENTACIÓN ──────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Segmentación del portfolio</div>',
                unsafe_allow_html=True)

    orden  = ['Bajo', 'Moderado', 'Alto', 'Muy Alto']
    seg_df = df_score.groupby('segmento', observed=True).agg(
        Clientes     = ('PD',     'count'),
        PD_Promedio  = ('PD',     'mean'),
        LGD_Promedio = ('LGD',    'mean'),
        EL_Total     = ('EL',     'sum'),
        Default_Real = ('target', 'mean'),
    ).reindex([s for s in orden if s in df_score['segmento'].unique()]).round(4)

    st.dataframe(
        seg_df.style.format({
            'PD_Promedio':  '{:.2%}',
            'LGD_Promedio': '{:.2%}',
            'EL_Total':     '${:,.0f}',
            'Default_Real': '{:.2%}',
        }),
        use_container_width=True,
    )

    # ── EXPORTAR ──────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Exportar reporte</div>',
                unsafe_allow_html=True)

    excel_data = generar_excel(df_score, scores, mc, auc_final, umbral)
    st.download_button(
        label     = "⬇ Descargar reporte Excel",
        data      = excel_data,
        file_name = "credit_risk_report.xlsx",
        mime      = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

else:
    if df_raw is not None:
        mode = "demo" if st.session_state.get('demo_mode') else "CSV cargado"
        st.info(f"Dataset listo ({mode}: {len(df_raw):,} filas). Presiona **▶ Ejecutar análisis**.")
    else:
        st.info("← Configura las columnas y presiona **Ejecutar análisis**.")
