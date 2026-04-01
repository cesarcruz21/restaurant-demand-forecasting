"""
tests/test_pipeline.py
Tests unitarios para pipeline/validation.py y pipeline/preprocessing.py
Ejecutar con: pytest tests/ -v
"""

import numpy as np
import pandas as pd
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.preprocessing import limpiar_datos, crear_features
from pipeline.validation import validar_dataset


# ── Fixtures ──────────────────────────────────────────────────────────────────

MAPEO_STD = {
    'target':         'SeriousDlqin2yrs',
    'ingreso':        'MonthlyIncome',
    'deuda_ratio':    'DebtRatio',
    'edad':           'age',
    'dependientes':   'NumberOfDependents',
    'utilizacion':    'RevolvingUtilizationOfUnsecuredLines',
    'atraso_30':      'NumberOfTime30-59DaysPastDueNotWorse',
    'atraso_60':      'NumberOfTime60-89DaysPastDueNotWorse',
    'atraso_90':      'NumberOfTimes90DaysLate',
    'lineas_abiertas':'NumberOfOpenCreditLinesAndLoans',
}


def _df_valido(n=1000, seed=42):
    """Dataset sintético válido para pruebas."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        'SeriousDlqin2yrs':                         rng.integers(0, 2, n),
        'MonthlyIncome':                             rng.lognormal(8.5, 0.6, n),
        'DebtRatio':                                 rng.uniform(0, 2, n),
        'age':                                       rng.integers(22, 75, n),
        'NumberOfDependents':                        rng.integers(0, 5, n),
        'RevolvingUtilizationOfUnsecuredLines':      rng.uniform(0, 1, n),
        'NumberOfTime30-59DaysPastDueNotWorse':      rng.integers(0, 5, n),
        'NumberOfTime60-89DaysPastDueNotWorse':      rng.integers(0, 3, n),
        'NumberOfTimes90DaysLate':                   rng.integers(0, 3, n),
        'NumberOfOpenCreditLinesAndLoans':           rng.integers(1, 20, n),
    })


# ════════════════════════════════════════════════════════════════════════════════
# TESTS — validation.py
# ════════════════════════════════════════════════════════════════════════════════

class TestValidarDataset:

    def test_dataset_valido_pasa(self):
        """Un dataset correcto debe validar sin errores."""
        df = _df_valido()
        es_valido, errores, _, _ = validar_dataset(df, MAPEO_STD)
        assert es_valido is True
        assert errores == []

    def test_columna_faltante_genera_error(self):
        """Si falta una columna mapeada debe retornar error."""
        df = _df_valido().drop(columns=['MonthlyIncome'])
        es_valido, errores, _, _ = validar_dataset(df, MAPEO_STD)
        assert es_valido is False
        assert any('MonthlyIncome' in e for e in errores)

    def test_target_con_valores_invalidos(self):
        """Target con valores distintos de 0/1 debe generar error."""
        df = _df_valido()
        df['SeriousDlqin2yrs'] = 5  # valor inválido
        es_valido, errores, _, _ = validar_dataset(df, MAPEO_STD)
        assert es_valido is False
        assert any('0 y 1' in e for e in errores)

    def test_dataset_pequeño_genera_error(self):
        """Dataset con menos de 500 filas debe generar error."""
        df = _df_valido(n=100)
        es_valido, errores, _, _ = validar_dataset(df, MAPEO_STD)
        assert es_valido is False
        assert any('500' in e for e in errores)

    def test_dataset_mediano_genera_advertencia(self):
        """Dataset entre 500 y 2000 filas debe generar advertencia, no error."""
        df = _df_valido(n=800)
        es_valido, errores, advertencias, _ = validar_dataset(df, MAPEO_STD)
        assert es_valido is True
        assert errores == []
        assert any('2,000' in a or '2000' in a for a in advertencias)

    def test_nulos_excesivos_generan_error(self):
        """Columna con más del 50% de nulos debe generar error."""
        df = _df_valido()
        df.loc[:600, 'MonthlyIncome'] = np.nan  # ~60% nulos
        es_valido, errores, _, _ = validar_dataset(df, MAPEO_STD)
        assert es_valido is False
        assert any('nulos' in e.lower() for e in errores)

    def test_nulos_moderados_generan_advertencia(self):
        """Columna con 20-50% de nulos debe generar advertencia."""
        df = _df_valido()
        df.loc[:250, 'MonthlyIncome'] = np.nan  # ~25% nulos
        es_valido, errores, advertencias, _ = validar_dataset(df, MAPEO_STD)
        assert es_valido is True
        assert any('MonthlyIncome' in a for a in advertencias)

    def test_tasa_default_baja_genera_advertencia(self):
        """Tasa de default < 1% debe generar advertencia."""
        df = _df_valido()
        df['SeriousDlqin2yrs'] = 0
        df.loc[:3, 'SeriousDlqin2yrs'] = 1  # 0.3% de defaults
        _, _, advertencias, _ = validar_dataset(df, MAPEO_STD)
        assert any('default' in a.lower() for a in advertencias)

    def test_edad_fuera_de_rango_genera_advertencia(self):
        """Edades negativas o > 120 deben generar advertencia."""
        df = _df_valido()
        df.loc[0, 'age'] = -5
        _, _, advertencias, _ = validar_dataset(df, MAPEO_STD)
        assert any('edad' in a.lower() for a in advertencias)

    def test_ingresos_negativos_generan_advertencia(self):
        """Ingresos negativos deben generar advertencia."""
        df = _df_valido()
        df.loc[:10, 'MonthlyIncome'] = -100
        _, _, advertencias, _ = validar_dataset(df, MAPEO_STD)
        assert any('negativo' in a.lower() for a in advertencias)


# ════════════════════════════════════════════════════════════════════════════════
# TESTS — preprocessing.py
# ════════════════════════════════════════════════════════════════════════════════

class TestLimpiarDatos:

    def test_renombra_columnas_correctamente(self):
        """Las columnas deben renombrarse a nombres internos."""
        df = _df_valido()
        df_clean = limpiar_datos(df, MAPEO_STD)
        assert 'target'   in df_clean.columns
        assert 'ingreso'  in df_clean.columns
        assert 'edad'     in df_clean.columns

    def test_imputa_nulos_con_mediana(self):
        """Los nulos en ingreso y dependientes deben imputarse."""
        df = _df_valido()
        df.loc[:100, 'MonthlyIncome'] = np.nan
        df_clean = limpiar_datos(df, MAPEO_STD)
        assert df_clean['ingreso'].isnull().sum() == 0

    def test_corrige_edad_cero(self):
        """Edades de 0 deben reemplazarse por la mediana."""
        df = _df_valido()
        df.loc[0, 'age'] = 0
        df_clean = limpiar_datos(df, MAPEO_STD)
        assert df_clean['edad'].min() > 0

    def test_clipea_utilizacion(self):
        """Utilización no puede superar 1.0 después de limpiar."""
        df = _df_valido()
        df.loc[:10, 'RevolvingUtilizationOfUnsecuredLines'] = 999.0
        df_clean = limpiar_datos(df, MAPEO_STD)
        assert df_clean['utilizacion'].max() <= 1.0

    def test_winsoriza_deuda_ratio(self):
        """DebtRatio debe cliparse en el percentil 99."""
        df = _df_valido()
        df.loc[0, 'DebtRatio'] = 1_000_000.0
        df_clean = limpiar_datos(df, MAPEO_STD)
        p99_original = df['DebtRatio'].quantile(0.99)
        assert df_clean['deuda_ratio'].max() <= p99_original * 1.01

    def test_sin_nulos_al_final(self):
        """No deben quedar nulos después de la limpieza."""
        df = _df_valido()
        df.loc[:50, 'MonthlyIncome']    = np.nan
        df.loc[:30, 'NumberOfDependents'] = np.nan
        df_clean = limpiar_datos(df, MAPEO_STD)
        assert df_clean.isnull().sum().sum() == 0

    def test_conserva_numero_de_filas(self):
        """La limpieza no debe eliminar filas."""
        df = _df_valido()
        df_clean = limpiar_datos(df, MAPEO_STD)
        assert len(df_clean) == len(df)


class TestCrearFeatures:

    def _df_limpio(self, n=500):
        df = _df_valido(n)
        return limpiar_datos(df, MAPEO_STD)

    def test_crea_total_past_due(self):
        """total_past_due debe crearse y ser >= 0."""
        df_feat = crear_features(self._df_limpio())
        assert 'total_past_due' in df_feat.columns
        assert (df_feat['total_past_due'] >= 0).all()

    def test_crea_clean_history(self):
        """clean_history debe ser binario (0 o 1)."""
        df_feat = crear_features(self._df_limpio())
        assert 'clean_history' in df_feat.columns
        assert set(df_feat['clean_history'].unique()).issubset({0, 1})

    def test_clean_history_consistente_con_past_due(self):
        """clean_history=1 solo cuando total_past_due=0."""
        df_feat = crear_features(self._df_limpio())
        mask_clean = df_feat['clean_history'] == 1
        assert (df_feat.loc[mask_clean, 'total_past_due'] == 0).all()

    def test_crea_high_utilization(self):
        """high_utilization debe ser binario."""
        df_feat = crear_features(self._df_limpio())
        assert 'high_utilization' in df_feat.columns
        assert set(df_feat['high_utilization'].unique()).issubset({0, 1})

    def test_high_utilization_correcto(self):
        """high_utilization=1 solo cuando utilizacion > 0.8."""
        df_feat = crear_features(self._df_limpio())
        mask_high = df_feat['high_utilization'] == 1
        assert (df_feat.loc[mask_high, 'utilizacion'] > 0.8).all()

    def test_crea_income_per_dependent(self):
        """income_per_dependent debe ser positivo."""
        df_feat = crear_features(self._df_limpio())
        assert 'income_per_dependent' in df_feat.columns
        assert (df_feat['income_per_dependent'] > 0).all()

    def test_age_segment_en_rango(self):
        """age_segment debe estar entre 0 y 3."""
        df_feat = crear_features(self._df_limpio())
        assert 'age_segment' in df_feat.columns
        assert df_feat['age_segment'].between(0, 3).all()

    def test_sin_nulos_en_features_creadas(self):
        """Las nuevas features no deben tener nulos."""
        df_feat = crear_features(self._df_limpio())
        nuevas = ['total_past_due', 'clean_history', 'high_utilization',
                  'income_per_dependent', 'debt_to_income', 'age_segment']
        for col in nuevas:
            if col in df_feat.columns:
                assert df_feat[col].isnull().sum() == 0, f"{col} tiene nulos"

    def test_no_elimina_filas(self):
        """El feature engineering no debe eliminar filas."""
        df_clean = self._df_limpio()
        df_feat  = crear_features(df_clean)
        assert len(df_feat) == len(df_clean)
