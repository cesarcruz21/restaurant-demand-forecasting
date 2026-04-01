"""
pipeline/modeling.py
Entrenamiento de modelos y evaluación para el Credit Risk Scoring System.
Autor: Cesar Cruz
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import (StratifiedKFold, cross_val_score,
                                     train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb


# Features candidatos en orden de importancia esperada
POSIBLES_FEATURES = [
    'utilizacion', 'edad', 'atraso_30', 'deuda_ratio', 'ingreso',
    'lineas_abiertas', 'atraso_90', 'atraso_60', 'dependientes',
    'total_past_due', 'high_utilization', 'clean_history',
    'debt_to_income', 'age_segment', 'income_per_dependent',
]


def entrenar_modelos(df_feat: pd.DataFrame, umbral: float) -> tuple:
    """
    Entrena Logística, Random Forest y XGBoost con validación cruzada 5-fold.

    Returns
    -------
    scores, modelo_xgb, X_test, y_test, y_proba, auc_final, features
    """
    features = [f for f in POSIBLES_FEATURES if f in df_feat.columns]

    X = df_feat[features]
    y = df_feat['target']
    scale_pos = (y == 0).sum() / (y == 1).sum()

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Pipelines
    pipe_lr = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(
            class_weight='balanced', max_iter=1_000, random_state=42
        )),
    ])
    pipe_rf = Pipeline([
        ('model', RandomForestClassifier(
            n_estimators=200, max_depth=8,
            class_weight='balanced', random_state=42, n_jobs=-1
        )),
    ])
    pipe_xgb = Pipeline([
        ('model', xgb.XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            scale_pos_weight=scale_pos, random_state=42, verbosity=0,
        )),
    ])

    # Validación cruzada
    scores = {}
    for nombre, pipe in [
        ('Logística',     pipe_lr),
        ('Random Forest', pipe_rf),
        ('XGBoost',       pipe_xgb),
    ]:
        s = cross_val_score(pipe, X, y, cv=skf, scoring='roc_auc', n_jobs=-1)
        scores[nombre] = {'mean': s.mean(), 'std': s.std()}

    # Entrenamiento final XGBoost en split 80/20
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    pipe_xgb.fit(X_train, y_train)
    y_proba   = pipe_xgb.predict_proba(X_test)[:, 1]
    auc_final = roc_auc_score(y_test, y_proba)

    return scores, pipe_xgb, X_test, y_test, y_proba, auc_final, features


def get_feature_importance(modelo_xgb: Pipeline, features: list) -> pd.DataFrame:
    """
    Extrae y ordena la importancia de features del modelo XGBoost.

    Returns
    -------
    DataFrame con columnas ['Feature', 'Importancia'] ordenado descendente.
    """
    xgb_model = modelo_xgb.named_steps['model']
    importances = xgb_model.feature_importances_

    df_imp = pd.DataFrame({
        'Feature':     features,
        'Importancia': importances,
    }).sort_values('Importancia', ascending=False).reset_index(drop=True)

    return df_imp
