# Análisis y Predicción de Ventas de Restaurante

> Proyecto de Data Science aplicado a la optimización de inventario, con **análisis actuarial de riesgo** integrado.

**Autor:** Cesar Cruz — Estudiante de Actuaría, enfoque en Análisis de Datos  
**Stack:** Python · scikit-learn · XGBoost · Pandas · SciPy · Matplotlib

---

## ¿Qué hace este proyecto?

Dado un historial de ventas diarias de un restaurante, este notebook:

1. **Limpia y prepara** los datos automáticamente, manejando formatos inconsistentes y valores atípicos (IQR).
2. **Explora** patrones semanales, estacionalidad y correlaciones con variables climáticas y festivos.
3. **Compara tres modelos predictivos** — Regresión Lineal, Random Forest y XGBoost — usando validación cruzada con `TimeSeriesSplit` para respetar el orden temporal.
4. **Cuantifica la incertidumbre** de las predicciones con intervalos de confianza al 80% derivados de los árboles individuales del Random Forest.
5. **Aplica análisis actuarial de riesgo:** Value at Risk (VaR), Expected Shortfall (CVaR) y simulación Monte Carlo para estimar la probabilidad de stockout.
6. **Calcula el stock de seguridad óptimo** para niveles de servicio del 90%, 95% y 99%.
7. **Exporta reportes** en JSON y Excel, incluyendo un CSV de muestra para reproducibilidad.

---

## Estructura del repositorio

```
├── Prediccion_de_ventas_FINAL.ipynb   # Notebook principal
├── ventas_muestra.csv                 # Datos de ejemplo para ejecutar el notebook
├── ventas_restaurante.csv             # (No incluido) — reemplazar con datos reales
├── inventario_tienda.csv              # (Opcional) — datos de inventario reales
├── reportes/                          # Generado al ejecutar el notebook
│   ├── reporte_YYYYMMDD.json
│   └── datos_YYYYMMDD.xlsx
└── README.md
```

---

## Instalación y uso

```bash
# 1. Clonar el repositorio
git clone https://github.com/tu-usuario/prediccion-ventas-restaurante.git
cd prediccion-ventas-restaurante

# 2. Instalar dependencias
pip install pandas numpy matplotlib seaborn scikit-learn xgboost scipy holidays openpyxl

# 3. (Opcional) Agregar datos reales
cp tu_archivo.csv ventas_restaurante.csv

# 4. Ejecutar el notebook
jupyter notebook Prediccion_de_ventas_FINAL.ipynb
```

Si no se proporciona `ventas_restaurante.csv`, el notebook genera datos simulados automáticamente y puede ejecutarse completo sin ningún archivo adicional.

---

## Metodología

### Modelos de predicción

| Modelo | Validación | Métrica principal |
|---|---|---|
| Regresión Lineal | TimeSeriesSplit (5 folds) | MAE, R² |
| Random Forest | TimeSeriesSplit (5 folds) | MAE, R², intervalos de predicción |
| XGBoost | TimeSeriesSplit (5 folds) | MAE, R² |

Se usa `TimeSeriesSplit` en lugar de K-Fold estándar para evitar *data leakage* temporal: el modelo nunca se entrena con datos futuros.

### Análisis actuarial de riesgo

Este es el elemento diferenciador del proyecto desde la perspectiva actuarial:

- **Intervalos de predicción:** se extraen las predicciones de cada árbol del Random Forest para construir una distribución empírica por observación, calculando los percentiles 10 y 90 como banda de incertidumbre.
- **Value at Risk (VaR 5%):** nivel de ventas que solo se supera en el 5% de los días peores. Se estima de forma empírica y con ajuste paramétrico (Normal y Log-Normal).
- **Expected Shortfall / CVaR 5%:** venta promedio condicionada a estar en el 5% de peores días. Métrica estándar en gestión de riesgos.
- **Stock de seguridad óptimo:** `SS = z × σ_demanda × √(lead_time)`, calculado para niveles de servicio 90/95/99%.
- **Simulación Monte Carlo (10,000 réplicas):** probabilidad empírica de stockout en una semana bajo distintos niveles de stock de seguridad.

---

## Resultados representativos (datos de muestra)

> Los valores exactos varían con datos reales. Estos son resultados típicos con el dataset de muestra incluido.

| Métrica | Valor |
|---|---|
| R² Random Forest (CV) | ~0.82 |
| MAE Random Forest (CV) | ~$180 |
| Cobertura intervalo 80% | ~78–82% |
| VaR 5% ventas | ~$2,100 |
| Stock de seguridad (95%) | ~$420 uds. |

---

## Variables utilizadas

| Variable | Tipo | Descripción |
|---|---|---|
| `dia_num` | Numérica | Índice temporal |
| `clientes` | Numérica | Clientes atendidos ese día |
| `temp_promedio` | Numérica | Temperatura media (°C) |
| `lluvia` | Numérica | Precipitación estimada |
| `es_festivo` | Booleana | Festivo oficial México |
| `mes` | Numérica | Mes del año |
| `es_fin_de_semana` | Booleana | Sábado o domingo |

---

## Formato de datos de entrada

```csv
fecha,ventas,clientes
2025-01-01,3450,120
2025-01-02,2980,98
2025-01-03,4100,145
```

El notebook normaliza automáticamente nombres de columnas alternativos (`date`, `total`, `customers`, etc.) y maneja separadores distintos.

---

## Tecnologías

- **Python 3.10+**
- `pandas`, `numpy` — manipulación de datos
- `scikit-learn` — modelos y validación
- `xgboost` — gradient boosting
- `scipy` — ajuste de distribuciones y estadística actuarial
- `matplotlib`, `seaborn` — visualización
- `holidays` — festivos oficiales de México
- `openpyxl` — exportación a Excel

---

## Próximos pasos

- [ ] Integrar modelos de series de tiempo (Prophet, ARIMA)
- [ ] Dashboard interactivo con Streamlit o Power BI
- [ ] Datos climáticos reales vía API (Open-Meteo)
- [ ] Análisis de rentabilidad por categoría de producto

---

*Proyecto desarrollado como parte del portafolio académico — Actuaría, análisis de datos.*
