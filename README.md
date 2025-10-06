# ZoneBasedPricePrediction

Predicción de precio **basada en zonas** (PDH/PDL, USA IB, VWAP±σ, VAH/POC/VAL, HVN/LVN, Stacked Imbalances) con **event labeling** (rebote/ruptura), **calibración** de probabilidades y **objetivos estáticos/dinámicos** (siguiente nivel).

## Estructura
Ver `src/ppz/*` para el paquete y `notebooks/` para el flujo reproducible (00..06).

## Empezar
```bash
# Crear entorno e instalar
pip install -r requirements.txt
pip install -e .

# (Opcional) variables de entorno
cp .env.example .env
```

## Notebooks
- `00_data_prep.ipynb` → carga desde Mongo, pretreatment (MA→pct→min-max), ATR, VP.
- `01_zones_detect.ipynb` → PDH/PDL, USA IB, VWAP±σ, VAH/POC/VAL, HVN/LVN.
- `02_event_labeling.ipynb` → reglas de Rebote/Ruptura + MFE/MAE/Hit@X.
- `03_baselines_cls_reg.ipynb` → modelos base y calibración.
- `04_seq_models.ipynb` → LSTM/Transformer (opcional).
- `05_backtest_rules.ipynb` → reglas con EV.
- `06_dashboard_eval.ipynb` → panel de métricas/figuras.

## Scripts CLI
- `scripts/build_features.py`
- `scripts/make_events.py`
- `scripts/train_baseline.py`
- `scripts/backtest.py`

## Config
- `configs/params.yml` → umbrales (p, X, Y, H), ventanas, radios, etc.
- `configs/zones.yml` → pesos por tipo de zona y activaciones.
- `configs/mongo.toml` → conexión (usa `.env` para secretos).