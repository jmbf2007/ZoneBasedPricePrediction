# Predicción de Precio Basada en Zonas

> Enfoque pragmático orientado a *eventos y ubicaciones* donde el mercado tiende a reaccionar, en lugar de intentar predecir cada vela.

---

## 1) Idea central

- No buscamos “predecir todas las velas”, sino evaluar **probabilidades y magnitudes de reacción** cuando el precio **llega o se acerca** a **zonas relevantes** (S/R, rangos, niveles estadísticos, clústeres de volumen, etc.).
- Formulamos el problema como:
  1. **Clasificación**: ¿Rebota o rompe la zona? (dirección y probabilidad)
  2. **Regresión**: ¿Cuál es la **extensión esperada** del movimiento (en ticks) condicionado a ese evento?
  3. **Calibración**: Asegurar que la probabilidad predicha refleje la frecuencia real (para tomar decisiones de riesgo).

---

## 2) Catálogo de Zonas (MVP)

1. **Estructurales (D1 / intradía)**
   - Máximo/ mínimo del día anterior (PDH/PDL)
   - Máximo/ mínimo de la sesión actual (HH/LL intradía)
   - **IB High / IB Low** (Initial Balance) – 1ª hora intradía
   - High/Low de apertura de sesión Europa/USA (cambios de sesión)
2. **VWAP y estadísticos**
   - VWAP actual y bandas k·σ (σ intradía)
   - AVWAP de apertura de sesión/semana/mes
3. **Volumen y microestructura**
   - **HVN/LVN** por vela o por tramo (del *Volume Profile*)
   - **MVC** (precio de mayor volumen en vela o agregado)
   - Zonas de **absorción** (delta extremo + rechazo)

> Nota: Empezar con 5–7 tipos de zona para rapidez; ampliar según resultados.

---

## 3) Etiquetado de Eventos

Para cada visita del precio a una zona Z en t₀:

- **Ventana de contexto**: últimas `L = 60–120` velas (5m) con features ya disponibles.
- **Ventana de evaluación**: `H = 6–24` velas posteriores a t₀.
- Etiquetas:
  - **Rebote** si el precio avanza ≥ `X` ticks en dirección opuesta a la llegada **sin** invalidar Z (ej. no supera un umbral de penetración `p` ticks).
  - **Ruptura** si penetra > `p` ticks y alcanza un **mínimo movimiento de confirmación** `Y` ticks a favor de la ruptura.
  - **Sin evento** si no cumple ninguno.
- **Magnitudes**:
  - `M_ext`: máxima excursión favorable (MFE) y desfavorable (MAE) en ticks dentro de H.
  - `T_hit`: tiempo hasta alcanzar `Y` ticks.

Parámetros iniciales (ES 5m): `p=4–6 ticks`, `X=8–12`, `Y=12–20`, `H=12`.

---

## 4) Feature set (condicionadas al acercamiento a Z)

- **Geométricas respecto a Z**
  - Distancia en ticks a Z (actual y media de los últimos k)
  - Nº de toques previos, confluencia de zonas (Z1∩Z2)
- **Tendencia/volatilidad**
  - Pendiente de precio y de VWAP (y VWAP±σ) en las últimas `m` velas
  - **ATR** (14) y cuantiles intradía; régimen de volatilidad (bajo/medio/alto)
- **Order flow**
  - **Delta** y su *z-score* local
  - **Skew** del **Volume Profile** de las últimas `r` velas; concentración en HVN/LVN
  - **MVC** relativo a rango de vela
- **Tiempo/sesión**
  - `sin_time`, `cos_time`, `asia/eu/usa` one‑hot
- **Contexto reciente**
  - Racha direccional, amplitud de swings, compresión (rango true/ATR)

> Reutilizar *pipeline* de pretratamiento y VP/MVC existentes; añadir *features* de distancia a zonas y toques.

---

## 5) Modelado

**Objetivo A (Clasificación)**: `P(rebote)`, `P(ruptura)` en Z

- Modelos base rápidos: `LogReg`, `HistGradientBoosting`, `XGBoost` (baseline fuerte)
- Secuenciales: `LSTM` corta / `Transformer` ligero (si añade valor)
- Calibración: `Platt` o `Isotonic` (Brier/ECE)

**Objetivo B (Regresión)**: `E[M_ext | evento, Z]` y/o cuantiles (Q10/Q50/Q90)

- `HGBRegressor` (robusto), o `Quantile Regression`

**Estrategia por régimen**

- Entrenar por **sesión** (Asia/Europa/USA)
- Filtrar por **rango/volatilidad** (cuantiles ATR)
- *Curriculum*: empezar con IBH/IBL + PDH/PDL + VWAP; luego HVN/LVN/MVC

---

## 6) Métricas y selección

- **Clasificación**: AUC‑ROC, **AUC‑PR**, **Brier**, **ECE** (calibración)
- **Regresión**: MAE/RMSE en ticks, **Pinball loss** (cuantiles)
- **Métrica de decisión**: **EV por trade** = `p_win·payoff − (1−p_win)·risk`
- **Backtest de reglas** sobre señales (umbral de prob. + filtros de riesgo)

---

## 7) Pipeline de datos (MongoDB → dataset de eventos)

1. **Carga** (`get_data_ticker`) por rango de fechas/TF
2. **Pretreatment** existente (MA10 → pct → normalización; VP interpolado, ATR)
3. **Detección de zonas** (módulo nuevo):
   - PDH/PDL, IBH/IBL, VWAP±kσ, AVWAPs, HVN/LVN, MVC agregados
4. **Detector de llegadas a Z** + **constructor de ejemplos** (contexto L, etiqueta dentro de H)
5. **Splits temporales**: train/val/test por bloques de fechas y por sesión

---

## 8) Notebook plan (ipynb)

1. `00_data_prep.ipynb` – carga, pretreatment, ATR/VP, guardado parquet
2. `01_zones_detect.ipynb` – cálculo de Z (PDH/PDL, IB, VWAP, HVN/LVN)
3. `02_event_labeling.ipynb` – reglas Rebote/Ruptura/None + magnitudes
4. `03_baselines_cls_reg.ipynb` – HGB/XGBoost + calibración
5. `04_seq_models.ipynb` – LSTM/Transformer ligeros (opcional)
6. `05_backtest_rules.ipynb` – política de entrada/salida y EV
7. `06_dashboard_eval.ipynb` – gráficos (mosaico, dispersiones, calibración)

---

## 9) Reglas iniciales de Backtest (MVP)

- Entrada si `P(evento|Z) ≥ θ` y **confluencia** (Z ∩ VWAP±σ o Z ∩ HVN)
- Stop = `s` ticks más allá de Z; TP por cuantiles esperados (`Q50/Q75`)
- Filtro de sesión (USA/Europa) y volatilidad (ATR quantile ≥ q)
- Salida por tiempo (`T_hit` máx) o por rastro de prob. decreciente

---

## 10) Entregables

- Datasets de eventos por tipo de zona
- Modelos calibrados (pickle/keras) + reportes métricas
- Cuaderno de *backtest* con EV y sensibilidad a `θ, s, TP`
- Gráficos plotly reutilizando funciones de `MLfunctions_plot`

---

## 11) Siguientes pasos (esta semana)

1. Implementar **módulo de Zonas**: PDH/PDL, IBH/IBL, VWAP±σ (k∈{1,2})
2. Implementar **labeling** Rebote/Ruptura con `H=12`, `p=6`, `X=10`, `Y=16`
3. Generar **dataset de eventos** (2023–2024) y baseline `HGB`
4. Reporte de **calibración** y **EV** por zona y sesión

---

### Notas de integración

- Reusar funciones existentes de pretreat/VP/ATR y plotly.
- Mantener *feature store* compacto (20–40 features/evento) para velocidad.
- Guardar *experimentos* con `mlflow` o JSON + carpeta `runs/` (simple).



---

# Anexo A · Datos de partida (base schema)

Estos son los **campos mínimos** que esperamos en el `DataFrame` base (por vela):

- **Time**: timestamp de apertura de la vela.
- **Open, High, Low, Close**: precios definitorios de la vela.
- **MVC**: *Maximum Volume Cluster*, precio con mayor volumen de la vela.
- **Volume**: volumen total negociado (Ask + Bid) en la vela.
- **Delta**: `Ask_total − Bid_total` de la vela.
- **Ask[ ]**: lista/array del volumen ejecutado por compra en cada nivel de precio (de High → Low).
- **Bid[ ]**: lista/array del volumen ejecutado por venta en cada nivel de precio (de High → Low).
- **NewSession**: `True` en la primera vela de cada sesión (resetea índices intradía, IB, PDH/PDL, VWAP sesión…).
- **NewWeek**: `True` en la primera vela de la semana (niveles y AVWAP semanales).
- **NewMonth**: `True` en la primera vela del mes (niveles y AVWAP mensuales).

> Parámetros globales: `tick_size` (ES=0.25), `n_bins_vp=10`, ventanas por defecto `n_short=20`, `n_long=60`, `ATR=14`, umbrales `p_prox=6 ticks`, `p_inval=6–8 ticks`, `Y_confirm=16 ticks`.

---

# Anexo B · Catálogo de features y recuento (MVP)

A continuación se listan **grupos de features** con su **cálculo** resumido y un **recuento aproximado** (MVP). Las opcionales no suman al total salvo que se activen.

## B.1 Tiempo y sesión (≈5)

- `sin_time`, `cos_time` (2)
- `asia_session`, `eu_session`, `usa_session` (3)

**Subtotal**: **5**

## B.2 Volume Profile (vela / tramo) (≈15)

- `VP0…VP9` (10) – VP normalizado por vela.
- Momentos del VP: `vp_skew`, `vp_kurt`, `vp_entropy` (3).
- Marcadores: `hvn_flag`, `lvn_flag` (2).

**Subtotal**: **15**

## B.3 MVC × tipo de vela (≈15)

- MVC relativos: `mvc_pos`, `mvc_offset_ticks`, `mvc_to_close_ticks`, `mvc_vol_ratio`, `mvc_delta_ratio` (5).
- Descriptores de vela: `dir`, `body_ratio`, `upper_wick_ratio`, `lower_wick_ratio` (4).
- Interacciones: `mvc_alignment`, `mvc_vs_body_center`, `impulse_flag`, `exhaustion_flag`, `absorption_flag_mvc` (5).

**Subtotal**: **15**

## B.4 Delta profile + Imbalances (diagonales) (≈12)

- Distribución delta: `delta_top_ratio`, `delta_bottom_ratio`, `delta_above_close_ratio` (3).
- Imbalances: `imbalance_count_buy`, `imbalance_count_sell` (2), `imbalance_top_ratio`, `imbalance_bottom_ratio` (2), `imbalance_strength_avg` (1), `imbalance_extreme_flag` (1).
- Señales de absorción: `absorption_buy_flag`, `absorption_sell_flag` (2).

**Subtotal**: **12**

## B.5 Stacked Imbalances (SI) (≈15)

- Conteos/longitudes: `si_buy_count`, `si_sell_count` (2), `si_buy_max_len`, `si_sell_max_len` (2).
- Localización: `si_buy_top`, `si_buy_bottom`, `si_sell_top`, `si_sell_bottom` (4).
- Intensidad: `si_buy_intensity`, `si_sell_intensity` (2).
- Flags/índices: `si_support_flag`, `si_resistance_flag`, `si_absorption_flag` (3), `si_alignment_score`, `absorption_struct_score` (2).

**Subtotal**: **15**

## B.6 Geometría respecto a Zonas “macro” (MVP por metatipo) (≈24)

**Metatipos incluidos (3):** {Diarios: `PDH/PDL`, IB: `IBH/IBL`, VWAP: `VWAP`, `VWAP±σ` (tratado como grupo)}.

Para cada metatipo Z, en la vela del evento:

- `dist_to_Z_ticks`, `signed_dist_to_Z_ticks`, `touches_Z_L60`, `confluence_count_Z`, `zone_age_Z`, `zone_strength_Z`, `approach_speed_ticks_Z`, `approach_angle_Z` → **8** por Z.

**Subtotal**: **3 × 8 = 24**

> Nota: Si se activan **todas** las zonas individuales (`PDH`, `PDL`, `IBH`, `IBL`, `VWAP`, `VWAP+σ`, `VWAP−σ`) por separado, este bloque subiría a ≈ **56** features.

## B.7 Libro de Niveles (activos de velas pasadas) (≈11)

- *Nearest SI (target)*: `nearest_SI_dist_ticks`, `nearest_SI_width`, `nearest_SI_intensity`, `nearest_SI_virgin`, `nearest_SI_tests`, `nearest_SI_age_bars`, `nearest_SI_confluence_count`, `nearest_SI_atr_scaled_distance` (8).
- Densidad local: `levels_above_count_total`, `cum_intensity_above`, `closest_mixed_confluence_dist` (3).

**Subtotal**: **11**

## B.8 Vector de Estado Reciente (n\_short=20, n\_long=60) (≈31)

- Tendencia: `ols_slope_close_ns`, `ols_slope_close_nl` (2), `ols_R2_ns`, `ols_R2_nl` (2), `trend_sign_consistency_ns` (1), `vwap_slope_ns` (1).
- Estructura HH/HL/LH/LL: `count_HH_ns`, `count_HL_ns`, `count_LH_ns`, `count_LL_ns`, `structure_bias_ns` (5).
- Eficiencia/serrucho: `path_efficiency_ns`, `sign_persistence_ns` (2).
- Congestión: `overlap_ratio_ns`, `inside_bar_ratio_ns`, `donchian_pos_ns`, `bb_squeeze_ns` (4).
- Presión de ruptura: `n_new_highs_ns`, `n_new_lows_ns`, `break_pressure` (3).
- Volatilidad: `atr_14`, `atr_trend_ns`, `range_norm_ns` (3), `vol_regime_ns` (3 one-hot).
- Scores: `trend_strength`, `lateral_score`, `breakout_score`, `compression_score` (4).

**Subtotal**: **31**

> Opcionales (+4): `vol_slope_ns`, `delta_slope_ns`, `delta_direction_consistency_ns`, `of_efficiency_ns`.

## B.9 Señales base de volumen/delta normalizados (≈2)

- `vol_pctchg_ma10_minmax`, `delta_zscore_L`.

**Subtotal**: **2**

---

## Total aproximado (MVP)

Suma de subtotales (B.1..B.9) = **5 + 15 + 15 + 12 + 15 + 24 + 11 + 31 + 2 = 130** features.

- **Rango esperado** con opcionales/zonas extra: **120–160**.
- Todos los bloques son **parametrizables**: podemos reducir con `K=1`, agrupar zonas por metatipo o activar/desactivar opcionales según latencia/overfitting.

---

## Nomenclatura y convenciones

- Sufijos: `_ns` = ventana corta (`n_short`), `_nl` = ventana larga (`n_long`), `_Z` = relativo a una zona, `_SI` = relativo a stacked imbalance.
- Distancias en **ticks**; versión escalada: `_atr` = distancia ÷ (ATR/tick\_size).
- Flags booleanos en `{0,1}`; *scores* en ℝ acotada si aplica.
- Reinicios intradía por `NewSession` (normalizaciones, IB, PDH/PDL, VWAP de sesión).

---

> Con este inventario dejamos cerrada la **documentación previa** de *Datos* y *Features*. En la siguiente fase añadiremos el **esquema del dataset de eventos** (columnas obligatorias, tipos y ejemplo) y, más adelante, pasaremos a la implementación del *módulo de zonas*, *libro de niveles* y *labeling* Rebote/Ruptura.



---

# Anexo C · Confluencias, Importancia de Zonas, USA IB, VAH/POC/VAL y Proyección

## C.1 Medición de **confluencias** entre zonas

Definimos **confluencia** cuando dos o más zonas activas tienen **distancias en precio** dentro de un radio `r_conf` (en ticks) respecto al **precio de referencia** (`p_ref`, por defecto el centro del nivel objetivo o la zona “host” que estamos evaluando).

**Componentes:**

1. **Conteo discreto**: `confluence_count = |{Z_j : |p_anchor_j − p_ref| ≤ r_conf}|` (excluye la propia zona host).
2. **Score continuo** (suavizado por distancia y tipo):

   `confluence_score = Σ_j w_type(Z_j) · K(|p_anchor_j − p_ref|)`

   con `K(d) = exp(−(d/σ_conf)^2)` (kernel gaussiano) o `K(d) = max(0, 1 − d/r_conf)` (triangular).
3. **Direccionalidad**: split **por lado**: `confluence_above`, `confluence_below` usando únicamente niveles con `p_anchor_j > p_ref` o `< p_ref`.
4. **Granularidad por metatipo**: `confluence_{PD,IB,VWAP,SI,HVN,VAL/POC/VAH}` (opcional) para saber **de qué familia** viene la confluencia.

**Parámetros**: `r_conf = 8–15 ticks`, `σ_conf = r_conf/2`. Todos configurables por activo.

---

## C.2 **Importancia de zonas** (ponderaciones)

Cada zona `Z` tiene una **fuerza** compuesta que influye en filtros y en el *ranking* de niveles cuando hay varios candidatos:

`zone_strength(Z) = w_type(Z) · w_intensity(Z) · w_status(Z) · w_recency(Z) · w_width(Z) · (1 + λ·confluence_local(Z))`

- **w\_type(Z)**: peso base por metatipo (puede ajustarse tras EDA):
  - `PDH/PDL` = 1.00
  - `USA_IBH/USA_IBL` = 0.95
  - `VAH/POC/VAL (D-1)` = 0.90 (POC suele > VAH/VAL → 0.92/0.90/0.90 si se distingue)
  - `AVWAP sesión/semana` = 0.88
  - `VWAP` actual = 0.85; `VWAP±1σ` = 0.80; `±2σ` = 0.75
  - `Stacked Imbalance (SI)` = 0.85 · **clip(intensity\_norm, 0.8, 1.2)**
  - `HVN/LVN` intradía = 0.80
- **w\_intensity(Z)**: normalizada a [0.8, 1.2] según el metatipo:
  - SI: media de ratios en el stack y longitud (más alto → más intenso)
  - PD/IB/VWAP: *touches* previos con rechazo limpio, MFE medio a favor
  - VAH/POC/VAL: volumen relativo en VA/POC del día anterior
- **w\_status(Z)**: estado del nivel
  - `virgin=True` → 1.05; `tests=1–2` → 1.00; `tests≥3` → 0.95
  - `invalidated=True` → 0.0 (se descarta)
- **w\_recency(Z)**: decaimiento temporal
  - `exp(− age_sessions / τ)` con `τ=3–7` (zonas recientes pesan más)
- **w\_width(Z)**: penaliza niveles excesivamente anchos
  - `w_width = 1 / (1 + width_ticks / w0)` con `w0=12` (o usar *clip* entre 0.85 y 1.05)
- **confluence\_local(Z)**: `confluence_score` medido **en el centro de Z**; `λ` pequeño (p.ej. 0.1)

> Estas ponderaciones son **semilla** para priorizar. El modelo/EDA podrá reajustarlas.

---

## C.3 **Initial Balance de la sesión USA**

El **IB\_USA** es el **rango High/Low de la primera hora** de la **sesión USA**. Para 5m → `N_IB=12` velas.

**Identificación robusta del inicio USA (sin depender de DST):**

1. **Calibración offline** (recomendado):
   - Para cada sesión (día), calcula el índice intradía `i ∈ [0, N−1]` de **máximo incremento conjunto** de `zscore(Volume)` y `zscore(TrueRange)` dentro de una ventana amplia (p.ej., indices 140–220 si `N=276`).
   - Define `usa_open_offset` como la **mediana** de esos índices en un mes de datos.
2. **Detección diaria online**:
   - En cada día, dentro de `usa_open_offset ± δ` (p.ej. ±6 velas), selecciona el `i*` que **maximiza** `S(i) = 0.6·zVol(i) + 0.4·zTR(i)` con condición de no pertenecer a Asia temprana.
   - Marca `usa_open_idx = i*` y construye `IB_USA = [Low_{i*:i*+N_IB−1}, High_{i*:i*+N_IB−1}]`.

**Zonas resultantes**: `USA_IBH`, `USA_IBL` para el día actual; además, `USA_IBH/IBL_{D-1}` para el día anterior si se desea.

---

## C.4 **VAH/POC/VAL del día anterior**

Calculamos el **Volume Profile diario** (sobre la sesión completa o sobre USA-RTH, a elegir) y derivamos:

- **VAH** (Value Area High), **POC** (Point of Control), **VAL** (Value Area Low).
- Guardamos `VAH_{D-1}`, `POC_{D-1}`, `VAL_{D-1}` (y opcionalmente de la sesión USA-RTH anterior: `USA_VAH_{D-1}`, `USA_POC_{D-1}`, `USA_VAL_{D-1}`).

**Features asociadas (en eventos de llegada):**

- `dist_to_VAH_{D-1}_ticks`, `dist_to_POC_{D-1}_ticks`, `dist_to_VAL_{D-1}_ticks` (y versiones `_atr`).
- Confluencias: incluir VAH/POC/VAL en `confluence_score`.
- Ponderación en `w_type`: POC ligeramente > VAH/VAL.

---

## C.5 Proyección del movimiento (extensión/rechazo)

Planteamos dos enfoques **compatibles**:

### C.5.1 Ventana estática

- **Etiquetado** con horizonte `H` (p.ej., 12–24 velas):
  - **MFE/MAE\_H** en ticks (máxima excursión favorable/desfavorable)
  - **Hit\@X**: probabilidad de alcanzar `X` ticks antes de `H` (`X` en {8, 12, 16, 20})
- **Modelos**: clasificación (Hit\@X) + regresión (MFE/MAE\_H o cuantiles Q10/Q50/Q90).
- **Uso**: reglas simples con TP/SL fijos por zona/sesión y EV calculable.

### C.5.2 Objetivo dinámico = **siguiente nivel en dirección**

- Define **siguiente nivel** `L_next` usando el **Libro de Niveles**:
  - Si se predice **ruptura** al alza: `L_next` = nivel activo más cercano **por encima** (prioriza por `zone_strength`).
  - Si se predice **rebote** a la baja: `L_next` = nivel activo más cercano **por debajo**.
- **Etiquetas**:
  - `Hit_next = 1` si se toca `L_next` antes de `H_max` velas (p.ej., 36) sin golpear un **stop** intermedio.
  - `T_next` = tiempo (velas) hasta `L_next` (regresión positiva truncada).
  - `overshoot_next_ticks` = cuántos ticks pasa de `L_next` (para medir breakout limpias).
- **Modelos**:
  - 1ª etapa: `P(evento ∈ {rebote, ruptura})` (clasificación); 2ª etapa: `P(Hit_next | evento)` y `E[T_next | Hit_next]` (clasificación + regresión).
- **Uso**: estrategias con **TP dinámico** (el nivel `L_next`) y **SL basado en la geometría** del nivel actual (ancho/penetración permitida `p_inval`).

> Podemos mantener **ambos** esquemas en paralelo para comparar: el estático ofrece una base sólida y el dinámico captura la microestructura del *mapa de niveles*.



---

# Anexo D · Estructura de carpetas del proyecto (repositorio)

> Diseño pensado para: notebooks separados, funciones reutilizables en `src/`, datasets intermedios/procesados versionables, modelos y experimentos, y configuración clara. Compatible Windows/Linux/Mac.

```text
PricePrediction-Zones/
├─ README.md
├─ LICENSE                      # opcional
├─ .gitignore
├─ pyproject.toml               # paquete editable (pip install -e .)
├─ requirements.txt             # o environment.yml si prefieres conda
├─ .env.example                 # plantilla (variables de conexión Mongo, etc.)
│
├─ src/
│  └─ ppz/                      # paquete del proyecto (Predicción Precio Zonas)
│     ├─ __init__.py
│     ├─ io/
│     │  ├─ mongo.py            # conexión/carga desde MongoDB
│     │  └─ paths.py            # utilidades de rutas (pathlib), carpetas estándar
│     ├─ features/
│     │  ├─ pretreatment.py     # MA→pct→minmax, ATR, VP por vela (← MLfunctions_pretreatment.py)
│     │  ├─ vp.py               # momentos VP, HVN/LVN, interpolaciones
│     │  ├─ mvc.py              # MVC × tipo de vela, posiciones/flags
│     │  ├─ of_imbalance.py     # delta profile, imbalances y stacked imbalances
│     │  ├─ state_vector.py     # vector de estado reciente (tendencia/lateralidad)
│     │  └─ zones.py            # geometría respecto a Z (PD/IB/VWAP/VAH/POC/VAL...)
│     ├─ levels/
│     │  └─ level_registry.py   # libro de niveles: alta, tests, invalidación, fusión
│     ├─ labeling/
│     │  └─ events.py           # reglas Rebote/Ruptura/None, MFE/MAE, Hit@X/Hit_next
│     ├─ models/
│     │  ├─ baselines.py        # LogReg/HGB/XGB, cuantiles, calibración
│     │  ├─ seq.py              # LSTM/Transformer (opcional)
│     │  └─ predict.py          # inferencia/utilidades (← MLfunctions_predictions.py)
│     ├─ utils/
│     │  ├─ time_session.py     # detección robusta USA open e IB_USA
│     │  ├─ metrics.py          # métricas (AUC‑PR, Brier/ECE, Pinball, EV)
│     │  ├─ plotting.py         # funciones de gráfico (← MLfunctions_plot.py)
│     │  └─ config.py           # carga de YAML/TOML, dataclasses de params
│     └─ pipelines/
│        └─ build_dataset.py    # orquestación de pasos para eventos/featurizado
│
├─ notebooks/
│  ├─ 00_data_prep.ipynb        # (← Fase1.ipynb si aplica)
│  ├─ 01_zones_detect.ipynb
│  ├─ 02_event_labeling.ipynb
│  ├─ 03_baselines_cls_reg.ipynb
│  ├─ 04_seq_models.ipynb
│  ├─ 05_backtest_rules.ipynb
│  ├─ 06_dashboard_eval.ipynb
│  └─ scratch/                  # pruebas rápidas, EDA temporal
│
├─ data/
│  ├─ raw/                      # dumps/parquets crudos, nunca tocados
│  ├─ interim/                  # temporales intermedios (cache transformaciones)
│  └─ processed/
│     ├─ features/              # matrices/ventanas de features
│     ├─ levels/                # snapshots del libro de niveles por día
│     └─ events/                # datasets de eventos etiquetados
│
├─ models/
│  ├─ checkpoints/              # guardados durante training
│  └─ artifacts/                # modelos finales, calibraciones
│
├─ configs/
│  ├─ params.yml                # ventanas, umbrales (p, X, Y, H, r_conf, etc.)
│  ├─ zones.yml                 # pesos w_type, tipos activos, σ VWAP, etc.
│  └─ mongo.toml                # credenciales/cluster (usar .env para secretos)
│
├─ experiments/
│  ├─ runs/                     # resultados por experimento (json, métricas)
│  └─ notebooks_output/         # export de figuras/tablas desde notebooks
│
├─ reports/
│  ├─ figures/                  # png/svg/pdf
│  └─ tables/                   # csv/markdown de tablas resumen
│
├─ scripts/
│  ├─ build_features.py         # CLI: generar features
│  ├─ make_events.py            # CLI: construir/actualizar dataset de eventos
│  ├─ train_baseline.py         # CLI: entrenar y calibrar baseline
│  └─ backtest.py               # CLI: backtest reglas/umbral
│
└─ tests/
   ├─ test_pretreatment.py
   ├─ test_of_imbalance.py
   ├─ test_zones.py
   ├─ test_level_registry.py
   └─ test_labeling.py
```

## Archivos raíz recomendados

- `` (extracto útil):
  ```gitignore
  # Python/IDE
  __pycache__/        
  *.pyc
  .ipynb_checkpoints/
  .vscode/
  .idea/

  # Entornos/secretos
  .env
  .venv/

  # Datos y modelos
  data/raw/
  data/interim/
  data/processed/
  models/
  experiments/runs/
  reports/figures/
  ```
- `` (mínimo para editable install):
  ```toml
  [build-system]
  requires = ["setuptools", "wheel"]
  build-backend = "setuptools.build_meta"

  [project]
  name = "ppz"
  version = "0.1.0"
  description = "Predicción de Precio Basada en Zonas"
  requires-python = ">=3.10"
  dependencies = []  # usar requirements.txt

  [tool.setuptools.packages.find]
  where = ["src"]
  ```
- `` (ejemplo inicial):
  ```
  numpy
  pandas
  scipy
  scikit-learn
  xgboost
  matplotlib
  plotly
  pyyaml
  tomli; python_version < "3.11"
  pymongo
  python-dotenv
  ```

## Mapeo de tus archivos actuales (propuesta)

- `Fase1.ipynb` → `notebooks/00_data_prep.ipynb`
- `MLfunctions_pretreatment.py` → `src/ppz/features/pretreatment.py`
- `MLfunctions_predictions.py` → `src/ppz/models/predict.py`
- `MLfunctions_plot.py` → `src/ppz/utils/plotting.py`

> Tras mover, instala en editable para importaciones limpias:
>
> ```bash
> pip install -e .
> ```
>
> y en notebooks: `from ppz.features.pretreatment import ...`.

## Notas de uso

- **Rutas** con `pathlib` desde `ppz.io.paths` (evita problemas Windows/Linux).
- **Secretos** via `.env` + `dotenv` (no subir a git).
- **Configs** centralizadas en `configs/*.yml` y cargadas con `ppz.utils.config`.
- **Datos grandes**: versionar *scripts* y *metadatos*, no binarios. (Si se desea, añadir DVC más adelante.)
- **Numeración de notebooks** como en el plan para mantener el flujo reproducible.



---

# Anexo E · Persistencia de datos entre notebooks (formatos y rutas)

## E.1 Formatos recomendados

- **Parquet (recomendada)**: compacto y rápido. Requiere `pyarrow` (y opcional `zstandard`).
  - Guardar: `df.to_parquet(path, engine="pyarrow", compression="zstd", index=False)`
  - Cargar: `pd.read_parquet(path)`
- **Feather**: muy rápido en local; no compresiona tanto como Parquet.
  - Guardar: `df.to_feather(path)`
  - Cargar: `pd.read_feather(path)`
- **Pickle**: usar **solo** si necesitas persistir columnas con listas/objetos (p.ej. `Ask`, `Bid`) sin fricción.
  - Guardar: `df.to_pickle(path)`
  - Cargar: `pd.read_pickle(path)`

> **Consejo**: calcula features de *order flow* (VAH/POC/VAL, imbalances) y **descarta** `Ask/Bid` antes de persistir datasets grandes.

---

## E.2 Rutas estándar (helpers)

En `src/ppz/io/paths.py`:

```python
from pathlib import Path

def project_root() -> Path:
    return Path(__file__).resolve().parents[3]

def ensure_dir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)

def interim_path(name: str) -> Path:
    p = project_root() / "data" / "interim" / name
    ensure_dir(p); return p

def events_path(name: str) -> Path:
    p = project_root() / "data" / "processed" / "events" / name
    ensure_dir(p); return p
```

**Uso en notebooks**:

```python
from ppz.io.paths import interim_path, events_path

# guardar
df.to_parquet(interim_path("ES_5m_2021_2024.parquet"), engine="pyarrow", compression="zstd", index=False)
events_df.to_parquet(events_path("events_first_pass_2021_2024.parquet"), engine="pyarrow", compression="zstd", index=False)

# cargar
df = pd.read_parquet(interim_path("ES_5m_2021_2024.parquet"))
events_df = pd.read_parquet(events_path("events_first_pass_2021_2024.parquet"))
```

---

## E.3 Flujo entre notebooks

- `` (salidas):
  - Base OHLCV + niveles: `data/interim/ES_5m_YYYY_YYYY.parquet`
  - Eventos detectados (sin label): `data/processed/events/events_first_pass_YYYY_YYYY.parquet`
- `` (entradas):
  - Carga ambos ficheros, genera `events_labeled` y guarda en\
    `data/processed/events/events_labeled_YYYY_YYYY.parquet`
- `` (entradas):
  - `events_labeled_*.parquet` + (si se requieren features extra) `ES_5m_*.parquet`

**Ejemplo **``**:**

```python
# 00_data_prep.ipynb (final)
df.to_parquet("data/interim/ES_5m_2021_2024.parquet", engine="pyarrow", compression="zstd", index=False)
events_df.to_parquet("data/processed/events/events_first_pass_2021_2024.parquet", engine="pyarrow", compression="zstd", index=False)
```

```python
# 02_event_labeling.ipynb (inicio)
df = pd.read_parquet("data/interim/ES_5m_2021_2024.parquet")
events_df = pd.read_parquet("data/processed/events/events_first_pass_2021_2024.parquet")
```

---

## E.4 Buenas prácticas

- **Versiona scripts y metadatos**; no subas binarios de `data/` ni `models/`.
- Ajusta **nombres** con patrón:
  - `ES_5m_YYYY_YYYY.parquet` (base)
  - `events_first_pass_YYYY_YYYY.parquet` (eventos sin label)
  - `events_labeled_YYYY_YYYY.parquet` (eventos etiquetados)
- **Compresión**: `zstd` en Parquet equilibra ratio y velocidad.
- **Tipos**: cuando persistas features, usa `float32` donde sea posible para ahorrar disco/RAM.
- **Chunking**: para rangos largos, divide lectura por semanas/meses y concatena; evita cargar `Ask/Bid` salvo para ventanas acotadas.
- **.env y secretos**: `.env` local, `.env.example` con placeholders. Nunca subas credenciales.

---

## E.5 Check rápido de integridad

```python
# Verificar formas y rangos tras recarga
print(df.shape, df["Time"].min(), df["Time"].max())
print(events_df.shape, events_df["Time"].min(), events_df["Time"].max())

# Índices
assert df["Time"].is_monotonic_increasing
assert events_df["idx"].between(0, len(df)-1).all()
```

