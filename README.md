# 📊 Análisis de Cartera de Renta Fija
# Hecho por: 
### Albert Martin
### Rodolfo Villena Lapaz
### Alejandro García-Caro Nombela




## 📋 Descripción del Proyecto

Este proyecto desarrolla un **análisis exhaustivo de un universo de bonos corporativos** y construye varias carteras de renta fija con diferentes estrategias. El objetivo es proporcionar herramientas profesionales para la valoración, análisis de riesgo y construcción de carteras optimizadas de bonos corporativos.

### 🎯 Objetivos Principales

- Realizar un análisis exploratorio completo del universo de bonos
- Valorar bonos utilizando curvas de descuento y comparar con precios de mercado
- Calcular métricas fundamentales (YTM, Duración, Convexidad, Spreads)
- Construir y optimizar carteras según mandatos específicos
- Implementar estrategias de cobertura de riesgo (tipos de interés y crédito)
- Realizar backtesting de estrategias de inversión

---

## 📁 Estructura del Proyecto

```
Analisis-Cartera-De-Renta-Fija/
│
├── data/                          # Datos del proyecto
│   ├── universo.csv               # Características de todos los bonos
│   ├── precios_historicos_universo.csv  # Precios históricos de bonos
│   ├── curvaESTR.csv             # Curva de tipos de interés €STR
│   └── precios_historicos_varios.csv    # Precios de índices y futuros
│
├── src/                           # Código fuente modular
│   ├── __init__.py
│   ├── analysis.py                # Análisis exploratorio de datos
│   ├── valuation.py               # Valoración de bonos
│   ├── metrics.py                 # Cálculo de métricas (YTM, Duración, etc.)
│   ├── portfolio.py               # Construcción y backtesting de carteras
│   ├── utils.py                   # Utilidades y carga de datos
│   └── TallerRF_AnálisisCartera_Enunciado.ipynb  # Notebook principal
│
├── README.md                       # Este archivo
├── RESUMEN_NOTEBOOK.md            # Resumen detallado del notebook
└── LICENSE                         # Licencia del proyecto
```

---

## 🔧 Requisitos e Instalación

### Dependencias

El proyecto requiere las siguientes librerías de Python:

```python
pandas >= 1.3.0
numpy >= 1.21.0
matplotlib >= 3.4.0
seaborn >= 0.11.0
scipy >= 1.7.0
dateutil >= 2.8.0
```

### Instalación

1. **Clonar o descargar el repositorio**

2. **Instalar dependencias**:
```bash
pip install pandas numpy matplotlib seaborn scipy python-dateutil
```

3. **Ejecutar el notebook**:
   - Abrir `src/TallerRF_AnálisisCartera_Enunciado.ipynb` en Jupyter Notebook o JupyterLab
   - Ejecutar las celdas en orden

---

## 📊 Datos Disponibles

El proyecto utiliza los siguientes archivos de datos:

### 1. `universo.csv`
**Descripción**: Características esenciales de todos los bonos del universo.

**Columnas principales**:
- `ISIN`: Identificador único del bono
- `Issuer`: Emisor del bono
- `Rating`: Calificación crediticia (AAA, AA, A, BBB, BB, etc.)
- `Coupon`: Tasa de cupón anual (%)
- `Coupon Type`: Tipo de cupón (FIXED, VARIABLE)
- `Coupon Frequency`: Frecuencia de pago (1=anual, 2=semestral, etc.)
- `Maturity`: Fecha de vencimiento
- `Next Call Date`: Fecha de call (para bonos callable)
- `Callable`: Indicador de si el bono es callable (Y/N)
- `Seniority`: Prelación (Sr Unsecured, Subordinated, etc.)
- `Industry Sector`: Sector industrial
- `Ccy`: Moneda
- `Price`: Precio de mercado (MID)
- `Bid Price` / `Ask Price`: Precios bid/ask
- `Outstanding Amount`: Nominal vivo
- `PD 1YR`: Probabilidad de default a 1 año

**Estadísticas**:
- ~2,255 bonos vivos en la fecha de análisis
- Mayoritariamente EUR (100%)
- 82% Investment Grade, 0.4% High Yield, 17.3% No Rated
- Concentración en sector Financial (45%)

### 2. `precios_historicos_universo.csv`
**Descripción**: Precios históricos de cierre de todos los bonos del universo.

**Estructura**:
- Filas: ISINs de los bonos
- Columnas: Fechas (formato DD/MM/YYYY)
- Valores: Precios de cierre

**Uso**: Backtesting de estrategias y análisis de evolución de precios.

### 3. `curvaESTR.csv`
**Descripción**: Curva de tipos de interés €STR (Euro Short-Term Rate).

**Columnas**:
- `Date`: Fecha del punto de la curva
- `Zero Rate`: Tipo de interés cero cupón (puede estar en % o decimal)
- `Discount`: Factor de descuento (calculado si no está presente)

**Uso**: Descuento de flujos de caja para valoración de bonos.

**Nota**: La interpolación se realiza de forma exponencial (log-lineal en discount factors).

### 4. `precios_historicos_varios.csv`
**Descripción**: Precios históricos de instrumentos para cobertura y benchmarking.

**Instrumentos incluidos**:
- **ITRAXX Main**: Índice CDS de Investment Grade europeo (5 años)
- **ITRAXX XOVER**: Índice CDS de High Yield europeo (5 años)
- **DU1 (Schatz)**: Futuro sobre bono alemán 2 años
- **OE1 (Bobl)**: Futuro sobre bono alemán 5 años
- **RX1 (Bund)**: Futuro sobre bono alemán 10 años
- **RECMTREU Index**: Benchmark de crédito corporativo europeo (Total Return)

**Uso**: 
- Cobertura de riesgo de tipos de interés (futuros)
- Cobertura de riesgo de crédito (CDS)
- Comparación de rendimiento (benchmark)

---

## 📚 Módulos del Proyecto

### 1. `utils.py` - Utilidades y Carga de Datos

**Funciones principales**:

- **`get_data_path(current_dir)`**: Obtiene la ruta a la carpeta de datos
- **`load_universe(data_path, fecha_analisis)`**: Carga y prepara el universo de bonos
  - Filtra bonos vivos (maturity > fecha análisis)
  - Calcula años hasta maturity
  - Calcula bid-ask spread
- **`load_and_prepare_curve(data_path, fecha_analisis)`**: Carga y prepara la curva €STR
  - Calcula tenors (años desde fecha análisis)
  - Normaliza tipos de interés (convierte % a decimal si es necesario)
  - Calcula discount factors si no están presentes
- **`load_historical_prices_universe(data_path)`**: Carga precios históricos del universo
- **`load_historical_prices_various(data_path)`**: Carga precios de índices y futuros
- **`get_effective_maturity(row)`**: Calcula vencimiento efectivo (considera callable)
- **`calculate_years_to_maturity(row, fecha_analisis)`**: Calcula años hasta maturity

**Ejemplo de uso**:
```python
from utils import load_universe, load_and_prepare_curve

fecha_analisis = datetime(2025, 10, 1)
data_path = get_data_path()

vivos = load_universe(data_path, fecha_analisis)
curva_work = load_and_prepare_curve(data_path, fecha_analisis)
```

---

### 2. `analysis.py` - Análisis Exploratorio

**Funciones principales**:

- **`analyze_currencies(df, print_results=True)`**: Analiza distribución por divisas
  - Lista divisas únicas
  - Calcula distribución porcentual
  - Retorna diccionario con estadísticas

- **`analyze_bond_types(df, print_results=True)`**: Analiza tipos de bonos
  - Distribución por tipo de cupón (FIXED, VARIABLE)
  - Análisis de opcionalidad (Callable)
  - Distribución por prelación (Seniority)
  - Identifica bonos perpetuos

- **`analyze_ratings(df, print_results=True)`**: Analiza ratings y riesgo de crédito
  - Distribución por rating (ordenada según escala Fitch)
  - Clasificación Investment Grade vs High Yield
  - Cálculo de PD 1YR promedio
  - Análisis por categoría (IG, HY, NR)

- **`analyze_sectors(df, print_results=True)`**: Analiza sectores y emisores
  - Distribución por sector industrial
  - Top emisores por número de emisiones
  - Índice de Herfindahl (concentración sectorial)
  - Análisis de concentración por emisor

- **`analyze_liquidity(df, print_results=True)`**: Analiza riesgo de liquidez
  - Estadísticas de bid-ask spread
  - Clasificación de liquidez (alta/media/baja)
  - Análisis de nominal vivo (Outstanding Amount)
  - Correlación entre spread y tamaño de emisión

**Ejemplo de uso**:
```python
from analysis import analyze_currencies, analyze_ratings, analyze_liquidity

# Análisis de divisas
divisas_results = analyze_currencies(vivos)

# Análisis de ratings
ratings_results = analyze_ratings(vivos)

# Análisis de liquidez
liquidity_results = analyze_liquidity(vivos)
```

**Constantes definidas**:
- `IG_RATINGS`: Lista de ratings Investment Grade
- `HY_RATINGS`: Lista de ratings High Yield
- `FITCH_RATING_ORDER`: Orden de ratings de mejor a peor

---

### 3. `valuation.py` - Valoración de Bonos

**Funciones principales**:

- **`get_discount_from_curve(curva_work, t)`**: Interpola factor de descuento
  - **Método**: Interpolación exponencial (log-lineal en discount factors)
  - **Parámetros**: 
    - `curva_work`: DataFrame con columnas 'Tenor' y 'Discount'
    - `t`: Tiempo en años
  - **Retorna**: Factor de descuento interpolado

- **`generate_coupon_dates(row, fecha_analisis, effective_maturity=None)`**: Genera fechas de cupón
  - Considera frecuencia de pago
  - Usa vencimiento efectivo (considera callable)
  - Retorna lista de fechas futuras ordenadas

- **`calculate_accrued_interest(coupon_per_period, pay_dates, fecha_analisis, freq)`**: Calcula cupón corrido
  - **Base**: ACT/365
  - Calcula días desde último cupón hasta fecha análisis

- **`valorar_bono(row, fecha_analisis, curva_work, spread_bps=0.0, nominal=100.0)`**: Valoración completa
  - **Inputs**:
    - `row`: Fila del DataFrame con información del bono
    - `fecha_analisis`: Fecha de valoración
    - `curva_work`: Curva de descuento
    - `spread_bps`: Spread de crédito en puntos básicos (opcional)
    - `nominal`: Valor nominal (default 100)
  - **Outputs**: Tupla `(precio_limpio, cupón_corrido, precio_sucio)`
  - **Método**:
    1. Genera flujos de caja futuros (cupones + principal)
    2. Descuenta cada flujo usando curva €STR + spread
    3. Calcula precio sucio (suma de valores presentes)
    4. Calcula cupón corrido
    5. Calcula precio limpio (sucio - corrido)

- **`spread_implicito(row, fecha_analisis, curva_work, nominal=100.0, tol=1e-6)`**: Calcula spread implícito
  - **Objetivo**: Encuentra el spread que hace que precio teórico = precio de mercado
  - **Método**: Resolución numérica (fsolve)
  - **Retorna**: Spread en puntos básicos (o NaN si no converge)

**Ejemplo de uso**:
```python
from valuation import valorar_bono, spread_implicito

# Valorar un bono sin spread
precio_limpio, cupon_corrido, precio_sucio = valorar_bono(
    row=vivos.iloc[0],
    fecha_analisis=fecha_analisis,
    curva_work=curva_work
)

# Calcular spread implícito
spread = spread_implicito(
    row=vivos.iloc[0],
    fecha_analisis=fecha_analisis,
    curva_work=curva_work
)
```

**Notas importantes**:
- La interpolación es **exponencial** (no lineal)
- Se usa base **ACT/365** para cupón corrido
- Para bonos callable, se usa `Next Call Date` como vencimiento efectivo
- Para bonos perpetuos, se usa `Next Call Date` si está disponible

---

### 4. `metrics.py` - Métricas de Bonos

**Funciones principales**:

- **`calculate_cash_flows(coupon, maturity_date, fecha_analisis, face=100.0, frequency=1, effective_maturity=None)`**: Calcula flujos de caja
  - Genera lista de flujos y tiempos (en años)
  - Considera frecuencia de pago
  - Último flujo incluye principal

- **`pv_cash_flows(ytm, cash_flows, times)`**: Calcula valor presente de flujos
  - Descuenta cada flujo usando YTM
  - Fórmula: `PV = Σ [CF_i / (1 + YTM)^t_i]`

- **`calculate_ytm(price, coupon, maturity_date, fecha_analisis, face=100.0, frequency=1, effective_maturity=None, initial_guess=0.05)`**: Calcula Yield to Maturity
  - **Definición**: Tasa de retorno interna que iguala PV(flujos) = precio
  - **Método**: Resolución numérica (fsolve)
  - **Retorna**: YTM anual (decimal, ej. 0.05 = 5%)

- **`calculate_modified_duration(price, ytm, coupon, maturity_date, fecha_analisis, face=100.0, frequency=1, effective_maturity=None)`**: Calcula Duración Modificada
  - **Definición**: Sensibilidad porcentual del precio ante cambio de 1% en YTM
  - **Fórmula**: `Modified Duration = Macaulay Duration / (1 + YTM)`
  - **Interpretación**: Si duración = 5 años, un aumento de 1% en YTM → precio cae ~5%
  - **Retorna**: Duración en años

- **`calculate_convexity(price, ytm, coupon, maturity_date, fecha_analisis, face=100.0, frequency=1, effective_maturity=None)`**: Calcula Convexidad
  - **Definición**: Mide la curvatura de la relación precio-YTM
  - **Fórmula**: `Convexity = (1/P) * d²P/dYTM²`
  - **Uso**: Corrige aproximación lineal de la duración
  - **Retorna**: Convexidad (adimensional)

- **`estimate_price_change(price, duration, convexity, yield_change_bps)`**: Estima cambio de precio
  - **Fórmula**: `ΔP/P ≈ -Duration * Δy + 0.5 * Convexity * (Δy)²`
  - **Parámetros**:
    - `yield_change_bps`: Cambio en yield en puntos básicos
  - **Retorna**: Tupla `(cambio_porcentual, nuevo_precio)`

**Ejemplo de uso**:
```python
from metrics import calculate_ytm, calculate_modified_duration, calculate_convexity

# Calcular YTM
ytm = calculate_ytm(
    price=100.5,
    coupon=3.5,
    maturity_date=datetime(2028, 10, 1),
    fecha_analisis=datetime(2025, 10, 1)
)

# Calcular duración
duration = calculate_modified_duration(
    price=100.5,
    ytm=ytm,
    coupon=3.5,
    maturity_date=datetime(2028, 10, 1),
    fecha_analisis=datetime(2025, 10, 1)
)

# Calcular convexidad
convexity = calculate_convexity(
    price=100.5,
    ytm=ytm,
    coupon=3.5,
    maturity_date=datetime(2028, 10, 1),
    fecha_analisis=datetime(2025, 10, 1)
)

# Estimar cambio de precio si YTM sube 50 bps
change_pct, new_price = estimate_price_change(
    price=100.5,
    duration=duration,
    convexity=convexity,
    yield_change_bps=50
)
```

**Notas importantes**:
- YTM asume reinversión de cupones al mismo YTM
- Duración modificada es más útil que Macaulay para gestión de riesgo
- Convexidad siempre es positiva para bonos estándar (beneficiosa)
- La fórmula de cambio de precio es una aproximación (Taylor de segundo orden)

---

### 5. `portfolio.py` - Construcción y Backtesting de Carteras

**Funciones principales**:

- **`get_alive_bonds_at_date(precios_df, universo_df, fecha)`**: Obtiene bonos vivos en una fecha
  - Filtra bonos con precio válido
  - Filtra bonos con maturity > fecha
  - Retorna índice con ISINs

- **`calculate_total_return(precio_inicial, precio_final, coupon_rate, days_held, frequency=1)`**: Calcula retorno total
  - **Fórmula**: `Retorno = (P_final - P_inicial) / P_inicial + (Cupón * días/365.25)`
  - Incluye retorno de precio y cupones recibidos
  - **Retorna**: Retorno en porcentaje

- **`backtest_equally_weighted_portfolio(precios_df, universo_df, precios_varios_df, fecha_inicio, fecha_fin=None, rebalance_frequency='M', benchmark_col='RECMTREU Index')`**: Backtest de cartera equiponderada
  - **Estrategia**: Invierte igual cantidad en cada bono disponible
  - **Rebalanceo**: Mensual (o frecuencia especificada)
  - **Cálculo de retornos**: Entre rebalanceos, incluyendo cupones
  - **Comparación**: Con benchmark (RECMTREU Index)
  - **Retorna**: Diccionario con:
    - `portfolio_value`: Series con valor de cartera
    - `benchmark_value`: Series con valor de benchmark
    - `portfolio_returns`: Series con retornos de cartera
    - `benchmark_returns`: Series con retornos de benchmark
    - `rebalance_dates`: Lista de fechas de rebalanceo
    - `positions`: DataFrame con posiciones en cada fecha

- **`calculate_performance_metrics(portfolio_returns, benchmark_returns)`**: Calcula métricas de rendimiento
  - **Métricas calculadas**:
    - Retorno total y anualizado
    - Volatilidad anualizada
    - Sharpe Ratio
    - Tracking Error
    - Information Ratio
    - Alpha y Beta
    - Maximum Drawdown
  - **Retorna**: Diccionario con todas las métricas

- **`build_optimized_portfolio(universo_df, fecha_analisis, max_bonds=20, max_duration=3.0, max_hy_exposure=0.10, min_outstanding=500000000, max_weight_per_bond=0.10, max_weight_per_issuer=0.15, exclude_subordinated=True)`**: Construye cartera optimizada
  - **Objetivo**: Maximizar YTM ponderado de la cartera
  - **Restricciones**:
    - Duración ≤ max_duration años
    - Exposición HY ≤ max_hy_exposure
    - Sin deuda subordinada (si exclude_subordinated=True)
    - Tamaño mínimo: Outstanding Amount ≥ min_outstanding
    - Peso máximo por bono: ≤ max_weight_per_bond
    - Peso máximo por emisor: ≤ max_weight_per_issuer
    - Máximo max_bonds bonos
  - **Método**: Optimización no lineal (SLSQP de scipy)
  - **Retorna**: Diccionario con:
    - `weights`: Series con pesos optimizados
    - `portfolio_df`: DataFrame con bonos seleccionados
    - `metrics`: Diccionario con métricas de la cartera
    - `optimization_result`: Resultado de la optimización

**Ejemplo de uso**:
```python
from portfolio import build_optimized_portfolio, backtest_equally_weighted_portfolio

# Construir cartera optimizada
resultado = build_optimized_portfolio(
    universo_df=vivos,
    fecha_analisis=datetime(2025, 10, 1),
    max_bonds=20,
    max_duration=3.0,
    max_hy_exposure=0.10
)

# Obtener pesos y bonos seleccionados
pesos = resultado['weights']
cartera_df = resultado['portfolio_df']
metricas = resultado['metrics']

# Backtest de cartera equiponderada
backtest_result = backtest_equally_weighted_portfolio(
    precios_df=precios_historicos_universo,
    universo_df=vivos,
    precios_varios_df=precios_varios,
    fecha_inicio=datetime(2023, 10, 1),
    fecha_fin=datetime(2025, 9, 1),
    rebalance_frequency='M'
)

# Calcular métricas de rendimiento
performance = calculate_performance_metrics(
    portfolio_returns=backtest_result['portfolio_returns'],
    benchmark_returns=backtest_result['benchmark_returns']
)
```

**Notas importantes**:
- La optimización puede no converger en algunos casos (se maneja con warnings)
- El backtest asume reinversión de cupones
- No se incluyen costos de transacción explícitamente
- El rebalanceo mensual puede ajustarse según necesidades

---

## 📖 Ejercicios del Notebook

El notebook principal (`TallerRF_AnálisisCartera_Enunciado.ipynb`) desarrolla los siguientes ejercicios:

### **Ejercicio 1: Análisis de Datos**

**Objetivo**: Realizar un análisis exploratorio completo del universo de bonos.

**Contenido**:
1. **Carga y preparación de datos**
   - Carga del universo de bonos
   - Filtrado de bonos vivos
   - Limpieza y normalización

2. **Análisis descriptivo**:
   - **Divisas**: Distribución por moneda (EUR, USD, etc.)
   - **Tipos de bono**: Fijos vs variables, callable, prelación, perpetuos
   - **Ratings**: Distribución por rating, IG vs HY, PD 1YR
   - **Sectores**: Distribución sectorial, índice de Herfindahl
   - **Emisores**: Top emisores, concentración
   - **Liquidez**: Bid-ask spreads, nominal vivo

3. **Visualizaciones**:
   - Gráficos de barras, pie charts, histogramas, boxplots
   - Análisis de correlaciones

**Resultado**: Entendimiento completo del universo que permite filtrar bonos elegibles e identificar riesgos.

---

### **Ejercicio 2: Valoración de Bonos**

**Objetivo**: Valorar los bonos del universo utilizando la curva de descuento €STR y comparar con precios de mercado.

**Proceso**:
1. **Carga de curva €STR**
   - Carga desde `curvaESTR.csv`
   - Preparación para interpolación

2. **Interpolación exponencial**
   - Método: Log-lineal en discount factors
   - Permite obtener tipo para cualquier plazo

3. **Cálculo de precios teóricos**:
   - Generación de flujos de caja (cupones + principal)
   - Descuento usando curva €STR
   - Cálculo de precio sucio y limpio
   - Cálculo de cupón corrido (base ACT/365)

4. **Comparación con mercado**:
   - Diferencias absolutas y porcentuales
   - Identificación de bonos sobre/infravalorados

**Resultado**: Precios teóricos para todos los bonos y base para calcular spreads.

---

### **Ejercicio 3: Cálculo de Spread Implícito**

**Objetivo**: Calcular el spread de crédito que el mercado exige sobre la curva libre de riesgo.

**Proceso**:
1. **Concepto**: El spread implícito compensa riesgo de crédito, liquidez y otros factores.

2. **Cálculo numérico**:
   - Resolver: `Precio_Mercado = Σ [Flujo_i / (1 + r(t_i) + s)^t_i]`
   - Donde `s` es el spread implícito
   - Método: Resolución iterativa (fsolve)

3. **Análisis**:
   - Distribución de spreads
   - Comparación por rating y sector
   - Correlación con otras variables

**Resultado**: Spread implícito para cada bono (en bps) y entendimiento del riesgo percibido.

---

### **Ejercicio 4: Métricas de Bonos (YTM, Duración, Convexidad)**

**Objetivo**: Calcular métricas fundamentales de riesgo y rentabilidad.

**Métricas calculadas**:

1. **Yield to Maturity (YTM)**:
   - Tasa de retorno interna que iguala PV(flujos) = precio
   - Mide rentabilidad esperada si se mantiene hasta vencimiento

2. **Duración Modificada**:
   - Sensibilidad del precio ante cambio de 1% en YTM
   - Fórmula: `Modified Duration = Macaulay Duration / (1 + YTM)`
   - Útil para gestión de riesgo de tipos

3. **Convexidad**:
   - Mide curvatura de relación precio-YTM
   - Corrige aproximación lineal de duración
   - Fórmula de cambio: `ΔP/P ≈ -Duration * Δy + 0.5 * Convexity * (Δy)²`

**Resultado**: Métricas completas para todos los bonos, base para construcción de carteras.

---

### **Ejercicio 5: Cartera Equiponderada y Backtest**

**Objetivo**: Construir una cartera simple (equiponderada) y realizar backtest comparándola con benchmark.

**Proceso**:
1. **Construcción**: Invertir igual cantidad en cada bono del universo
2. **Backtesting**:
   - Rebalanceo mensual
   - Cálculo de retornos (precio + cupones)
   - Tracking de valor de cartera
3. **Comparación con benchmark** (RECMTREU Index):
   - Retorno total y anualizado
   - Volatilidad
   - Sharpe Ratio
   - Maximum Drawdown
   - Tracking Error, Alpha, Beta

**Resultado**: Entendimiento del rendimiento de estrategia pasiva y base de comparación.

---

### **Ejercicio 6: Cartera Optimizada**

**Objetivo**: Construir una cartera optimizada de máximo 20 bonos que maximice rentabilidad sujeto a restricciones.

**Mandato (Restricciones)**:
- Duración máxima: ≤ 3 años
- Exposición HY máxima: ≤ 10%
- Sin deuda subordinada
- Tamaño mínimo: Outstanding Amount ≥ 500M
- Peso máximo por bono: ≤ 10%
- Concentración máxima por emisor: ≤ 15%
- Número máximo de bonos: 20

**Proceso**:
1. **Filtrado inicial**: Bonos que cumplen restricciones básicas
2. **Optimización**:
   - Objetivo: Maximizar YTM ponderado
   - Método: Optimización no lineal (SLSQP)
   - Variables: Pesos de cada bono
3. **Análisis de riesgos**:
   - Riesgo de crédito: PD ponderada, concentración
   - Riesgo de liquidez: Spread promedio, volumen
4. **Backtest** (6.5): Metodología descrita para backtesting con rebalanceo periódico

**Resultado**: Cartera optimizada que cumple todas las restricciones y análisis completo de riesgos.

---

### **Ejercicio 7: Cobertura de Tipos de Interés**

**Objetivo**: Cubrir la exposición de la cartera a movimientos de tipos de interés usando futuros.

**Proceso**:
1. **Cálculo de DV01** (Dollar Value of 01):
   - `DV01_cartera = Inversión × Duración × 0.0001`
   - Mide cambio de valor si tipos suben 1bp

2. **Selección de instrumento**:
   - **DU1 (Schatz)**: 2 años, Duración = 1.92 años
   - **OE1 (Bobl)**: 5 años, Duración = 5.44 años
   - **RX1 (Bund)**: 10 años, Duración = 10.00 años
   - **Criterio**: Priorizar futuro con duración ≥ duración de cartera

3. **Cálculo de contratos**:
   - `DV01_futuro = Duración × Tamaño_contrato × 0.0001`
   - `Número_contratos = DV01_cartera / DV01_futuro`

4. **Posición**: VENDER futuros (posición corta) para compensar pérdidas si suben tipos

5. **Análisis de escenarios**: Cobertura total vs parcial (40% recomendada)

**Código ejecutable**: ⭐ Incluye cálculo dinámico de DV01, selección automática de instrumento y análisis de escenarios.

**Resultado**: Número exacto de contratos necesarios y análisis completo de estrategias.

---

### **Ejercicio 8: Cobertura de Riesgo de Crédito**

**Objetivo**: Cubrir total o parcialmente el riesgo de crédito usando índices CDS.

**Proceso**:
1. **Selección de índice CDS**:
   - **ITRAXX Main**: Investment Grade europeo (5 años)
   - **ITRAXX XOVER**: High Yield europeo (5 años)
   - **Criterio**: Según exposición HY de la cartera

2. **Cálculo de sensibilidad**:
   - Sensibilidad CDS: 4,500 €/bp por 10M€ notional

3. **Cobertura total vs parcial**:
   - **100%**: Elimina exposición pero elimina prima de crédito
   - **50% (recomendada)**: Protege ante estrés, mantiene beta al mercado

4. **Posición**: COMPRAR protección (long CDS) para compensar pérdidas si spreads aumentan

5. **Análisis de niveles**: 0%, 30%, 50%, 70%, 100%

**Código ejecutable**: ⭐ Incluye selección automática de índice, cálculo dinámico de notional y comparación de niveles.

**Resultado**: Notional óptimo calculado automáticamente y análisis completo de estrategias.

---

### **Ejercicio 9: Estrategia Propia**

**Objetivo**: Diseñar una estrategia propia combinando coberturas parciales.

**Estrategia propuesta**: Coberturas Parciales Combinadas

**Componentes**:
1. **Cobertura parcial de tipos (40%)**:
   - Instrumento: Futuros Bobl (OE1)
   - Efecto: Reduce sensibilidad ante subidas bruscas, mantiene exposición moderada

2. **Cobertura parcial de crédito (50%)**:
   - Instrumento: ITRAXX Main CDS
   - Efecto: Protege ante estrés severo, mantiene beta al mercado IG

3. **Gestión de riesgo idiosincrático**:
   - Monitorización de emisores vulnerables
   - Ajustes dinámicos ante señales de deterioro

4. **Rebalanceo dinámico**:
   - Frecuencia: Trimestral
   - Ajustes según condiciones de mercado

**Ventajas**:
- Controla volatilidad sin eliminar rentabilidad
- Protege ante eventos adversos
- Mantiene exposición a movimientos moderados
- Reduce costes vs cobertura total

**Código ejecutable**: ⭐ Incluye cálculo automático de coberturas combinadas, análisis de riesgo residual y resumen ejecutivo.

**Resultado**: Estrategia completa con cálculos dinámicos basados en la cartera optimizada real.

---

## 🚀 Características Avanzadas

### Código Ejecutable Dinámico

El proyecto incluye código ejecutable para los ejercicios 6.5, 7, 8 y 9 que calcula todo dinámicamente basándose en la cartera optimizada real:

- ✅ **Backtest de cartera optimizada** (Ejercicio 6.5)
- ✅ **Cálculo dinámico de cobertura de tipos** (Ejercicio 7)
- ✅ **Cálculo dinámico de cobertura de crédito** (Ejercicio 8)
- ✅ **Estrategia combinada dinámica** (Ejercicio 9)

Todos los cálculos se adaptan automáticamente a la cartera optimizada construida en el Ejercicio 6.

---

## 📝 Notas Importantes

### Consideraciones Prácticas

- **Costos de transacción**: No incluidos explícitamente pero mencionados
- **Liquidez**: Considerada en análisis pero no siempre en ejecución
- **Rebalanceo**: Frecuencia debe ajustarse según costos y mandato
- **Datos históricos**: Limitados a fechas disponibles

### Limitaciones

- **Asunciones simplificadoras**: 
  - Curva plana para YTM
  - Reinversión al mismo YTM
  - No considera costos de transacción detallados, slippage, market impact
- **Optimización**: Puede no converger en algunos casos (se maneja con warnings)

### Mejoras Futuras Sugeridas

- Implementar optimización con PuLP para mayor claridad
- Añadir stress testing (+100bps en ESTR, widening spreads)
- Optimización multi-objetivo (max YTM + min VaR)
- Integración con APIs para datos en tiempo real
- Visualizaciones interactivas (gauge charts, dashboards)

---

## 👥 Autores

- **Albert Martin**
- **Rodolfo Villena**
- **Alejandro García-Caro Nombela**

---

## 📄 Licencia

Ver archivo `LICENSE` para más detalles.

---

## 📚 Referencias y Recursos

### Conceptos Teóricos

- **Valoración de bonos**: Descuento de flujos de caja, interpolación de curvas
- **Métricas de bonos**: YTM, Duración, Convexidad
- **Optimización de carteras**: Optimización con restricciones, Sharpe Ratio
- **Cobertura de riesgo**: DV01, cobertura con futuros y CDS

### Fuentes de Información

- Curvas de tipos de interés: €STR (European Short-Term Rate)
- Índices de crédito: ITRAXX Main, ITRAXX XOVER
- Futuros: Schatz (DU1), Bobl (OE1), Bund (RX1)
- Benchmark: RECMTREU Index

---

## 🔍 Guía de Uso Rápido

### 1. Configuración Inicial

```python
import pandas as pd
import numpy as np
from datetime import datetime
from utils import get_data_path, load_universe, load_and_prepare_curve

# Configurar fecha de análisis
fecha_analisis = datetime(2025, 10, 1)

# Cargar datos
data_path = get_data_path()
vivos = load_universe(data_path, fecha_analisis)
curva_work = load_and_prepare_curve(data_path, fecha_analisis)
```

### 2. Análisis Exploratorio

```python
from analysis import analyze_currencies, analyze_ratings, analyze_liquidity

# Análisis completo
analyze_currencies(vivos)
analyze_ratings(vivos)
analyze_liquidity(vivos)
```

### 3. Valoración y Métricas

```python
from valuation import valorar_bono, spread_implicito
from metrics import calculate_ytm, calculate_modified_duration

# Valorar un bono
precio_limpio, _, _ = valorar_bono(vivos.iloc[0], fecha_analisis, curva_work)

# Calcular métricas
ytm = calculate_ytm(price=100.5, coupon=3.5, ...)
duration = calculate_modified_duration(price=100.5, ytm=ytm, ...)
```

### 4. Construcción de Cartera

```python
from portfolio import build_optimized_portfolio

# Construir cartera optimizada
resultado = build_optimized_portfolio(
    universo_df=vivos,
    fecha_analisis=fecha_analisis,
    max_bonds=20,
    max_duration=3.0
)

pesos = resultado['weights']
cartera_df = resultado['portfolio_df']
```

### 5. Backtesting

```python
from portfolio import backtest_equally_weighted_portfolio

# Backtest de cartera equiponderada
backtest_result = backtest_equally_weighted_portfolio(
    precios_df=precios_historicos_universo,
    universo_df=vivos,
    precios_varios_df=precios_varios,
    fecha_inicio=datetime(2023, 10, 1)
)
```

---

## ❓ Preguntas Frecuentes

### ¿Cómo cambio la fecha de análisis?

Modifica la variable `fecha_analisis` en la celda de configuración inicial:
```python
fecha_analisis = datetime(2025, 10, 1)  # Cambiar aquí
```

### ¿Qué hago si la optimización no converge?

- Verifica que hay suficientes bonos elegibles
- Ajusta las restricciones (p. ej., aumenta max_duration)
- Revisa que los datos estén correctos (precios, métricas)

### ¿Cómo interpreto el spread implícito?

- Spread alto (>200 bps): Mercado percibe alto riesgo
- Spread bajo (<50 bps): Mercado percibe bajo riesgo
- Compara con spreads históricos y del sector

### ¿Puedo usar otros benchmarks?

Sí, modifica el parámetro `benchmark_col` en las funciones de backtest:
```python
backtest_result = backtest_equally_weighted_portfolio(
    ...,
    benchmark_col='TU_INDICE_AQUI'
)
```

---

## 📞 Soporte

Para preguntas o problemas, consulta:
1. El notebook principal (`TallerRF_AnálisisCartera_Enunciado.ipynb`)
2. El resumen detallado (`RESUMEN_NOTEBOOK.md`)
3. Los comentarios en el código de los módulos

---

**Última actualización**: Noviembre 2025
