# RESUMEN COMPLETO DEL NOTEBOOK: ANÁLISIS DE CARTERA DE RENTA FIJA

## 📋 ÍNDICE
1. [Introducción y Objetivos](#introducción)
2. [Punto 1: Análisis de Datos](#punto-1)
3. [Punto 2: Valoración de Bonos](#punto-2)
4. [Punto 3: Cálculo de Spread Implícito](#punto-3)
5. [Punto 4: Métricas de Bonos (YTM, Duración, Convexidad)](#punto-4)
6. [Punto 5: Cartera Equiponderada y Backtest](#punto-5)
7. [Punto 6: Cartera Optimizada](#punto-6)
8. [Punto 7: Cobertura de Tipos de Interés](#punto-7)
9. [Punto 8: Cobertura de Riesgo de Crédito](#punto-8)
10. [Punto 9: Estrategia Propia](#punto-9)
11. [Estructura Técnica del Notebook](#estructura-técnica-del-notebook)
12. [Notas Importantes](#notas-importantes)
13. [Conclusión](#conclusión)
14. [⭐ Mejoras y Funcionalidades Ejecutables](#mejoras-y-funcionalidades-ejecutables-)

---

## 🎯 INTRODUCCIÓN Y OBJETIVOS {#introducción}

### Propósito del Notebook
Este notebook desarrolla un **análisis exhaustivo de un universo de bonos corporativos** y construye varias carteras de renta fija con diferentes estrategias. El objetivo es:

- Analizar características de bonos corporativos
- Valorar bonos usando curvas de descuento
- Construir y optimizar carteras según mandatos específicos
- Implementar estrategias de cobertura de riesgo
- Realizar backtesting de estrategias

### Datos Disponibles
- **universo.csv**: Características esenciales de los bonos (ISIN, emisor, rating, cupón, vencimiento, etc.)
- **precios_historicos_universo.csv**: Precios históricos de cierre de los bonos
- **curvaESTR.csv**: Curva de tipos de interés €STR (curva libre de riesgo)
- **precios_historicos_varios.csv**: Precios de:
  - Índices de crédito (ITRAXX Main, ITRAXX XOVER)
  - Futuros (Schatz DU1, Bobl OE1, Bund RX1)
  - Benchmark (RECMTREU Index)

---

## 📊 PUNTO 1: ANÁLISIS DE DATOS {#punto-1}

### Objetivo
Realizar un análisis exploratorio completo del universo de bonos para entender sus características, identificar gaps en los datos y preparar la información para los siguientes puntos.

### ¿Qué se hace?

#### 1.1. Carga y Preparación de Datos
- **Carga del universo**: Se lee el archivo `universo.csv` con información de todos los bonos
- **Limpieza de datos**: Se identifican y tratan valores faltantes (NaN)
- **Normalización**: Se estandarizan formatos de fechas, precios, ratings, etc.

#### 1.2. Análisis Descriptivo
El notebook analiza múltiples dimensiones del universo:

**a) Análisis de Monedas**
- Distribución de bonos por moneda (EUR, USD, etc.)
- Identifica la moneda dominante

**b) Análisis de Tipos de Bono**
- Distribución entre bonos fijos, variables, etc.
- Identifica características de cupones

**c) Análisis de Ratings**
- Distribución de ratings crediticios (AAA, AA, A, BBB, BB, etc.)
- Clasificación entre Investment Grade (IG) y High Yield (HY)
- Cálculo de probabilidades de default (PD 1YR)

**d) Análisis Sectorial**
- Distribución de bonos por sector (Financial, Industrial, Utilities, etc.)
- Identifica concentración sectorial
- Calcula índice de Herfindahl para medir diversificación

**e) Análisis de Emisores**
- Top emisores por número de emisiones
- Concentración por emisor
- Identifica diversificación del universo

**f) Análisis de Liquidez**
- Calcula bid-ask spreads
- Identifica bonos más/menos líquidos por sector
- Analiza relación entre liquidez y otras características

#### 1.3. Visualizaciones
- Gráficos de barras para distribuciones
- Gráficos de pie para proporciones
- Histogramas para spreads y precios
- Boxplots para comparaciones

#### 1.4. Identificación de Gaps
- Lista valores faltantes por columna
- Propone estrategias de imputación
- Identifica bonos con información incompleta

### Resultado
Un entendimiento completo del universo que permite:
- Filtrar bonos elegibles para carteras
- Identificar riesgos potenciales
- Entender la estructura del mercado

---

## 💰 PUNTO 2: VALORACIÓN DE BONOS {#punto-2}

### Objetivo
Valorar los bonos del universo utilizando la curva de descuento €STR y comparar los precios teóricos con los precios de mercado.

### ¿Qué se hace?

#### 2.1. Carga de la Curva de Descuento
- Se carga la curva €STR desde `curvaESTR.csv`
- La curva contiene tipos de interés para diferentes plazos (1 mes, 3 meses, 1 año, etc.)
- Se prepara para interpolación

#### 2.2. Interpolación de la Curva
- **Método**: Interpolación exponencial (o lineal si se especifica)
- Permite obtener el tipo de interés para cualquier plazo entre los puntos conocidos
- Se usa para descontar flujos de caja futuros

#### 2.3. Cálculo de Precios Teóricos
Para cada bono:
1. **Generación de flujos de caja**:
   - Cupones periódicos según frecuencia de pago
   - Principal al vencimiento
   - Considera fechas de cupón exactas

2. **Descuento de flujos**:
   - Cada flujo se descuenta usando el tipo de la curva €STR para su plazo específico
   - Fórmula: `PV = Flujo / (1 + r(t))^t`
   - Donde `r(t)` es el tipo interpolado para el plazo `t`

3. **Cálculo de precio sucio**:
   - Suma de todos los valores presentes de flujos
   - Incluye cupón corrido (interés acumulado desde último pago)

4. **Cálculo de precio limpio**:
   - Precio sucio menos cupón corrido
   - Es el precio que se negocia en el mercado

#### 2.4. Comparación con Precios de Mercado
- Se comparan precios teóricos vs precios de mercado (MID)
- Se calculan diferencias absolutas y porcentuales
- Se identifican bonos sobrevalorados/infravalorados

### Resultado
- Precios teóricos para todos los bonos
- Identificación de discrepancias con mercado
- Base para calcular spreads en el siguiente punto

---

## 📈 PUNTO 3: CÁLCULO DE SPREAD IMPLÍCITO {#punto-3}

### Objetivo
Calcular el spread de crédito que el mercado exige sobre la curva libre de riesgo para cada bono.

### ¿Qué se hace?

#### 3.1. Concepto de Spread Implícito
El **spread implícito** es el diferencial adicional que el mercado exige sobre la curva libre de riesgo para compensar:
- **Riesgo de crédito**: Probabilidad de default del emisor
- **Riesgo de liquidez**: Dificultad para vender el bono
- **Otros factores**: Opcionalidad, impuestos, etc.

#### 3.2. Cálculo del Spread
Se resuelve numéricamente el spread `s` que hace que:
```
Precio_Mercado = Σ [Flujo_i / (1 + r(t_i) + s)^t_i]
```

Donde:
- `r(t_i)` es el tipo de la curva €STR para el plazo `t_i`
- `s` es el spread implícito (en decimales)
- Se resuelve iterativamente hasta que el precio teórico coincida con el precio de mercado

#### 3.3. Análisis del Spread
- **Distribución**: Histograma de spreads
- **Por rating**: Comparación de spreads por categoría crediticia
- **Por sector**: Identifica sectores con mayor riesgo percibido
- **Relación con otras variables**: Correlación con duración, liquidez, etc.

### Resultado
- Spread implícito para cada bono (en puntos básicos)
- Entendimiento del riesgo de crédito percibido por el mercado
- Base para construcción de carteras

---

## 📐 PUNTO 4: MÉTRICAS DE BONOS (YTM, DURACIÓN, CONVEXIDAD) {#punto-4}

### Objetivo
Calcular métricas fundamentales de riesgo y rentabilidad para cada bono.

### ¿Qué se hace?

#### 4.1. Yield to Maturity (YTM)
**¿Qué es?**
- Tasa de retorno interna (TIR) que iguala el valor presente de flujos futuros al precio actual
- Mide la rentabilidad esperada si se mantiene el bono hasta vencimiento

**Cálculo**:
- Se resuelve numéricamente: `Precio = Σ [Flujo_i / (1 + YTM)^t_i]`
- Se asume reinversión de cupones al mismo YTM

**Interpretación**:
- YTM alto = mayor rentabilidad esperada (pero también mayor riesgo)
- Permite comparar bonos con diferentes cupones y vencimientos

#### 4.2. Duración Modificada
**¿Qué es?**
- Mide la sensibilidad del precio del bono a cambios en el YTM
- Indica cuánto cambiará el precio si el YTM cambia en 1%

**Cálculo**:
```
Duración Modificada = - (1/Precio) * dP/dYTM
```

**Interpretación**:
- Duración alta = bono muy sensible a cambios de tipos
- Ejemplo: Duración 5 años → si YTM sube 1%, precio cae ~5%
- Útil para gestión de riesgo de tipos de interés

#### 4.3. Convexidad
**¿Qué es?**
- Mide la curvatura de la relación precio-YTM
- Corrige la aproximación lineal de la duración

**Cálculo**:
```
Convexidad = (1/Precio) * d²P/dYTM²
```

**Interpretación**:
- Convexidad positiva = beneficiosa (amplifica ganancias, reduce pérdidas)
- Permite estimar cambios de precio más precisos:
  ```
  ΔPrecio ≈ -Duración * ΔYTM + 0.5 * Convexidad * (ΔYTM)²
  ```

#### 4.4. Análisis de Métricas
- Distribuciones de YTM, Duración, Convexidad
- Correlaciones entre métricas
- Identificación de bonos con características extremas

### Resultado
- Métricas completas para todos los bonos
- Base para construcción de carteras optimizadas
- Herramientas para gestión de riesgo

---

## 📊 PUNTO 5: CARTERA EQUIPONDERADA Y BACKTEST {#punto-5}

### Objetivo
Construir una cartera simple (equiponderada) con todos los bonos del universo y realizar un backtest comparándola con el benchmark.

### ¿Qué se hace?

#### 5.1. Construcción de Cartera Equiponderada
- **Estrategia**: Invertir igual cantidad en cada bono del universo
- **Peso por bono**: `1 / N` donde N es el número de bonos
- **Sin restricciones**: Incluye todos los bonos disponibles

#### 5.2. Backtesting
**Metodología**:
1. **Fecha inicial**: Primera fecha con datos disponibles
2. **Rebalanceo**: Mensual o según frecuencia especificada
3. **Cálculo de retornos**:
   - Entre rebalanceos: `Retorno = (Precio_t+1 - Precio_t + Cupones) / Precio_t`
   - Retorno de cartera: `Σ (peso_i * retorno_i)`
4. **Tracking**: Evolución del valor de la cartera en el tiempo

#### 5.3. Comparación con Benchmark
- **Benchmark**: RECMTREU Index (índice de crédito corporativo europeo)
- **Métricas comparativas**:
  - Retorno total y anualizado
  - Volatilidad
  - Sharpe Ratio
  - Maximum Drawdown
  - Tracking Error
  - Alpha y Beta

#### 5.4. Visualizaciones
- Gráfico de evolución de valor de cartera vs benchmark
- Gráfico de retornos acumulados
- Tabla comparativa de métricas

### Resultado
- Entendimiento del rendimiento de una estrategia pasiva
- Base de comparación para carteras optimizadas
- Métricas de riesgo y retorno

---

## 🎯 PUNTO 6: CARTERA OPTIMIZADA {#punto-6}

### Objetivo
Construir una cartera optimizada de máximo 20 bonos que maximice la rentabilidad sujeto a restricciones específicas del mandato.

### Mandato (Restricciones)
1. **Duración máxima**: ≤ 3 años
2. **Exposición HY máxima**: ≤ 10% de la cartera
3. **Sin deuda subordinada**: Excluir bonos subordinados
4. **Tamaño mínimo**: Outstanding Amount ≥ 500 millones
5. **Peso máximo por bono**: ≤ 10% del capital
6. **Concentración máxima por emisor**: ≤ 15% del capital
7. **Número máximo de bonos**: 20

### ¿Qué se hace?

#### 6.1. Construcción de la Cartera Optimizada
**Proceso de optimización**:

1. **Filtrado inicial**:
   - Bonos vivos (maturity > fecha análisis)
   - Con precios y métricas válidas
   - Que cumplen restricciones básicas (tamaño, seniority)

2. **Optimización**:
   - **Objetivo**: Maximizar YTM ponderado de la cartera
   - **Método**: Optimización no lineal (SLSQP de scipy)
   - **Variables**: Pesos de cada bono (0 a 10% por bono)
   - **Restricciones**:
     - Suma de pesos = 1 (100% invertido)
     - Duración ponderada ≤ 3 años
     - Exposición HY ponderada ≤ 10%
     - Peso por bono ≤ 10%
     - Peso por emisor ≤ 15%
     - Máximo 20 bonos con peso > 0

3. **Resultado**:
   - Lista de bonos seleccionados
   - Pesos optimizados
   - Métricas de la cartera (YTM, duración, exposición HY)

#### 6.2. Restricciones Adicionales Sugeridas
**Análisis de qué otras restricciones añadir**:
- Restricción de liquidez (bid-ask spread máximo)
- Restricción geográfica (países permitidos)
- Restricción de vencimiento (rango de años)
- Restricción de sector (máxima concentración sectorial)

#### 6.3. Medición de Riesgo de Crédito
**Métricas propuestas**:
- **Exposición HY**: Porcentaje en bonos High Yield
- **PD ponderada**: Probabilidad de default promedio ponderada
- **Concentración**: Índice de Herfindahl por emisor
- **Rating promedio**: Rating medio de la cartera
- **Worst-case scenario**: Análisis de pérdidas potenciales

#### 6.4. Medición de Riesgo de Liquidez
**Métricas propuestas**:
- **Bid-ask spread promedio**: Costo de transacción
- **Volumen promedio**: Facilidad de compra/venta
- **Días para liquidar**: Tiempo estimado para vender posiciones
- **Liquidez por sector**: Identificar sectores menos líquidos

#### 6.5. Backtest de la Cartera Optimizada
**Metodología descrita** (con código implementado):

1. **Inicialización**:
   - Fecha de inicio con datos disponibles
   - Capital inicial (ej. 10M€)

2. **Rebalanceo periódico**:
   - Frecuencia: Mensual o trimestral
   - En cada fecha:
     a. Filtrar bonos elegibles (vivos, con precios)
     b. Re-optimizar cartera con datos actualizados
     c. Calcular trades necesarios
     d. Aplicar costos de transacción

3. **Cálculo de retornos**:
   - Retorno por bono: `(Precio_t+1 - Precio_t + Cupones) / Precio_t`
   - Retorno de cartera: `Σ (peso_i * retorno_i)`
   - Total Return: Incluye cupones reinvertidos

4. **Métricas finales**:
   - Retorno total y anualizado
   - Volatilidad
   - Sharpe Ratio
   - Maximum Drawdown
   - Tracking Error vs benchmark

**Código implementado**: Función `backtest_optimized_portfolio()` ⭐ **NUEVA FUNCIONALIDAD**

**Características de la función**:
- **Backtest completo con rebalanceo**: Re-optimiza la cartera en cada fecha de rebalanceo
- **Cálculo automático de retornos**: Calcula retornos entre rebalanceos incluyendo cupones
- **Métricas de rendimiento**: Calcula retorno total, anualizado, volatilidad
- **Tracking completo**: Rastrea el valor de la cartera a lo largo del tiempo
- **Manejo de bonos vivos**: Filtra automáticamente bonos vivos en cada fecha
- **Re-optimización periódica**: Construye nueva cartera optimizada en cada rebalanceo

**Parámetros configurables**:
- Frecuencia de rebalanceo (mensual 'M' o trimestral 'Q')
- Capital inicial
- Fechas de inicio y fin
- Aplica las mismas restricciones del mandato en cada rebalanceo

**Uso**:
```python
resultado = backtest_optimized_portfolio(
    precios_df=precios_historicos_universo,
    universo_df=universo,
    fecha_inicio=datetime(2023, 10, 1),
    fecha_fin=datetime(2025, 9, 1),
    rebalance_freq='M',
    initial_capital=10_000_000
)
```

### Resultado
- Cartera optimizada que cumple todas las restricciones
- Análisis completo de riesgos
- Metodología de backtesting implementada

---

## 🛡️ PUNTO 7: COBERTURA DE TIPOS DE INTERÉS {#punto-7}

### Objetivo
Cubrir la exposición de la cartera a movimientos de tipos de interés usando futuros sobre bonos gubernamentales.

### ¿Qué se hace?

#### 7.1. Cálculo de Sensibilidad (DV01)
**DV01** (Dollar Value of 01) mide cuánto cambia el valor de la cartera si los tipos suben 1 punto básico (0.01%).

```
DV01_cartera = Inversión_total × Duración_cartera × 0.0001
```

Ejemplo: 10M€ × 2.99 años × 0.0001 = 2,988 €/bp

#### 7.2. Selección del Instrumento de Cobertura
**Futuros disponibles**:
- **DU1 (Schatz)**: Futuro sobre bono alemán 2 años, Duración = 1.92 años
- **OE1 (Bobl)**: Futuro sobre bono alemán 5 años, Duración = 5.44 años
- **RX1 (Bund)**: Futuro sobre bono alemán 10 años, Duración = 10.00 años

**Criterio de selección**:
- Se prioriza el futuro con **duración >= duración de la cartera** (mejor práctica)
- Si hay varios, se elige el más cercano (menor mismatch)
- **Razón**: Reduce basis risk y requiere menos contratos

#### 7.3. Cálculo del Número de Contratos
```
DV01_futuro = Duración_futuro × Tamaño_contrato × 0.0001
Número_contratos = DV01_cartera / DV01_futuro
```

Ejemplo con Bobl (OE1):
- DV01_Bobl = 5.44 × 100,000 × 0.0001 = 54.4 €/bp
- Contratos = 2,988 / 54.4 ≈ 55 contratos

#### 7.4. Posición de Cobertura
- **Posición**: VENDER futuros (posición corta)
- **Razón**: Si suben tipos → bonos caen (pérdida) pero futuros suben (ganancia)
- La posición corta compensa la pérdida en la cartera

#### 7.5. Análisis de Escenarios
- **Sobrecobertura**: ¿Qué pasa si vendemos más contratos de los necesarios?
- **Subcobertura**: ¿Qué pasa si vendemos menos?
- **Cobertura parcial**: Ejemplo de 40% (20 contratos en vez de 55)

#### 7.6. Código Ejecutable Dinámico ⭐ **NUEVO**
**Funcionalidad implementada**:
- **Cálculo automático de DV01**: Calcula dinámicamente la sensibilidad de la cartera basándose en la duración real
- **Selección automática de instrumento**: Prioriza futuros con duración >= duración de la cartera
- **Cálculo de contratos**: Calcula automáticamente el número óptimo de contratos necesarios
- **Análisis de escenarios**: 
  - Cobertura total (100%)
  - Cobertura parcial (40% recomendada)
  - Sobrecobertura (100 contratos para análisis)
- **Tabla resumen comparativa**: Muestra diferentes estrategias de cobertura con sus impactos

**Características del código**:
- Se adapta automáticamente a la cartera optimizada construida en el punto 6
- Verifica coherencia con la explicación teórica
- Genera tablas y análisis detallados

#### 7.7. Tabla Resumen
El código genera una tabla comparativa mostrando:
- Cobertura total vs parcial
- DV01 cubierto y expuesto
- Impacto de cada estrategia

### Resultado
- Número exacto de contratos necesarios
- Instrumento óptimo seleccionado automáticamente
- Análisis completo de estrategias de cobertura
- **Código ejecutable que calcula todo dinámicamente** ⭐

---

## 💳 PUNTO 8: COBERTURA DE RIESGO DE CRÉDITO {#punto-8}

### Objetivo
Cubrir total o parcialmente el riesgo de crédito de la cartera usando índices CDS (Credit Default Swaps).

### ¿Qué se hace?

#### 8.1. Selección del Índice CDS
**Índices disponibles**:
- **ITRAXX Main**: Índice de Investment Grade europeo, 5 años
- **ITRAXX XOVER**: Índice de High Yield europeo, 5 años

**Criterio de selección**:
- Si cartera es mayoritariamente IG → ITRAXX Main
- Si cartera tiene exposición HY significativa → ITRAXX XOVER
- Se determina dinámicamente según exposición HY de la cartera

#### 8.2. Cálculo de Sensibilidad
**Sensibilidad CDS**: 4,500 €/bp por 10 millones de notional
- Mide cuánto cambia el valor del CDS si el spread cambia 1 punto básico

#### 8.3. Cobertura Total vs Parcial
**Cobertura Total (100%)**:
- Notional = Inversión total (10M€)
- Elimina completamente la exposición a spreads de crédito
- **Problema**: Elimina la prima de crédito (fuente de rentabilidad)

**Cobertura Parcial (recomendada: 50%)**:
- Notional = 50% de la inversión (5M€)
- Protege ante estrés crediticio severo
- Mantiene exposición a movimientos moderados (beta al mercado IG)

#### 8.4. Posición de Cobertura
- **Posición**: COMPRAR protección (long CDS)
- **Razón**: Si spreads aumentan → bonos caen (pérdida) pero CDS sube (ganancia)
- La posición long CDS compensa la pérdida en la cartera

#### 8.5. Código Ejecutable Dinámico ⭐ **NUEVO**
**Funcionalidad implementada**:
- **Selección automática de índice CDS**: Selecciona ITRAXX Main o XOVER según exposición HY de la cartera
- **Cálculo dinámico de notional**: Calcula automáticamente el notional necesario para cada nivel de cobertura
- **Comparación de niveles**: Analiza múltiples niveles de cobertura (0%, 30%, 50%, 70%, 100%)
- **Verificación de coherencia**: Verifica que la selección del índice coincide con la explicación teórica
- **Tabla comparativa**: Genera tabla con todos los niveles de cobertura y sus características

**Características del código**:
- Se adapta automáticamente a la cartera optimizada
- Determina dinámicamente si usar ITRAXX Main (IG) o XOVER (HY)
- Calcula sensibilidad y notional para cada nivel de cobertura

#### 8.6. Análisis de Niveles de Cobertura
El código compara diferentes niveles:
- 0% (sin cobertura)
- 30% (cobertura ligera)
- 50% (cobertura moderada) ← Recomendado
- 70% (cobertura alta)
- 100% (cobertura total)

#### 8.7. Instrumentos Alternativos
- CDS single-name (cobertura idiosincrática)
- ETF corporativos inversos
- Venta de índices de crédito
- Bonos gubernamentales (flight-to-quality)

### Resultado
- Notional óptimo de cobertura calculado automáticamente
- Índice CDS seleccionado dinámicamente según perfil de la cartera
- Análisis completo de diferentes estrategias
- **Código ejecutable que calcula todo dinámicamente** ⭐

---

## 🎲 PUNTO 9: ESTRATEGIA PROPIA {#punto-9}

### Objetivo
Diseñar una estrategia propia que combine coberturas parciales para gestionar el riesgo sin eliminar completamente la rentabilidad.

### Estrategia Propuesta: Coberturas Parciales

#### 9.1. Filosofía de la Estrategia
- **Problema**: Coberturas totales eliminan rentabilidad
- **Solución**: Coberturas parciales que controlan volatilidad manteniendo exposición moderada
- **Objetivo**: Balance entre protección y rentabilidad

#### 9.2. Componentes de la Estrategia

**1. Cobertura Parcial de Tipos (40%)**
- **Instrumento**: Futuros Bobl (OE1)
- **Posición**: Vender 20 contratos (40% de 55 totales)
- **Efecto**: Reduce sensibilidad ante subidas bruscas, mantiene exposición moderada
- **Razón**: Permite capturar rolldown y movimientos moderados

**2. Cobertura Parcial de Crédito (50%)**
- **Instrumento**: ITRAXX Main CDS
- **Notional**: 5 millones (50% de 10M€)
- **Efecto**: Protege ante estrés crediticio severo, mantiene beta al mercado IG
- **Razón**: La prima de crédito es fuente importante de rentabilidad

**3. Gestión de Riesgo Idiosincrático**
- **Monitorización**: Emisores con mayor vulnerabilidad (Alstom, Vonovia, Amprion)
- **Acción**: Ante señales de deterioro (downgrade, CDS widening), reducir posición o añadir CDS single-name

**4. Rebalanceo Dinámico**
- **Frecuencia**: Trimestral
- **Ajustes**:
  - Recalcular contratos si cambia la duración
  - Ajustar cobertura de crédito según compresión/widening de spreads
  - Incrementar coberturas ante señales de estrés macroeconómico
  - Reducir coberturas si spreads se comprimen en exceso

#### 9.3. Ventajas de la Estrategia
✅ Controla volatilidad sin eliminar rentabilidad
✅ Protege ante eventos adversos (subidas de tipos, estrés crediticio)
✅ Mantiene exposición a movimientos moderados (fuente de retorno)
✅ Reduce costes de cobertura vs. cobertura total
✅ Permite ajustes dinámicos según condiciones de mercado

#### 9.4. Código Ejecutable Dinámico ⭐ **NUEVO**
**Funcionalidad implementada**:
- **Cálculo automático de coberturas parciales combinadas**:
  - Cobertura parcial de tipos (40%): Calcula automáticamente el número de contratos
  - Cobertura parcial de crédito (50%): Calcula automáticamente el notional
- **Análisis de riesgo residual**: Calcula la exposición restante después de las coberturas
- **Verificación de coherencia**: Compara los cálculos dinámicos con la explicación teórica
- **Resumen ejecutivo**: Genera un resumen completo de la estrategia con todos los parámetros

**Características del código**:
- Combina las coberturas de tipos y crédito
- Verifica que la duración y exposición HY coinciden con lo esperado
- Calcula DV01 cubierto y expuesto
- Genera informe completo de la estrategia

#### 9.5. Cálculo Dinámico
El código calcula automáticamente:
- Número de contratos para cobertura parcial de tipos
- Notional para cobertura parcial de crédito
- DV01 cubierto y expuesto
- Verificación de coherencia con explicación teórica
- **Todos los cálculos se hacen dinámicamente basándose en la cartera optimizada real** ⭐

### Resultado
- Estrategia completa y justificada
- Cálculos dinámicos basados en la cartera optimizada real
- Plan de implementación y rebalanceo
- **Código ejecutable completo que implementa toda la estrategia** ⭐

---

## 🔧 ESTRUCTURA TÉCNICA DEL NOTEBOOK

### Módulos Utilizados

#### `utils.py`
- `get_data_path()`: Obtiene ruta de datos
- `load_universe()`: Carga universo de bonos
- `load_and_prepare_curve()`: Carga y prepara curva €STR
- `load_historical_prices_universe()`: Carga precios históricos
- `get_effective_maturity()`: Calcula vencimiento efectivo

#### `valuation.py`
- `valorar_bono()`: Valora bono usando curva de descuento
- `spread_implicito()`: Calcula spread implícito
- `get_discount_from_curve()`: Interpola curva para obtener descuento

#### `metrics.py`
- `calculate_ytm()`: Calcula Yield to Maturity
- `calculate_modified_duration()`: Calcula duración modificada
- `calculate_convexity()`: Calcula convexidad
- `estimate_price_change()`: Estima cambio de precio

#### `portfolio.py`
- `build_optimized_portfolio()`: Construye cartera optimizada
- `backtest_equally_weighted_portfolio()`: Backtest cartera equiponderada
- `get_alive_bonds_at_date()`: Obtiene bonos vivos en fecha
- `calculate_performance_metrics()`: Calcula métricas de rendimiento

#### `analysis.py`
- `analyze_currencies()`: Analiza distribución por moneda
- `analyze_bond_types()`: Analiza tipos de bono
- `analyze_ratings()`: Analiza ratings y riesgo de crédito
- `analyze_sectors()`: Analiza distribución sectorial
- `analyze_liquidity()`: Analiza liquidez

### Flujo de Ejecución

1. **Configuración inicial**: Carga librerías, define paths, fecha de análisis
2. **Punto 1**: Análisis exploratorio de datos
3. **Punto 2**: Valoración de bonos
4. **Punto 3**: Cálculo de spreads
5. **Punto 4**: Cálculo de métricas (YTM, Duración, Convexidad)
6. **Punto 5**: Construcción y backtest de cartera equiponderada
7. **Punto 6**: Construcción de cartera optimizada
8. **Punto 7**: Cálculo de cobertura de tipos de interés
9. **Punto 8**: Cálculo de cobertura de crédito
10. **Punto 9**: Estrategia propia combinada

---

## 📝 NOTAS IMPORTANTES

### Consideraciones Prácticas
- **Costos de transacción**: No incluidos explícitamente pero mencionados
- **Liquidez**: Considerada en análisis pero no siempre en ejecución
- **Rebalanceo**: Frecuencia debe ajustarse según costos y mandato
- **Datos históricos**: Limitados a fechas disponibles

### Limitaciones
- **Asunciones simplificadoras**: Curva plana para YTM, reinversión al mismo YTM
- **No considera**: Costos de transacción detallados, slippage, market impact
- **Optimización**: Puede no converger en algunos casos (se maneja con warnings)

### Mejoras Futuras Sugeridas
- Implementar optimización con PuLP para mayor claridad
- Añadir stress testing (+100bps en ESTR, widening spreads)
- Optimización multi-objetivo (max YTM + min VaR)
- Integración con APIs para datos en tiempo real
- Visualizaciones interactivas (gauge charts, dashboards)

---

## ✅ CONCLUSIÓN

Este notebook proporciona un **análisis completo y profesional** de un universo de bonos corporativos, desde el análisis exploratorio inicial hasta la implementación de estrategias avanzadas de cobertura. 

**Puntos fuertes**:
- Análisis exhaustivo de datos
- Valoración rigurosa usando curvas de descuento
- Optimización de carteras con restricciones realistas
- Estrategias de cobertura bien fundamentadas
- Código modular y reutilizable
- Visualizaciones claras

**Aplicabilidad**:
- Gestión profesional de carteras de renta fija
- Análisis de riesgo crediticio y de tipos
- Construcción de estrategias de inversión
- Backtesting de estrategias

El notebook está **listo para uso profesional** y puede servir como base para análisis más avanzados o integración en sistemas de trading.

---

## 🚀 MEJORAS Y FUNCIONALIDADES EJECUTABLES ⭐

### Novedades Añadidas

Este notebook ha sido mejorado significativamente con **código ejecutable dinámico** para los puntos 6.5, 7, 8 y 9. A diferencia de la versión anterior que solo contenía descripciones teóricas, ahora incluye:

#### 1. Backtest de Cartera Optimizada (Punto 6.5) ⭐
- **Función completa**: `backtest_optimized_portfolio()`
- **Capacidades**:
  - Re-optimización automática en cada fecha de rebalanceo
  - Cálculo de retornos con reinversión de cupones
  - Tracking completo del valor de la cartera
  - Métricas de rendimiento automáticas
- **Ejecutable**: Solo necesitas descomentar y ejecutar el código

#### 2. Cálculo Dinámico de Cobertura de Tipos (Punto 7) ⭐
- **Cálculo automático de DV01**: Basado en la duración real de la cartera
- **Selección inteligente de instrumento**: Prioriza futuros con duración >= cartera
- **Análisis de escenarios**: 
  - Cobertura total (100%)
  - Cobertura parcial (40%)
  - Análisis de sobrecobertura
- **Tabla comparativa**: Genera automáticamente tablas de comparación

#### 3. Cálculo Dinámico de Cobertura de Crédito (Punto 8) ⭐
- **Selección automática de índice**: ITRAXX Main o XOVER según exposición HY
- **Cálculo dinámico de notional**: Para diferentes niveles de cobertura
- **Comparación de niveles**: 0%, 30%, 50%, 70%, 100%
- **Verificación de coherencia**: Confirma que la selección coincide con la teoría

#### 4. Estrategia Combinada Dinámica (Punto 9) ⭐
- **Cálculo de coberturas parciales combinadas**:
  - 40% cobertura de tipos de interés
  - 50% cobertura de crédito
- **Análisis de riesgo residual**: Calcula exposición restante
- **Verificación completa**: Compara cálculos dinámicos con explicación teórica
- **Resumen ejecutivo**: Genera informe completo de la estrategia

### Ventajas de las Mejoras

✅ **Código ejecutable**: Ya no es solo teoría, puedes ejecutarlo directamente
✅ **Cálculos dinámicos**: Se adapta automáticamente a la cartera optimizada real
✅ **Coherencia verificada**: Compara automáticamente con la explicación teórica
✅ **Análisis completo**: Genera tablas y resúmenes automáticamente
✅ **Listo para uso**: Solo descomenta y ejecuta

### Archivos Nuevos

- **RESUMEN_NOTEBOOK.md**: Documentación completa de todo el notebook (este archivo)
- **TallerRF_AnálisisCartera_Enunciado.html**: Exportación HTML del notebook completo

### Cómo Usar las Nuevas Funcionalidades

1. **Ejecuta primero el punto 6.1** para construir la cartera optimizada
2. **Para coberturas (Puntos 7, 8, 9)**: El código detecta automáticamente la cartera optimizada y calcula todo dinámicamente
3. **Para backtest (Punto 6.5)**: Descomenta el código de ejemplo y ajusta las fechas según necesites

---

**Última actualización**: Noviembre 2025 - Merge de mejoras-hedging-backtest a main

