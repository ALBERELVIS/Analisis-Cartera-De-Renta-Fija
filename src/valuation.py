"""
Módulo de valoración de bonos.

Este módulo contiene funciones para:
- Interpolación de curvas de descuento
- Generación de fechas de cupón
- Valoración de bonos (precio limpio, cupón corrido, precio sucio)
- Cálculo de spreads implícitos
"""

import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
from scipy.optimize import fsolve
from typing import Tuple, Optional
try:
    from .utils import get_effective_maturity
except ImportError:
    from utils import get_effective_maturity


def get_discount_from_curve(curva_work: pd.DataFrame, t: float) -> float:
    """
    Devuelve el factor de descuento DF(t) interpolando exponencialmente 
    los factores de descuento de la curva.
    
    La interpolación es log-lineal en los factores de descuento,
    lo que corresponde a interpolación exponencial en tasas.
    
    Parameters
    ----------
    curva_work : pd.DataFrame
        DataFrame con columnas 'Tenor' y 'Discount'
    t : float
        Tiempo (en años) para el cual calcular el factor de descuento
    
    Returns
    -------
    float
        Factor de descuento interpolado
    """
    tenors = curva_work['Tenor'].to_numpy()
    discounts = curva_work['Discount'].to_numpy()

    # Extrapolación antes del primer tenor
    if t <= tenors[0]:
        return float(discounts[0])
    
    # Extrapolación después del último tenor
    if t >= tenors[-1]:
        return float(discounts[-1])

    # Interpolación entre tenors
    idx = np.searchsorted(tenors, t)
    t1, t2 = tenors[idx-1], tenors[idx]
    d1, d2 = discounts[idx-1], discounts[idx]

    # Interpolación exponencial → log-lineal en DF
    if d1 <= 0 or d2 <= 0:
        # Si hay factores de descuento no positivos, usar interpolación lineal
        w = (t - t1) / (t2 - t1)
        return float(d1 + w * (d2 - d1))

    # Interpolación log-lineal
    logd1, logd2 = np.log(d1), np.log(d2)
    w = (t - t1) / (t2 - t1)
    return float(np.exp(logd1 + w * (logd2 - logd1)))


def generate_coupon_dates(
    row: pd.Series, 
    fecha_analisis: datetime,
    effective_maturity: Optional[pd.Timestamp] = None
) -> list:
    """
    Genera fechas de cupón futuras para un bono.
    
    Usa la fecha de vencimiento efectiva (considerando callable) y
    la frecuencia de cupón para generar todas las fechas futuras.
    
    Parameters
    ----------
    row : pd.Series
        Fila del DataFrame con información del bono
    fecha_analisis : datetime
        Fecha de análisis
    effective_maturity : pd.Timestamp, optional
        Fecha de vencimiento efectiva. Si es None, se calcula automáticamente
    
    Returns
    -------
    list
        Lista de fechas de cupón futuras (ordenadas)
    """
    # Obtener frecuencia de cupón
    try:
        freq = int(row["Coupon Frequency"])
        if freq <= 0:
            freq = 1
    except (ValueError, TypeError, KeyError):
        freq = 1

    months_step = int(12 // freq) if freq > 0 else 12

    # Obtener fecha de vencimiento efectiva
    if effective_maturity is None:
        effective_maturity = get_effective_maturity(row)

    if effective_maturity is None or pd.isna(effective_maturity):
        return []

    # Generar fechas desde el vencimiento hacia atrás
    dates = []
    d = effective_maturity
    while d > fecha_analisis:
        dates.append(d)
        try:
            d = d - relativedelta(months=months_step)
        except:
            break

    return sorted(dates)


def calculate_accrued_interest(
    coupon_per_period: float,
    pay_dates: list,
    fecha_analisis: datetime,
    freq: int
) -> float:
    """
    Calcula el cupón corrido usando base ACT/365.
    
    Parameters
    ----------
    coupon_per_period : float
        Cupón por periodo
    pay_dates : list
        Lista de fechas de cupón (debe estar ordenada)
    fecha_analisis : datetime
        Fecha de análisis
    freq : int
        Frecuencia de pago de cupones
    
    Returns
    -------
    float
        Cupón corrido
    """
    if not pay_dates:
        return 0.0
    
    # Encontrar el último cupón pagado y el próximo
    next_coupon = pay_dates[0]
    step_days = int(round(365 / freq))
    
    # Buscar el último cupón pagado
    last_coupon = None
    for i, coupon_date in enumerate(pay_dates):
        if coupon_date <= fecha_analisis:
            last_coupon = coupon_date
        else:
            next_coupon = coupon_date
            if i > 0:
                last_coupon = pay_dates[i-1]
            break
    
    # Si no hay último cupón, calcularlo desde el próximo
    if last_coupon is None:
        last_coupon = next_coupon - relativedelta(months=int(12/freq))
    
    # Calcular días transcurridos
    days_since = max((fecha_analisis - last_coupon).days, 0)
    days_full = max((next_coupon - last_coupon).days, 1)
    accrual_factor = days_since / days_full

    return coupon_per_period * accrual_factor


def valorar_bono(
    row: pd.Series,
    fecha_analisis: datetime,
    curva_work: pd.DataFrame,
    spread_bps: float = 0.0,
    nominal: float = 100.0
) -> Tuple[float, float, float]:
    """
    Valoración de un bono según el enunciado.
    
    Calcula precio limpio, cupón corrido y precio sucio utilizando:
    - Cupones fijos hasta vencimiento/call
    - Base ACT/365
    - Interpolación exponencial de discount factors
    - Spread de crédito opcional (en bps)
    
    Parameters
    ----------
    row : pd.Series
        Fila del DataFrame con información del bono
    fecha_analisis : datetime
        Fecha de análisis
    curva_work : pd.DataFrame
        Curva de descuento con columnas 'Tenor' y 'Discount'
    spread_bps : float, default 0.0
        Spread de crédito en puntos básicos
    nominal : float, default 100.0
        Valor nominal del bono
    
    Returns
    -------
    tuple
        (precio_limpio, cupón_corridido, precio_sucio)
    """
    # Validar moneda (solo EUR)
    if row.get("Ccy", "").upper() != "EUR":
        return np.nan, np.nan, np.nan

    # Obtener cupón anual (%)
    try:
        coup_rate = float(row["Coupon"])
    except (ValueError, TypeError, KeyError):
        return np.nan, np.nan, np.nan

    # Obtener frecuencia
    try:
        freq = int(row["Coupon Frequency"])
        if freq <= 0:
            freq = 1
    except (ValueError, TypeError, KeyError):
        freq = 1

    # Generar fechas de cupón
    pay_dates = generate_coupon_dates(row, fecha_analisis)
    if not pay_dates:
        return np.nan, np.nan, np.nan

    # Cupón por periodo
    coupon_per_period = nominal * (coup_rate / 100.0) / freq

    # Calcular precio sucio = suma de valores presentes
    dirty = 0.0
    for d in pay_dates:
        # Calcular tiempo en años (ACT/365)
        t = (d - fecha_analisis).days / 365.0
        if t <= 0:
            continue

        # Obtener factor de descuento
        df = get_discount_from_curve(curva_work, t)

        # Aplicar spread de crédito si es necesario
        if spread_bps != 0.0:
            s = spread_bps / 10000.0
            df *= np.exp(-s * t)

        # Calcular flujo de caja
        cf = coupon_per_period
        if d == pay_dates[-1]:  # Último cupón incluye principal
            cf += nominal

        dirty += cf * df

    # Calcular cupón corrido
    accrued = calculate_accrued_interest(
        coupon_per_period, pay_dates, fecha_analisis, freq
    )
    
    # Precio limpio = precio sucio - cupón corrido
    clean = dirty - accrued

    return clean, accrued, dirty


def spread_implicito(
    row: pd.Series,
    fecha_analisis: datetime,
    curva_work: pd.DataFrame,
    nominal: float = 100.0,
    tol: float = 1e-6
) -> float:
    """
    Calcula el spread de crédito implícito (en bps) tal que el precio
    limpio teórico coincide con el precio de mercado.
    
    Parameters
    ----------
    row : pd.Series
        Fila del DataFrame con información del bono (debe incluir 'Price')
    fecha_analisis : datetime
        Fecha de análisis
    curva_work : pd.DataFrame
        Curva de descuento con columnas 'Tenor' y 'Discount'
    nominal : float, default 100.0
        Valor nominal del bono
    tol : float, default 1e-6
        Tolerancia para la convergencia del solver
    
    Returns
    -------
    float
        Spread implícito en puntos básicos (o NaN si no se puede calcular)
    """
    precio_mercado = row.get("Price")
    
    # Validar precio de mercado
    if pd.isna(precio_mercado) or precio_mercado <= 0:
        return np.nan

    # Función objetivo para fsolve
    def objetivo(spread_bps):
        precio_modelo, _, _ = valorar_bono(
            row, fecha_analisis, curva_work, 
            spread_bps=spread_bps[0], nominal=nominal
        )
        if pd.isna(precio_modelo):
            return 1e6  # Penalización si no se puede valorar
        return precio_modelo - precio_mercado

    try:
        # Resolver para encontrar el spread que iguala los precios
        # Usamos un punto inicial razonable (100 bps)
        sol = fsolve(objetivo, x0=[100.0], xtol=tol, maxfev=100)
        spread = float(sol[0])
        
        # Validar resultado razonable (entre -500 y 5000 bps)
        if -500 <= spread <= 5000:
            return spread
        else:
            return np.nan
    except:
        return np.nan

