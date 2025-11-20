"""
Módulo de métricas de bonos.

Este módulo contiene funciones para calcular:
- Yield to Maturity (YTM)
- Modified Duration (Duración Modificada)
- Convexidad
"""

import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
from scipy.optimize import fsolve
from typing import Optional, Tuple
try:
    from .utils import get_effective_maturity
except ImportError:
    from utils import get_effective_maturity


def calculate_cash_flows(
    coupon: float,
    maturity_date: datetime,
    fecha_analisis: datetime,
    face: float = 100.0,
    frequency: int = 1,
    effective_maturity: Optional[datetime] = None
) -> Tuple[list, list]:
    """
    Calcula los flujos de caja y tiempos de un bono.
    
    Parameters
    ----------
    coupon : float
        Cupón anual (%)
    maturity_date : datetime
        Fecha de vencimiento
    fecha_analisis : datetime
        Fecha de análisis
    face : float, default 100.0
        Valor nominal
    frequency : int, default 1
        Frecuencia de pago (1=anual, 2=semestral, 4=trimestral, etc.)
    effective_maturity : datetime, optional
        Fecha de vencimiento efectiva (para callables). Si es None, usa maturity_date
    
    Returns
    -------
    tuple
        (cash_flows, times) donde cash_flows es lista de flujos y times es lista de tiempos en años
    """
    # Usar maturity efectiva si está disponible
    if effective_maturity is None:
        effective_maturity = maturity_date
    
    if effective_maturity is None or pd.isna(effective_maturity):
        return [], []
    
    # Calcular años hasta vencimiento
    remaining_years = (effective_maturity - fecha_analisis).days / 365.25
    if remaining_years <= 0:
        return [], []
    
    # Calcular número de pagos
    num_payments = int(remaining_years * frequency)
    if num_payments <= 0:
        num_payments = 1
    
    # Generar flujos de caja
    cash_flows = []
    times = []
    
    coupon_per_period = (coupon / 100.0) * face / frequency
    
    for i in range(1, num_payments + 1):
        t = i / frequency
        if t <= remaining_years:
            # Cupón
            cash_flows.append(coupon_per_period)
            times.append(t)
    
    # Último pago incluye principal
    if times:
        cash_flows[-1] += face
    
    return cash_flows, times


def pv_cash_flows(ytm: float, cash_flows: list, times: list) -> float:
    """
    Calcula el valor presente de flujos de caja dados un YTM.
    
    Parameters
    ----------
    ytm : float
        Yield to Maturity (tasa anual)
    cash_flows : list
        Lista de flujos de caja
    times : list
        Lista de tiempos en años
    
    Returns
    -------
    float
        Valor presente
    """
    if not cash_flows or not times or len(cash_flows) != len(times):
        return 0.0
    
    pv = sum(cf / ((1 + ytm) ** t) for cf, t in zip(cash_flows, times))
    return pv


def calculate_ytm(
    price: float,
    coupon: float,
    maturity_date: datetime,
    fecha_analisis: datetime,
    face: float = 100.0,
    frequency: int = 1,
    effective_maturity: Optional[datetime] = None,
    initial_guess: float = 0.05
) -> float:
    """
    Calcula el Yield to Maturity (YTM) de un bono.
    
    Parameters
    ----------
    price : float
        Precio de mercado del bono (limpio o sucio)
    coupon : float
        Cupón anual (%)
    maturity_date : datetime
        Fecha de vencimiento
    fecha_analisis : datetime
        Fecha de análisis
    face : float, default 100.0
        Valor nominal
    frequency : int, default 1
        Frecuencia de pago
    effective_maturity : datetime, optional
        Fecha de vencimiento efectiva (para callables)
    initial_guess : float, default 0.05
        Estimación inicial del YTM (5%)
    
    Returns
    -------
    float
        YTM anual (o NaN si no se puede calcular)
    """
    if price <= 0 or pd.isna(price):
        return np.nan
    
    # Calcular flujos de caja
    cash_flows, times = calculate_cash_flows(
        coupon, maturity_date, fecha_analisis, face, frequency, effective_maturity
    )
    
    if not cash_flows:
        return np.nan
    
    # Función objetivo: PV(flujos) - precio = 0
    def objetivo(ytm):
        pv = pv_cash_flows(ytm[0], cash_flows, times)
        return pv - price
    
    try:
        # Resolver para YTM
        sol = fsolve(objetivo, x0=[initial_guess], xtol=1e-8, maxfev=100)
        ytm = float(sol[0])
        
        # Validar resultado razonable (entre -10% y 50%)
        if -0.10 <= ytm <= 0.50:
            return ytm
        else:
            return np.nan
    except:
        return np.nan


def calculate_modified_duration(
    price: float,
    ytm: float,
    coupon: float,
    maturity_date: datetime,
    fecha_analisis: datetime,
    face: float = 100.0,
    frequency: int = 1,
    effective_maturity: Optional[datetime] = None
) -> float:
    """
    Calcula la Modified Duration (Duración Modificada) de un bono.
    
    La Modified Duration mide la sensibilidad porcentual del precio
    ante un cambio de 1% en el YTM.
    
    Parameters
    ----------
    price : float
        Precio de mercado del bono
    ytm : float
        Yield to Maturity
    coupon : float
        Cupón anual (%)
    maturity_date : datetime
        Fecha de vencimiento
    fecha_analisis : datetime
        Fecha de análisis
    face : float, default 100.0
        Valor nominal
    frequency : int, default 1
        Frecuencia de pago
    effective_maturity : datetime, optional
        Fecha de vencimiento efectiva (para callables)
    
    Returns
    -------
    float
        Modified Duration en años (o NaN si no se puede calcular)
    """
    if pd.isna(ytm) or price <= 0:
        return np.nan
    
    # Calcular flujos de caja
    cash_flows, times = calculate_cash_flows(
        coupon, maturity_date, fecha_analisis, face, frequency, effective_maturity
    )
    
    if not cash_flows:
        return np.nan
    
    # Calcular Macaulay Duration
    pv_sum = 0.0
    weighted_sum = 0.0
    
    for cf, t in zip(cash_flows, times):
        pv = cf / ((1 + ytm) ** t)
        pv_sum += pv
        weighted_sum += t * pv
    
    if pv_sum <= 0:
        return np.nan
    
    macaulay_duration = weighted_sum / pv_sum
    
    # Modified Duration = Macaulay Duration / (1 + ytm)
    modified_duration = macaulay_duration / (1 + ytm)
    
    return modified_duration


def calculate_convexity(
    price: float,
    ytm: float,
    coupon: float,
    maturity_date: datetime,
    fecha_analisis: datetime,
    face: float = 100.0,
    frequency: int = 1,
    effective_maturity: Optional[datetime] = None
) -> float:
    """
    Calcula la Convexidad de un bono.
    
    La Convexidad mide la curvatura de la relación precio-yield,
    corrigiendo la aproximación lineal de la Duration.
    
    Parameters
    ----------
    price : float
        Precio de mercado del bono
    ytm : float
        Yield to Maturity
    coupon : float
        Cupón anual (%)
    maturity_date : datetime
        Fecha de vencimiento
    fecha_analisis : datetime
        Fecha de análisis
    face : float, default 100.0
        Valor nominal
    frequency : int, default 1
        Frecuencia de pago
    effective_maturity : datetime, optional
        Fecha de vencimiento efectiva (para callables)
    
    Returns
    -------
    float
        Convexidad (o NaN si no se puede calcular)
    """
    if pd.isna(ytm) or price <= 0:
        return np.nan
    
    # Calcular flujos de caja
    cash_flows, times = calculate_cash_flows(
        coupon, maturity_date, fecha_analisis, face, frequency, effective_maturity
    )
    
    if not cash_flows:
        return np.nan
    
    # Calcular convexidad
    convexity_sum = 0.0
    
    for cf, t in zip(cash_flows, times):
        pv = cf / ((1 + ytm) ** t)
        convexity_sum += pv * t * (t + 1) / ((1 + ytm) ** 2)
    
    if price <= 0:
        return np.nan
    
    convexity = convexity_sum / price
    
    return convexity


def estimate_price_change(
    price: float,
    duration: float,
    convexity: float,
    yield_change_bps: float
) -> Tuple[float, float]:
    """
    Estima el cambio de precio usando Duration y Convexidad.
    
    Fórmula: ΔP/P ≈ -Duration * Δy + 0.5 * Convexity * (Δy)^2
    
    Parameters
    ----------
    price : float
        Precio inicial
    duration : float
        Modified Duration
    convexity : float
        Convexidad
    yield_change_bps : float
        Cambio en yield en puntos básicos
    
    Returns
    -------
    tuple
        (cambio_porcentual, nuevo_precio)
    """
    if pd.isna(duration) or pd.isna(convexity):
        return np.nan, np.nan
    
    # Convertir bps a decimal
    dy = yield_change_bps / 10000.0
    
    # Estimar cambio porcentual
    change_pct = -duration * dy + 0.5 * convexity * (dy ** 2)
    
    # Estimar nuevo precio
    new_price = price * (1 + change_pct)
    
    return change_pct * 100, new_price

