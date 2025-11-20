"""
Módulo de utilidades para análisis de carteras de renta fija.

Este módulo contiene funciones para:
- Carga y preparación de datos
- Manejo de curvas de tipos de interés
- Funciones auxiliares de fechas y cálculos
"""

import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
from typing import Optional, Tuple
import os


def get_data_path(current_dir: str = None) -> str:
    """
    Obtiene la ruta a la carpeta de datos.
    
    Parameters
    ----------
    current_dir : str, optional
        Directorio actual. Si es None, usa os.getcwd()
    
    Returns
    -------
    str
        Ruta a la carpeta de datos
    """
    if current_dir is None:
        current_dir = os.getcwd()
    
    if 'src' in current_dir:
        data_path = os.path.join(os.path.dirname(current_dir), 'data')
    elif os.path.exists('../data'):
        data_path = '../data'
    else:
        data_path = 'data'
    
    return data_path


def load_universe(data_path: str, fecha_analisis: datetime) -> pd.DataFrame:
    """
    Carga y prepara el universo de bonos.
    
    Parameters
    ----------
    data_path : str
        Ruta a la carpeta de datos
    fecha_analisis : datetime
        Fecha de análisis para filtrar bonos vivos
    
    Returns
    -------
    pd.DataFrame
        DataFrame con el universo de bonos preparado
    """
    # Cargar universo
    universo = pd.read_csv(os.path.join(data_path, 'universo.csv'), sep=';')
    
    # Convertir fechas
    universo['Maturity'] = pd.to_datetime(universo['Maturity'], format='%d/%m/%Y', errors='coerce')
    
    if 'Next Call Date' in universo.columns:
        universo['Next Call Date'] = pd.to_datetime(
            universo['Next Call Date'], format='%d/%m/%Y', errors='coerce'
        )
    
    # Filtrar bonos vivos
    vivos = universo[universo['Maturity'] > fecha_analisis].copy()
    
    # Calcular años a maturity (considerando callable)
    vivos['Años a maturity'] = vivos.apply(
        lambda row: calculate_years_to_maturity(row, fecha_analisis), axis=1
    )
    
    # Calcular Bid-Ask Spread
    if 'Bid Price' in vivos.columns and 'Ask Price' in vivos.columns:
        bid_price = pd.to_numeric(vivos['Bid Price'], errors='coerce')
        ask_price = pd.to_numeric(vivos['Ask Price'], errors='coerce')
        vivos['Bid-Ask Spread'] = ask_price - bid_price
    else:
        vivos['Bid-Ask Spread'] = np.nan
        print("Advertencia: No se encontraron columnas 'Bid Price' o 'Ask Price'")
    
    return vivos


def calculate_years_to_maturity(row: pd.Series, fecha_analisis: datetime) -> float:
    """
    Calcula años hasta maturity, considerando bonos callable.
    
    Parameters
    ----------
    row : pd.Series
        Fila del DataFrame con información del bono
    fecha_analisis : datetime
        Fecha de análisis
    
    Returns
    -------
    float
        Años hasta maturity (o NaN si no hay fecha válida)
    """
    # Si es callable y tiene Next Call Date, usar esa fecha
    if pd.notna(row.get('Callable')) and str(row.get('Callable')).upper() == 'Y':
        if pd.notna(row.get('Next Call Date')):
            eff_maturity = row['Next Call Date']
        else:
            eff_maturity = row['Maturity']
    else:
        eff_maturity = row['Maturity']
    
    # Si no hay maturity válida, retornar NaN
    if pd.isna(eff_maturity):
        return np.nan
    
    # Calcular años desde fecha_analisis
    return (eff_maturity - fecha_analisis).days / 365.25


def get_effective_maturity(row: pd.Series) -> Optional[pd.Timestamp]:
    """
    Obtiene la fecha de vencimiento efectiva de un bono (considerando callable).
    
    Parameters
    ----------
    row : pd.Series
        Fila del DataFrame con información del bono
    
    Returns
    -------
    pd.Timestamp or None
        Fecha de vencimiento efectiva
    """
    call_flag = str(row.get('Callable', 'N')).upper()
    call_date = row.get('Next Call Date')
    maturity = row.get('Maturity')
    
    # Si es callable y tiene Next Call Date, usar esa fecha
    if call_flag == 'Y' and pd.notna(call_date):
        return call_date
    
    return maturity if pd.notna(maturity) else None


def load_and_prepare_curve(data_path: str, fecha_analisis: datetime) -> pd.DataFrame:
    """
    Carga y prepara la curva de tipos de interés €STR.
    
    Parameters
    ----------
    data_path : str
        Ruta a la carpeta de datos
    fecha_analisis : datetime
        Fecha de análisis para calcular tenors
    
    Returns
    -------
    pd.DataFrame
        DataFrame con la curva preparada (Tenor, Zero Rate, Discount)
    """
    # Cargar curva
    curva = pd.read_csv(os.path.join(data_path, 'curvaESTR.csv'), sep=';')
    curva_work = curva.copy()
    
    # Convertir fechas
    curva_work['Date'] = pd.to_datetime(curva_work['Date'], format='%d/%m/%Y', errors='coerce')
    curva_work = curva_work.dropna(subset=['Date'])
    
    # Calcular tenors
    curva_work['Tenor'] = (curva_work['Date'] - fecha_analisis).dt.days / 365.25
    
    # Normalizar Zero Rate (si está en porcentaje, convertir a decimal)
    if 'Zero Rate' in curva_work.columns:
        if curva_work['Zero Rate'].max() > 1:
            curva_work['Zero Rate'] /= 100
    
    # Calcular factores de descuento si no existen
    if 'Discount' not in curva_work.columns or curva_work['Discount'].isna().any():
        curva_work['Discount'] = np.exp(-curva_work['Zero Rate'] * curva_work['Tenor'])
    
    # Ordenar y seleccionar columnas relevantes
    curva_work = curva_work[['Tenor', 'Zero Rate', 'Discount']].sort_values('Tenor')
    
    return curva_work


def load_historical_prices_universe(data_path: str) -> pd.DataFrame:
    """
    Carga y prepara los precios históricos del universo.
    
    Parameters
    ----------
    data_path : str
        Ruta a la carpeta de datos
    
    Returns
    -------
    pd.DataFrame
        DataFrame con precios históricos (ISIN x fechas)
    """
    precios_universo = pd.read_csv(
        os.path.join(data_path, 'precios_historicos_universo.csv'), 
        sep=';', 
        low_memory=False
    )
    
    # Establecer ISIN como índice
    precios_universo.set_index('Unnamed: 0', inplace=True)
    
    # Reemplazar #N/D con NaN
    precios_universo = precios_universo.replace('#N/D', pd.NA)
    
    return precios_universo


def load_historical_prices_various(data_path: str) -> pd.DataFrame:
    """
    Carga y prepara los precios históricos de varios instrumentos.
    
    Parameters
    ----------
    data_path : str
        Ruta a la carpeta de datos
    
    Returns
    -------
    pd.DataFrame
        DataFrame con precios históricos (fechas x instrumentos)
    """
    precios_varios = pd.read_csv(
        os.path.join(data_path, 'precios_historicos_varios.csv'), 
        sep=';', 
        index_col=0
    )
    
    # Convertir índice a datetime
    precios_varios.index = pd.to_datetime(precios_varios.index, format='%d/%m/%Y', errors='coerce')
    
    # Reemplazar #N/D y convertir a numérico
    precios_varios = precios_varios.replace('#N/D', pd.NA)
    precios_varios = precios_varios.apply(pd.to_numeric, errors='coerce')
    
    return precios_varios


def clean_missing_values(df: pd.DataFrame, method: str = 'ffill', axis: int = 0) -> pd.DataFrame:
    """
    Limpia valores faltantes en un DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame a limpiar
    method : str, default 'ffill'
        Método de limpieza: 'ffill' (forward fill), 'bfill' (backward fill), 'drop' (eliminar)
    axis : int, default 0
        Eje sobre el cual aplicar el método (0=filas, 1=columnas)
    
    Returns
    -------
    pd.DataFrame
        DataFrame limpio
    """
    if method == 'ffill':
        return df.ffill(axis=axis)
    elif method == 'bfill':
        return df.bfill(axis=axis)
    elif method == 'drop':
        return df.dropna(axis=axis)
    else:
        raise ValueError(f"Método '{method}' no reconocido. Use 'ffill', 'bfill' o 'drop'")


