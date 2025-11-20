"""
Módulo de gestión de carteras y backtesting.

Este módulo contiene funciones para:
- Construcción de carteras equiponderadas
- Backtesting de estrategias
- Cálculo de retornos totales
- Comparación con benchmarks
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


def get_alive_bonds_at_date(
    precios_df: pd.DataFrame,
    universo_df: pd.DataFrame,
    fecha: datetime
) -> pd.Index:
    """
    Obtiene los ISINs de bonos vivos en una fecha determinada.
    
    Un bono está vivo si:
    - Tiene precio en esa fecha (no es NaN)
    - Su maturity es posterior a la fecha
    
    Parameters
    ----------
    precios_df : pd.DataFrame
        DataFrame con precios históricos (ISIN x fechas)
    universo_df : pd.DataFrame
        DataFrame con información del universo (debe tener columna 'Maturity')
    fecha : datetime
        Fecha de análisis
    
    Returns
    -------
    pd.Index
        Índice con ISINs de bonos vivos en esa fecha
    """
    # Convertir columnas de precios_df a datetime si es necesario
    if len(precios_df.columns) > 0:
        if not isinstance(precios_df.columns[0], pd.Timestamp):
            try:
                precios_df_cols = pd.to_datetime(precios_df.columns, format='%d/%m/%Y', errors='coerce')
                precios_df.columns = precios_df_cols
            except:
                pass
    
    # Obtener bonos con precio en esa fecha
    # Las columnas pueden ser datetime o strings
    if fecha not in precios_df.columns:
        # Intentar buscar fecha más cercana disponible
        available_dates = [d for d in precios_df.columns if pd.notna(d)]
        if len(available_dates) == 0:
            return pd.Index([])
        # Convertir a datetime si es necesario
        if isinstance(available_dates[0], str):
            available_dates_dt = pd.to_datetime(available_dates, format='%d/%m/%Y', errors='coerce')
            closest_dates = [d for d in available_dates_dt if pd.notna(d) and d <= fecha]
        else:
            closest_dates = [d for d in available_dates if d <= fecha]
        if len(closest_dates) == 0:
            return pd.Index([])
        fecha = max(closest_dates)
    
    try:
        precios_fecha = precios_df[fecha]
        bonos_con_precio = precios_fecha.dropna().index
        # Asegurar que los índices sean strings para comparación
        bonos_con_precio = pd.Index([str(x) for x in bonos_con_precio])
    except Exception as e:
        print(f"Error obteniendo precios en fecha {fecha}: {e}")
        return pd.Index([])
    
    # Filtrar por maturity (bonos vivos)
    universo_copy = universo_df.copy()
    
    # Asegurar que existe columna ISIN en universo
    if 'ISIN' not in universo_copy.columns:
        # Si no existe, intentar usar el índice
        if isinstance(universo_copy.index, pd.Index):
            universo_copy['ISIN'] = universo_copy.index.astype(str)
        else:
            universo_copy['ISIN'] = universo_copy.index.astype(str)
    
    if 'Maturity' in universo_copy.columns:
        universo_copy['Maturity'] = pd.to_datetime(universo_copy['Maturity'], errors='coerce')
        bonos_vivos = universo_copy[universo_copy['Maturity'] > fecha].copy()
        
        # Obtener ISINs de bonos vivos y convertir a string para comparación
        if 'ISIN' in bonos_vivos.columns:
            bonos_vivos_isins = pd.Index([str(x) for x in bonos_vivos['ISIN'].dropna().unique()])
        else:
            bonos_vivos_isins = pd.Index([str(x) for x in bonos_vivos.index])
    else:
        # Si no hay columna Maturity, asumir que todos los bonos con precio están vivos
        bonos_vivos_isins = bonos_con_precio
    
    # Intersectar: bonos con precio Y vivos
    # Ambos ya están como Index de strings
    result = bonos_con_precio.intersection(bonos_vivos_isins)
    return result


def calculate_total_return(
    precio_inicial: float,
    precio_final: float,
    coupon_rate: float,
    days_held: int,
    frequency: int = 1
) -> float:
    """
    Calcula el retorno total de un bono (precio + cupones).
    
    Parameters
    ----------
    precio_inicial : float
        Precio inicial (clean price)
    precio_final : float
        Precio final (clean price)
    coupon_rate : float
        Tasa de cupón anual (%)
    days_held : int
        Días mantenidos
    frequency : int, default 1
        Frecuencia de pago de cupones (1=anual, 2=semestral, etc.)
    
    Returns
    -------
    float
        Retorno total (en %)
    """
    if precio_inicial <= 0 or pd.isna(precio_inicial) or pd.isna(precio_final):
        return 0.0
    
    # Retorno por precio
    price_return = (precio_final - precio_inicial) / precio_inicial
    
    # Cupones recibidos durante el período
    coupons_received = (coupon_rate / 100.0) * (days_held / 365.25) * frequency
    
    # Retorno total
    total_return = price_return + coupons_received
    
    return total_return * 100  # En porcentaje


def backtest_equally_weighted_portfolio(
    precios_df: pd.DataFrame,
    universo_df: pd.DataFrame,
    precios_varios_df: pd.DataFrame,
    fecha_inicio: datetime,
    fecha_fin: Optional[datetime] = None,
    rebalance_frequency: str = 'M',
    benchmark_col: str = 'RECMTREU Index'
) -> Dict:
    """
    Hace backtest de una cartera equiponderada con rebalanceo periódico.
    
    Parameters
    ----------
    precios_df : pd.DataFrame
        DataFrame con precios históricos del universo (ISIN x fechas)
    universo_df : pd.DataFrame
        DataFrame con información del universo (debe incluir 'ISIN', 'Coupon', 'Maturity')
    precios_varios_df : pd.DataFrame
        DataFrame con precios de otros instrumentos incluyendo benchmark (fechas x instrumentos)
    fecha_inicio : datetime
        Fecha de inicio del backtest
    fecha_fin : datetime, optional
        Fecha de fin del backtest. Si es None, usa la última fecha disponible
    rebalance_frequency : str, default 'M'
        Frecuencia de rebalanceo ('M'=mensual, 'W'=semanal, 'D'=diario)
    benchmark_col : str, default 'RECMTREU Index'
        Nombre de la columna del benchmark en precios_varios_df
    
    Returns
    -------
    dict
        Diccionario con resultados del backtest:
        - 'portfolio_value': Series con valor de la cartera en cada fecha
        - 'benchmark_value': Series con valor del benchmark en cada fecha
        - 'portfolio_returns': Series con retornos de la cartera
        - 'benchmark_returns': Series con retornos del benchmark
        - 'rebalance_dates': Lista de fechas de rebalanceo
        - 'positions': DataFrame con posiciones en cada fecha de rebalanceo
    """
    # Preparar fechas
    if fecha_fin is None:
        # Obtener última fecha disponible en precios
        fechas_disponibles = pd.to_datetime(precios_df.columns, format='%d/%m/%Y', errors='coerce')
        fecha_fin = fechas_disponibles.max()
    
    # Convertir columnas de precios_df a datetime para facilitar búsqueda
    fecha_cols = pd.to_datetime(precios_df.columns, format='%d/%m/%Y', errors='coerce')
    precios_df.columns = fecha_cols
    
    # Generar fechas de rebalanceo
    rebalance_dates = pd.date_range(start=fecha_inicio, end=fecha_fin, freq=rebalance_frequency)
    rebalance_dates = rebalance_dates[rebalance_dates <= fecha_fin]
    
    # Inicializar variables
    portfolio_value = pd.Series(dtype=float)
    portfolio_holdings = {}
    
    # Valor inicial de la cartera (normalizado a 100)
    initial_value = 100.0
    current_value = initial_value
    
    # Preparar universo con ISIN como índice si no lo está
    if 'ISIN' in universo_df.columns:
        try:
            universo_indexed = universo_df.set_index('ISIN', drop=False)
        except:
            # Si hay duplicados, mantener índice original
            universo_indexed = universo_df.copy()
    else:
        universo_indexed = universo_df.copy()
    
    # Iterar sobre fechas de rebalanceo
    for i, rebalance_date in enumerate(rebalance_dates):
        # Obtener bonos vivos en esta fecha
        alive_isins = get_alive_bonds_at_date(precios_df, universo_df, rebalance_date)
        
        if len(alive_isins) == 0:
            # Si no hay bonos vivos, mantener valor actual
            portfolio_value[rebalance_date] = current_value
            continue
        
        # Obtener precios en fecha de rebalanceo
        try:
            precios_fecha = precios_df.loc[alive_isins, rebalance_date]
        except KeyError:
            # Si la fecha no existe en precios, buscar fecha más cercana
            available_dates = [d for d in precios_df.columns if pd.notna(d) and d <= rebalance_date]
            if len(available_dates) == 0:
                portfolio_value[rebalance_date] = current_value
                continue
            closest_date = max(available_dates)
            precios_fecha = precios_df.loc[alive_isins, closest_date]
        
        precios_fecha = precios_fecha.dropna()
        
        # Si no hay precios disponibles, mantener valor actual
        if len(precios_fecha) == 0:
            portfolio_value[rebalance_date] = current_value
            continue
        
        # Construir cartera equiponderada solo con bonos que tienen precio
        alive_isins = precios_fecha.index
        num_bonds = len(alive_isins)
        weight_per_bond = 1.0 / num_bonds if num_bonds > 0 else 0.0
        
        # Guardar posición
        holdings = pd.Series(weight_per_bond, index=alive_isins)
        portfolio_holdings[rebalance_date] = holdings
        
        # Si no es la última fecha, calcular valor hasta siguiente rebalanceo
        if i < len(rebalance_dates) - 1:
            next_rebalance = rebalance_dates[i + 1]
            
            # Calcular retorno entre fechas
            fecha_actual = rebalance_date
            fecha_siguiente = next_rebalance
            
            # Obtener precios en fecha actual
            try:
                precios_actual = precios_df.loc[alive_isins, fecha_actual]
            except KeyError:
                available_dates = [d for d in precios_df.columns if pd.notna(d) and d <= fecha_actual]
                if len(available_dates) == 0:
                    portfolio_value[rebalance_date] = current_value
                    continue
                fecha_actual = max(available_dates)
                precios_actual = precios_df.loc[alive_isins, fecha_actual]
            
            # Obtener precios en fecha siguiente
            if fecha_siguiente in precios_df.columns:
                precios_siguiente = precios_df.loc[alive_isins, fecha_siguiente]
            else:
                # Buscar fecha más cercana disponible
                available_dates = [d for d in precios_df.columns if pd.notna(d) and fecha_actual < d <= fecha_siguiente]
                if len(available_dates) == 0:
                    portfolio_value[rebalance_date] = current_value
                    continue
                fecha_siguiente = min(available_dates)
                precios_siguiente = precios_df.loc[alive_isins, fecha_siguiente]
            
            if precios_siguiente.isna().all():
                # Si no hay precios en la siguiente fecha, mantener valor
                portfolio_value[rebalance_date] = current_value
                continue
            
            # Calcular retorno de la cartera
            portfolio_return = 0.0
            days_held = (fecha_siguiente - fecha_actual).days
            
            for isin in alive_isins:
                # Buscar información del bono en el universo
                bond_info = None
                coupon_rate = 0
                freq = 1
                
                # Intentar diferentes formas de buscar el bono
                if isinstance(universo_indexed.index, pd.Index):
                    if isin in universo_indexed.index:
                        bond_info = universo_indexed.loc[isin]
                    elif 'ISIN' in universo_indexed.columns:
                        # Buscar por columna ISIN
                        mask = universo_indexed['ISIN'] == isin
                        if mask.any():
                            bond_info = universo_indexed[mask].iloc[0]
                
                if bond_info is not None:
                    if isinstance(bond_info, pd.Series):
                        coupon_rate = bond_info.get('Coupon', 0) if pd.notna(bond_info.get('Coupon', 0)) else 0
                        freq = int(bond_info.get('Coupon Frequency', 1)) if pd.notna(bond_info.get('Coupon Frequency', 1)) else 1
                    else:
                        coupon_rate = getattr(bond_info, 'Coupon', 0) if hasattr(bond_info, 'Coupon') else 0
                        freq = int(getattr(bond_info, 'Coupon Frequency', 1)) if hasattr(bond_info, 'Coupon Frequency') else 1
                
                # Obtener precios
                precio_inicial = precios_actual.get(isin) if isinstance(precios_actual, pd.Series) else precios_actual.get(isin, None)
                precio_final = precios_siguiente.get(isin) if isinstance(precios_siguiente, pd.Series) else precios_siguiente.get(isin, None)
                
                # Calcular retorno si tenemos ambos precios válidos
                if pd.notna(precio_inicial) and pd.notna(precio_final) and precio_inicial > 0:
                    bond_return = calculate_total_return(
                        float(precio_inicial), float(precio_final), 
                        float(coupon_rate), days_held, freq
                    )
                    portfolio_return += weight_per_bond * bond_return
            
            # Actualizar valor de la cartera
            current_value = current_value * (1 + portfolio_return / 100.0)
        
        portfolio_value[rebalance_date] = current_value
    
    # Preparar resultados del benchmark
    if benchmark_col in precios_varios_df.columns:
        benchmark_series = precios_varios_df[benchmark_col].dropna()
        # Normalizar benchmark a 100 en la fecha de inicio
        if fecha_inicio in benchmark_series.index:
            benchmark_value = (benchmark_series / benchmark_series.loc[fecha_inicio]) * initial_value
            benchmark_returns = benchmark_value.pct_change() * 100
        else:
            # Encontrar fecha más cercana
            closest_date = benchmark_series.index[benchmark_series.index >= fecha_inicio][0] if len(benchmark_series.index[benchmark_series.index >= fecha_inicio]) > 0 else benchmark_series.index[0]
            benchmark_value = (benchmark_series / benchmark_series.loc[closest_date]) * initial_value
            benchmark_returns = benchmark_value.pct_change() * 100
    else:
        benchmark_value = pd.Series(dtype=float)
        benchmark_returns = pd.Series(dtype=float)
    
    # Calcular retornos de la cartera
    portfolio_returns = portfolio_value.pct_change() * 100
    
    # Preparar DataFrame de posiciones
    positions_df = pd.DataFrame(portfolio_holdings).T
    
    return {
        'portfolio_value': portfolio_value,
        'benchmark_value': benchmark_value,
        'portfolio_returns': portfolio_returns,
        'benchmark_returns': benchmark_returns,
        'rebalance_dates': rebalance_dates.tolist(),
        'positions': positions_df
    }


def calculate_performance_metrics(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series
) -> Dict:
    """
    Calcula métricas de rendimiento de la cartera vs benchmark.
    
    Parameters
    ----------
    portfolio_returns : pd.Series
        Retornos de la cartera
    benchmark_returns : pd.Series
        Retornos del benchmark
    
    Returns
    -------
    dict
        Diccionario con métricas de rendimiento
    """
    # Alinear series
    common_dates = portfolio_returns.index.intersection(benchmark_returns.index)
    port_ret_aligned = portfolio_returns.loc[common_dates]
    bench_ret_aligned = benchmark_returns.loc[common_dates]
    
    # Eliminar NaN
    valid_mask = port_ret_aligned.notna() & bench_ret_aligned.notna()
    port_ret_clean = port_ret_aligned[valid_mask]
    bench_ret_clean = bench_ret_aligned[valid_mask]
    
    if len(port_ret_clean) == 0:
        return {}
    
    # Métricas básicas
    portfolio_total_return = (port_ret_clean + 1).prod() - 1
    benchmark_total_return = (bench_ret_clean + 1).prod() - 1
    
    portfolio_annual_return = ((1 + portfolio_total_return) ** (252 / len(port_ret_clean))) - 1
    benchmark_annual_return = ((1 + benchmark_total_return) ** (252 / len(bench_ret_clean))) - 1
    
    portfolio_volatility = port_ret_clean.std() * np.sqrt(252)
    benchmark_volatility = bench_ret_clean.std() * np.sqrt(252)
    
    # Sharpe ratio (asumiendo risk-free = 0 por simplicidad)
    portfolio_sharpe = portfolio_annual_return / portfolio_volatility if portfolio_volatility > 0 else 0
    benchmark_sharpe = benchmark_annual_return / benchmark_volatility if benchmark_volatility > 0 else 0
    
    # Tracking error
    active_returns = port_ret_clean - bench_ret_clean
    tracking_error = active_returns.std() * np.sqrt(252)
    
    # Information ratio
    active_return_mean = active_returns.mean() * 252
    information_ratio = active_return_mean / tracking_error if tracking_error > 0 else 0
    
    # Beta y Alpha
    if len(port_ret_clean) > 1:
        covariance = np.cov(port_ret_clean, bench_ret_clean)[0, 1]
        bench_variance = bench_ret_clean.var()
        beta = covariance / bench_variance if bench_variance > 0 else 1
        alpha = portfolio_annual_return - (beta * benchmark_annual_return)
    else:
        beta = 1.0
        alpha = 0.0
    
    # Maximum drawdown
    cumulative_port = (1 + port_ret_clean).cumprod()
    cumulative_bench = (1 + bench_ret_clean).cumprod()
    
    running_max_port = cumulative_port.cummax()
    running_max_bench = cumulative_bench.cummax()
    
    drawdown_port = (cumulative_port - running_max_port) / running_max_port
    drawdown_bench = (cumulative_bench - running_max_bench) / running_max_bench
    
    max_drawdown_port = drawdown_port.min()
    max_drawdown_bench = drawdown_bench.min()
    
    return {
        'portfolio_total_return': portfolio_total_return * 100,
        'benchmark_total_return': benchmark_total_return * 100,
        'portfolio_annual_return': portfolio_annual_return * 100,
        'benchmark_annual_return': benchmark_annual_return * 100,
        'portfolio_volatility': portfolio_volatility * 100,
        'benchmark_volatility': benchmark_volatility * 100,
        'portfolio_sharpe': portfolio_sharpe,
        'benchmark_sharpe': benchmark_sharpe,
        'tracking_error': tracking_error * 100,
        'information_ratio': information_ratio,
        'beta': beta,
        'alpha': alpha * 100,
        'max_drawdown_portfolio': max_drawdown_port * 100,
        'max_drawdown_benchmark': max_drawdown_bench * 100
    }

