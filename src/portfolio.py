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
from typing import Dict, Tuple, Optional, List
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

try:
    from utils import get_effective_maturity
except ImportError:
    get_effective_maturity = None


def get_alive_bonds_at_date(
    precios_df: pd.DataFrame,
    universo_df: pd.DataFrame,
    fecha: datetime
) -> pd.Index:
    """
    Obtiene los ISINs de bonos vivos en una fecha determinada.
    
    Un bono está vivo si tiene precio válido en esa fecha y su Effective Maturity
    es posterior a la fecha (considerando bonos callable y perpetuos).
    
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
        bonos_con_precio = pd.Index([str(x) for x in bonos_con_precio])
    except Exception:
        return pd.Index([])
    
    universo_copy = universo_df.copy()
    if 'ISIN' not in universo_copy.columns:
        universo_copy['ISIN'] = universo_copy.index.astype(str)
    
    # Asegurar que las fechas estén en formato datetime
    if 'Maturity' in universo_copy.columns:
        if not pd.api.types.is_datetime64_any_dtype(universo_copy['Maturity']):
            universo_copy['Maturity'] = pd.to_datetime(universo_copy['Maturity'], format='%d/%m/%Y', errors='coerce')
    if 'Next Call Date' in universo_copy.columns:
        if not pd.api.types.is_datetime64_any_dtype(universo_copy['Next Call Date']):
            universo_copy['Next Call Date'] = pd.to_datetime(universo_copy['Next Call Date'], format='%d/%m/%Y', errors='coerce')
    
    # Importar get_effective_maturity si no está disponible
    try:
        from utils import get_effective_maturity
    except ImportError:
        # Si no se puede importar, usar una función simple
        def get_effective_maturity(row):
            call_flag = str(row.get('Callable', 'N')).upper()
            call_date = row.get('Next Call Date')
            maturity = row.get('Maturity')
            
            if call_flag == 'Y' and pd.notna(call_date):
                return call_date if isinstance(call_date, pd.Timestamp) else pd.to_datetime(call_date, errors='coerce')
            elif pd.notna(maturity):
                return maturity if isinstance(maturity, pd.Timestamp) else pd.to_datetime(maturity, errors='coerce')
            return None
    
    if 'Maturity' in universo_copy.columns:
        universo_copy['Effective Maturity'] = universo_copy.apply(
            lambda row: get_effective_maturity(row), axis=1
        )
        # Filtrar solo bonos con Effective Maturity válida y mayor que fecha
        mask = universo_copy['Effective Maturity'].notna()
        mask = mask & (universo_copy['Effective Maturity'] > fecha)
        bonos_vivos = universo_copy[mask].copy()
        
        if 'ISIN' in bonos_vivos.columns:
            bonos_vivos_isins = pd.Index([str(x) for x in bonos_vivos['ISIN'].dropna().unique()])
        else:
            bonos_vivos_isins = pd.Index([str(x) for x in bonos_vivos.index])
    else:
        bonos_vivos_isins = bonos_con_precio
    
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
        [DEPRECATED] Parámetro mantenido por compatibilidad, ya no se usa.
        El cálculo usa directamente coupon_rate anual y días mantenidos.
    
    Returns
    -------
    float
        Retorno total (en %)
    
    Notes
    -----
    El cálculo de cupones es proporcional: coupon_rate ya es anual (%), 
    days_held/365.25 da la fracción del año. No se multiplica por frequency
    para evitar doble conteo.
    """
    if precio_inicial <= 0 or pd.isna(precio_inicial) or pd.isna(precio_final):
        return 0.0
    
    price_return = (precio_final - precio_inicial) / precio_inicial
    coupons_received = (coupon_rate / 100.0) * (days_held / 365.25)
    total_return = price_return + coupons_received
    
    return total_return * 100


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
    if fecha_fin is None:
        fechas_disponibles = pd.to_datetime(precios_df.columns, format='%d/%m/%Y', errors='coerce')
        fecha_fin = fechas_disponibles.max()
    
    fecha_cols = pd.to_datetime(precios_df.columns, format='%d/%m/%Y', errors='coerce')
    precios_df.columns = fecha_cols
    
    rebalance_dates = pd.date_range(start=fecha_inicio, end=fecha_fin, freq=rebalance_frequency)
    rebalance_dates = rebalance_dates[rebalance_dates <= fecha_fin]
    
    portfolio_value = pd.Series(dtype=float)
    portfolio_holdings = {}
    initial_value = 100.0
    current_value = initial_value
    
    portfolio_value[rebalance_dates[0]] = initial_value
    if 'ISIN' in universo_df.columns:
        try:
            universo_indexed = universo_df.set_index('ISIN', drop=False)
        except:
            universo_indexed = universo_df.copy()
    else:
        universo_indexed = universo_df.copy()
    for i, rebalance_date in enumerate(rebalance_dates):
        alive_isins = get_alive_bonds_at_date(precios_df, universo_df, rebalance_date)
        
        if len(alive_isins) == 0:
            portfolio_value[rebalance_date] = current_value
            continue
        
        try:
            precios_fecha = precios_df.loc[alive_isins, rebalance_date]
        except KeyError:
            available_dates = [d for d in precios_df.columns if pd.notna(d) and d <= rebalance_date]
            if len(available_dates) == 0:
                portfolio_value[rebalance_date] = current_value
                continue
            closest_date = max(available_dates)
            precios_fecha = precios_df.loc[alive_isins, closest_date]
        
        precios_fecha = precios_fecha.dropna()
        
        if len(precios_fecha) == 0:
            portfolio_value[rebalance_date] = current_value
            continue
        
        alive_isins = precios_fecha.index
        num_bonds = len(alive_isins)
        weight_per_bond = 1.0 / num_bonds if num_bonds > 0 else 0.0
        
        holdings = pd.Series(weight_per_bond, index=alive_isins)
        portfolio_holdings[rebalance_date] = holdings
        portfolio_value[rebalance_date] = current_value
        if i < len(rebalance_dates) - 1:
            next_rebalance = rebalance_dates[i + 1]
            
            fecha_actual_work = rebalance_date
            try:
                precios_actual = precios_df.loc[alive_isins, fecha_actual_work]
            except (KeyError, IndexError):
                available_dates = [d for d in precios_df.columns if pd.notna(d) and d <= rebalance_date]
                if len(available_dates) == 0:
                    portfolio_value[next_rebalance] = current_value
                    continue
                fecha_actual_work = max(available_dates)
                try:
                    precios_actual = precios_df.loc[alive_isins, fecha_actual_work]
                except (KeyError, IndexError):
                    portfolio_value[next_rebalance] = current_value
                    continue
            
            fecha_siguiente_work = next_rebalance
            try:
                if fecha_siguiente_work in precios_df.columns:
                    precios_siguiente = precios_df.loc[alive_isins, fecha_siguiente_work]
                else:
                    available_dates = [d for d in precios_df.columns if pd.notna(d) and fecha_actual_work < d]
                    if len(available_dates) == 0:
                        portfolio_value[next_rebalance] = current_value
                        continue
                    fecha_siguiente_work = min(available_dates, key=lambda x: abs((x - next_rebalance).days))
                    precios_siguiente = precios_df.loc[alive_isins, fecha_siguiente_work]
            except (KeyError, IndexError):
                portfolio_value[next_rebalance] = current_value
                continue
            
            precios_actual_clean = precios_actual.dropna()
            precios_siguiente_clean = precios_siguiente.dropna()
            bonos_validos = precios_actual_clean.index.intersection(precios_siguiente_clean.index)
            
            if len(bonos_validos) == 0:
                portfolio_value[next_rebalance] = current_value
                continue
            
            portfolio_return = 0.0
            days_held = (fecha_siguiente_work - fecha_actual_work).days
            if days_held <= 0:
                portfolio_value[next_rebalance] = current_value
                continue
            
            weight_per_bond_valid = 1.0 / len(bonos_validos) if len(bonos_validos) > 0 else 0.0
            
            for isin in bonos_validos:
                bond_info = None
                coupon_rate = 0
                
                if isinstance(universo_indexed.index, pd.Index):
                    if isin in universo_indexed.index:
                        bond_info = universo_indexed.loc[isin]
                    elif 'ISIN' in universo_indexed.columns:
                        mask = universo_indexed['ISIN'].astype(str) == str(isin)
                        if mask.any():
                            bond_info = universo_indexed[mask].iloc[0]
                
                if bond_info is not None:
                    if isinstance(bond_info, pd.Series):
                        coupon_rate = bond_info.get('Coupon', 0) if pd.notna(bond_info.get('Coupon', 0)) else 0
                    else:
                        coupon_rate = getattr(bond_info, 'Coupon', 0) if hasattr(bond_info, 'Coupon') else 0
                
                precio_inicial = float(precios_actual_clean.loc[isin])
                precio_final = float(precios_siguiente_clean.loc[isin])
                
                if precio_inicial > 0 and precio_final > 0:
                    bond_return = calculate_total_return(
                        precio_inicial, precio_final, 
                        float(coupon_rate), days_held
                    )
                    portfolio_return += weight_per_bond_valid * (bond_return / 100.0)
            
            current_value = current_value * (1 + portfolio_return)
            portfolio_value[next_rebalance] = current_value
    
    if benchmark_col in precios_varios_df.columns:
        benchmark_series = precios_varios_df[benchmark_col].dropna()
        if fecha_inicio in benchmark_series.index:
            benchmark_value = (benchmark_series / benchmark_series.loc[fecha_inicio]) * initial_value
        else:
            closest_date = benchmark_series.index[benchmark_series.index >= fecha_inicio][0] if len(benchmark_series.index[benchmark_series.index >= fecha_inicio]) > 0 else benchmark_series.index[0]
            benchmark_value = (benchmark_series / benchmark_series.loc[closest_date]) * initial_value
        
        benchmark_value_rebalance = benchmark_value.reindex(rebalance_dates, method='ffill')
        benchmark_returns = benchmark_value_rebalance.pct_change()
    else:
        benchmark_value = pd.Series(dtype=float)
        benchmark_returns = pd.Series(dtype=float)
    
    portfolio_returns = portfolio_value.pct_change()
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
    common_dates = portfolio_returns.index.intersection(benchmark_returns.index)
    port_ret_aligned = portfolio_returns.loc[common_dates]
    bench_ret_aligned = benchmark_returns.loc[common_dates]
    
    valid_mask = port_ret_aligned.notna() & bench_ret_aligned.notna()
    port_ret_clean = port_ret_aligned[valid_mask]
    bench_ret_clean = bench_ret_aligned[valid_mask]
    
    if len(port_ret_clean) == 0:
        return {}
    
    portfolio_total_return = (1 + port_ret_clean).prod() - 1
    benchmark_total_return = (1 + bench_ret_clean).prod() - 1
    if len(port_ret_clean) > 1:
        date_range = (port_ret_clean.index[-1] - port_ret_clean.index[0]).days
        if date_range > 0:
            periods_per_year_port = (len(port_ret_clean) - 1) * 365.25 / date_range
        else:
            periods_per_year_port = 12
        
        if len(bench_ret_clean) > 1:
            date_range_bench = (bench_ret_clean.index[-1] - bench_ret_clean.index[0]).days
            if date_range_bench > 0:
                periods_per_year_bench = (len(bench_ret_clean) - 1) * 365.25 / date_range_bench
            else:
                periods_per_year_bench = 252
        else:
            periods_per_year_bench = 252
        
        portfolio_annual_return = ((1 + portfolio_total_return) ** (periods_per_year_port / len(port_ret_clean))) - 1
        benchmark_annual_return = ((1 + benchmark_total_return) ** (periods_per_year_bench / len(bench_ret_clean))) - 1
    else:
        portfolio_annual_return = portfolio_total_return
        benchmark_annual_return = benchmark_total_return
    if len(port_ret_clean) > 1:
        date_range = (port_ret_clean.index[-1] - port_ret_clean.index[0]).days
        if date_range > 0:
            periods_per_year_port = (len(port_ret_clean) - 1) * 365.25 / date_range
        else:
            periods_per_year_port = 12
    else:
        periods_per_year_port = 12
    
    if len(bench_ret_clean) > 1:
        date_range_bench = (bench_ret_clean.index[-1] - bench_ret_clean.index[0]).days
        if date_range_bench > 0:
            periods_per_year_bench = (len(bench_ret_clean) - 1) * 365.25 / date_range_bench
        else:
            periods_per_year_bench = 252
    else:
        periods_per_year_bench = 252
    
    portfolio_volatility = port_ret_clean.std() * np.sqrt(periods_per_year_port)
    benchmark_volatility = bench_ret_clean.std() * np.sqrt(periods_per_year_bench)
    
    portfolio_sharpe = portfolio_annual_return / portfolio_volatility if portfolio_volatility > 0 else 0
    benchmark_sharpe = benchmark_annual_return / benchmark_volatility if benchmark_volatility > 0 else 0
    
    active_returns = port_ret_clean - bench_ret_clean
    if len(port_ret_clean) > 1:
        date_range = (port_ret_clean.index[-1] - port_ret_clean.index[0]).days
        if date_range > 0:
            periods_per_year = (len(port_ret_clean) - 1) * 365.25 / date_range
        else:
            periods_per_year = 12
    else:
        periods_per_year = 12
    
    tracking_error = active_returns.std() * np.sqrt(periods_per_year)
    active_return_mean = active_returns.mean() * periods_per_year
    information_ratio = active_return_mean / tracking_error if tracking_error > 0 else 0
    
    if len(port_ret_clean) > 1:
        covariance = np.cov(port_ret_clean, bench_ret_clean)[0, 1]
        bench_variance = bench_ret_clean.var()
        beta = covariance / bench_variance if bench_variance > 0 else 1
        alpha = portfolio_annual_return - (beta * benchmark_annual_return)
    else:
        beta = 1.0
        alpha = 0.0
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


def build_optimized_portfolio(
    universo_df: pd.DataFrame,
    fecha_analisis: datetime,
    max_bonds: int = 20,
    max_duration: float = 3.0,
    max_hy_exposure: float = 0.10,
    min_outstanding: float = 500000000,
    max_weight_per_bond: float = 0.10,
    max_weight_per_issuer: float = 0.15,
    exclude_subordinated: bool = True
) -> Dict:
    """
    Construye una cartera optimizada según el mandato del punto 6.
    
    Parameters
    ----------
    universo_df : pd.DataFrame
        DataFrame con información del universo de bonos
    fecha_analisis : datetime
        Fecha de análisis
    max_bonds : int, default 20
        Número máximo de bonos en la cartera
    max_duration : float, default 3.0
        Duración máxima de la cartera (años)
    max_hy_exposure : float, default 0.10
        Exposición máxima a High Yield (10%)
    min_outstanding : float, default 500000000
        Tamaño mínimo de emisión (500M)
    max_weight_per_bond : float, default 0.10
        Peso máximo por emisión (10%)
    max_weight_per_issuer : float, default 0.15
        Peso máximo por emisor (15%)
    exclude_subordinated : bool, default True
        Si True, excluye deuda subordinada
    
    Returns
    -------
    dict
        Diccionario con:
        - 'weights': Series con pesos de cada bono
        - 'portfolio_df': DataFrame con información de bonos seleccionados
        - 'metrics': Diccionario con métricas de la cartera
        - 'optimization_result': Resultado de la optimización
    """
    from analysis import HY_RATINGS
    
    universo = universo_df.copy()
    
    print("="*70)
    print("CONSTRUCCIÓN DE CARTERA OPTIMIZADA")
    print("="*70)
    print(f"\nFecha de análisis: {fecha_analisis}")
    print(f"Bonos en universo inicial: {len(universo)}")
    
    if 'Outstanding Amount' in universo.columns:
        outstanding = pd.to_numeric(universo['Outstanding Amount'], errors='coerce')
        universo = universo[outstanding > min_outstanding].copy()
        print(f"Después de filtrar por tamaño > {min_outstanding/1e6:.0f}M: {len(universo)} bonos")
    
    if exclude_subordinated and 'Seniority' in universo.columns:
        universo = universo[~universo['Seniority'].str.contains('Subordinated', case=False, na=False)].copy()
        print(f"Después de excluir subordinada: {len(universo)} bonos")
    
    required_cols = ['YTM', 'Modified_Duration', 'Rating', 'Issuer', 'Price']
    missing_cols = [col for col in required_cols if col not in universo.columns]
    if missing_cols:
        print(f"[WARNING] Faltan columnas necesarias: {missing_cols}")
        print("   Asegúrate de haber ejecutado las celdas anteriores (valoración, métricas)")
        return {}
    
    universo = universo[
        universo['YTM'].notna() & 
        universo['Modified_Duration'].notna() & 
        universo['Price'].notna() &
        (universo['Price'] > 0)
    ].copy()
    print(f"Después de filtrar por métricas válidas: {len(universo)} bonos")
    
    if len(universo) == 0:
        print("[ERROR] No hay bonos elegibles después de aplicar filtros")
        return {}
    
    universo['Is_HY'] = universo['Rating'].isin(HY_RATINGS) if 'Rating' in universo.columns else False
    
    universo_sorted = universo.sort_values('YTM', ascending=False)
    n_candidates = min(3 * max_bonds, len(universo_sorted))
    universo_candidates = universo_sorted.head(n_candidates).copy()
    
    print(f"Pre-filtrado: {n_candidates} candidatos seleccionados (top por YTM)")
    
    n_bonds = len(universo_candidates)
    ytm_values = universo_candidates['YTM'].values
    durations = universo_candidates['Modified_Duration'].values
    is_hy = universo_candidates['Is_HY'].values.astype(float)
    issuers = universo_candidates['Issuer'].values if 'Issuer' in universo_candidates.columns else np.arange(n_bonds)
    
    def objective(weights):
        return -np.dot(weights, ytm_values)
    
    constraints = []
    constraints.append({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
    constraints.append({
        'type': 'ineq',
        'fun': lambda w: (max_duration - 0.01) - np.dot(w, durations)
    })
    constraints.append({
        'type': 'ineq',
        'fun': lambda w: max_hy_exposure - np.dot(w, is_hy)
    })
    
    unique_issuers = np.unique(issuers)
    for issuer in unique_issuers:
        issuer_mask = (issuers == issuer)
        constraints.append({
            'type': 'ineq',
            'fun': lambda w, mask=issuer_mask: max_weight_per_issuer - np.sum(w[mask])
        })
    
    bounds = [(0, max_weight_per_bond) for _ in range(n_bonds)]
    
    top_bonds = min(max_bonds, n_bonds)
    initial_weights = np.zeros(n_bonds)
    top_indices = np.argsort(ytm_values)[-top_bonds:]
    initial_weights[top_indices] = 1.0 / top_bonds
    
    print(f"\nOptimizando cartera con {n_bonds} bonos elegibles...")
    print(f"Restricciones:")
    print(f"  - Máximo {max_bonds} bonos")
    print(f"  - Duración <= {max_duration} años")
    print(f"  - Exposición HY <= {max_hy_exposure*100:.0f}%")
    print(f"  - Peso máximo por bono: {max_weight_per_bond*100:.0f}%")
    print(f"  - Peso máximo por emisor: {max_weight_per_issuer*100:.0f}%")
    print(f"  - Total de restricciones: {len(constraints)}")
    print(f"  - Emisores únicos: {len(unique_issuers)}")
    
    # Optimizar
    try:
        import time
        start_time = time.time()
        print(f"\n[LOG] Iniciando optimización...")
        print(f"[LOG] Método: SLSQP")
        print(f"[LOG] Iteraciones máximas: 500")
        print(f"[LOG] Tolerancia: 1e-6")
        
        result = minimize(
            objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 500, 'ftol': 1e-6, 'disp': False}
        )
        
        elapsed_time = time.time() - start_time
        print(f"[LOG] Optimización completada en {elapsed_time:.2f} segundos")
        print(f"[LOG] Iteraciones realizadas: {result.nit}")
        print(f"[LOG] Evaluaciones de función: {result.nfev}")
        print(f"[LOG] Éxito: {result.success}")
        if not result.success:
            print(f"[LOG] Mensaje: {result.message}")
        
        if not result.success:
            print(f"[WARNING] Advertencia: La optimizacion no convergio: {result.message}")
            # Usar solución encontrada de todas formas si tiene sentido
            if np.sum(result.x) < 0.5:
                print("[ERROR] Error: Los pesos optimizados no suman ~1")
                print(f"[LOG] Suma de pesos: {np.sum(result.x):.6f}")
                return {}
        
        weights = result.x
        weights = weights / np.sum(weights)
        
        selected_mask = weights > 0.001
        selected_weights = weights[selected_mask]
        selected_weights = selected_weights / np.sum(selected_weights)
        
        selected_bonds = universo_candidates[selected_mask].copy()
        selected_bonds['Weight'] = selected_weights
        selected_bonds = selected_bonds.sort_values('Weight', ascending=False)
        
        print(f"\n[OK] Optimización completada")
        print(f"  Bonos seleccionados: {len(selected_bonds)}")
        print(f"  Peso total: {np.sum(selected_weights):.4f}")
        
        portfolio_duration = np.dot(selected_weights, selected_bonds['Modified_Duration'].values)
        portfolio_ytm = np.dot(selected_weights, selected_bonds['YTM'].values)
        portfolio_hy_exposure = np.dot(selected_weights, selected_bonds['Is_HY'].values)
        
        issuer_weights = selected_bonds.groupby('Issuer')['Weight'].sum()
        max_issuer_concentration = issuer_weights.max() if len(issuer_weights) > 0 else 0
        
        metrics = {
            'num_bonds': len(selected_bonds),
            'portfolio_ytm': portfolio_ytm,
            'portfolio_duration': portfolio_duration,
            'hy_exposure': portfolio_hy_exposure,
            'max_issuer_concentration': max_issuer_concentration,
            'max_bond_weight': selected_weights.max(),
            'min_bond_weight': selected_weights.min()
        }
        
        print(f"\nMétricas de la cartera:")
        print(f"  YTM promedio: {portfolio_ytm*100:.2f}%")
        print(f"  Duración: {portfolio_duration:.2f} años")
        print(f"  Exposición HY: {portfolio_hy_exposure*100:.2f}%")
        print(f"  Concentración máxima emisor: {max_issuer_concentration*100:.2f}%")
        print(f"  Peso máximo bono: {selected_weights.max()*100:.2f}%")
        print("="*70 + "\n")
        
        weights_series = pd.Series(selected_weights, index=selected_bonds.index)
        
        return {
            'weights': weights_series,
            'portfolio_df': selected_bonds,
            'metrics': metrics,
            'optimization_result': result
        }
        
    except Exception as e:
        print(f"[ERROR] Error en optimizacion: {e}")
        import traceback
        traceback.print_exc()
        return {}

