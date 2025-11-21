"""
Módulo de análisis de carteras de renta fija.

Este módulo contiene funciones para:
- Análisis de divisas, tipos de bonos, sectores
- Análisis de ratings y riesgo de crédito
- Análisis de liquidez
- Estadísticas descriptivas
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple


# Definiciones de ratings
IG_RATINGS = ['AAA', 'AA+', 'AA', 'AA-', 'A+', 'A', 'A-', 'BBB+', 'BBB', 'BBB-']
HY_RATINGS = ['BB+', 'BB', 'BB-', 'B+', 'B', 'B-', 'CCC+', 'CCC', 'CCC-', 'CC', 'C', 'D']


def analyze_currencies(df: pd.DataFrame, print_results: bool = True) -> Dict:
    """
    Analiza la distribución de divisas en el universo.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con información de bonos (debe incluir columna 'Ccy')
    print_results : bool, default True
        Si es True, imprime los resultados
    
    Returns
    -------
    dict
        Diccionario con estadísticas de divisas
    """
    if 'Ccy' not in df.columns:
        raise ValueError("DataFrame debe incluir columna 'Ccy'")
    
    divisas = df['Ccy'].unique()
    divisas_count = df['Ccy'].value_counts()
    
    results = {
        'unique_currencies': list(divisas),
        'num_currencies': len(divisas),
        'distribution': divisas_count.to_dict(),
        'percentages': (divisas_count / len(df) * 100).to_dict()
    }
    
    if print_results:
        print("="*60)
        print("ANÁLISIS DE DIVISAS")
        print("="*60)
        print(f"\nDivisas presentes en el universo: {', '.join(map(str, divisas))}")
        print(f"\nDistribución por divisa:")
        for ccy, count in divisas_count.items():
            pct = (count / len(df)) * 100
            print(f"  {ccy}: {count} bonos ({pct:.1f}%)")
        
        print(f"\nConclusión:")
        if len(divisas) == 1:
            print(f"  El universo está compuesto exclusivamente por bonos en {divisas[0]}.")
            print(f"  No hay exposición a riesgo cambiario, pero tampoco hay diversificación en divisas.")
        else:
            print(f"  El universo tiene exposición a {len(divisas)} divisas diferentes.")
            print(f"  Esto añade riesgo cambiario pero también diversificación.")
        print("="*60 + "\n")
    
    return results


def analyze_bond_types(df: pd.DataFrame, print_results: bool = True) -> Dict:
    """
    Analiza los tipos de bonos en el universo.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con información de bonos
    print_results : bool, default True
        Si es True, imprime los resultados
    
    Returns
    -------
    dict
        Diccionario con estadísticas de tipos de bonos
    """
    results = {}
    
    # Tipos de cupón
    if 'Coupon Type' in df.columns:
        tipos_bonos = df['Coupon Type'].value_counts()
        results['coupon_types'] = tipos_bonos.to_dict()
        results['coupon_types_pct'] = (tipos_bonos / len(df) * 100).to_dict()
    
    # Opcionalidad (Callable)
    if 'Callable' in df.columns:
        callable_count = df['Callable'].value_counts()
        results['callable'] = callable_count.to_dict()
        results['callable_pct'] = (callable_count / len(df) * 100).to_dict()
    
    # Seniority
    if 'Seniority' in df.columns:
        seniority_count = df['Seniority'].value_counts()
        results['seniority'] = seniority_count.head(10).to_dict()
        results['seniority_pct'] = (seniority_count.head(10) / len(df) * 100).to_dict()
    
    # Bonos perpetuos
    if 'Maturity' in df.columns:
        perpetuos = df[df['Maturity'].isna() | (df['Maturity'].astype(str).str.strip() == '')]
        results['perpetuos_count'] = len(perpetuos)
        results['perpetuos_pct'] = len(perpetuos) / len(df) * 100
    
    if print_results:
        print("="*60)
        print("ANÁLISIS DE TIPOS DE BONO")
        print("="*60)
        
        if 'coupon_types' in results:
            print(f"\n1. Tipo de Cupón (Coupon Type):")
            tipos_bonos = pd.Series(results['coupon_types'])
            for tipo, count in tipos_bonos.items():
                pct = results['coupon_types_pct'][tipo]
                print(f"  {tipo}: {count} bonos ({pct:.1f}%)")
        
        if 'callable' in results:
            print(f"\n2. Opcionalidad (Callable):")
            callable_count = pd.Series(results['callable'])
            for callable, count in callable_count.items():
                pct = results['callable_pct'][callable]
                estado = "Sí (Callable)" if callable else "No (No Callable)"
                print(f"  {estado}: {count} bonos ({pct:.1f}%)")
        
        if 'seniority' in results:
            print(f"\n3. Prelación (Seniority):")
            seniority_count = pd.Series(results['seniority'])
            for seniority, count in seniority_count.items():
                pct = results['seniority_pct'][seniority]
                print(f"  {seniority}: {count} bonos ({pct:.1f}%)")
        
        if 'perpetuos_count' in results:
            print(f"\n4. Bonos Perpetuos:")
            print(f"  Número de bonos perpetuos: {results['perpetuos_count']} ({results['perpetuos_pct']:.1f}%)")
            if results['perpetuos_count'] > 0:
                print(f"  Nota: Para estos bonos, se usa Next Call Date como fecha de vencimiento según el enunciado.")
        
        print("="*60 + "\n")
    
    return results


def analyze_ratings(df: pd.DataFrame, print_results: bool = True) -> Dict:
    """
    Analiza los ratings y el riesgo de crédito del universo.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con información de bonos (debe incluir columna 'Rating')
    print_results : bool, default True
        Si es True, imprime los resultados
    
    Returns
    -------
    dict
        Diccionario con estadísticas de ratings
    """
    if 'Rating' not in df.columns:
        raise ValueError("DataFrame debe incluir columna 'Rating'")
    
    ratings = df['Rating'].value_counts(normalize=True) * 100
    pd_1yr = df['PD 1YR'].mean() if 'PD 1YR' in df.columns else None
    
    # Clasificar en IG/HY
    ig_count = df[df['Rating'].isin(IG_RATINGS)].shape[0]
    hy_count = df[df['Rating'].isin(HY_RATINGS)].shape[0]
    nr_count = df[df['Rating'] == 'NR'].shape[0]
    
    results = {
        'ratings_distribution': ratings.to_dict(),
        'ig_count': ig_count,
        'ig_pct': ig_count / len(df) * 100,
        'hy_count': hy_count,
        'hy_pct': hy_count / len(df) * 100,
        'nr_count': nr_count,
        'nr_pct': nr_count / len(df) * 100,
        'pd_1yr_mean': pd_1yr
    }
    
    if print_results:
        print("="*60)
        print("ANÁLISIS DE RATINGS (RIESGO DE CRÉDITO)")
        print("="*60)
        
        print(f"\n1. Distribución por Rating:")
        for rating, pct in ratings.head(15).items():
            count = len(df[df['Rating'] == rating])
            print(f"  {rating}: {count} bonos ({pct:.1f}%)")
        
        print(f"\n2. Clasificación Investment Grade vs High Yield:")
        print(f"  Investment Grade (IG): {ig_count} bonos ({results['ig_pct']:.1f}%)")
        print(f"  High Yield (HY): {hy_count} bonos ({results['hy_pct']:.1f}%)")
        print(f"  No Rated (NR): {nr_count} bonos ({results['nr_pct']:.1f}%)")
        
        if pd_1yr is not None:
            print(f"\n3. Probabilidad de Default (PD 1YR):")
            print(f"  PD 1YR media: {pd_1yr:.4f} ({pd_1yr*100:.2f}%)")
            
            if 'PD 1YR' in df.columns:
                pd_hy = df[df['Rating'].isin(HY_RATINGS)]['PD 1YR'].mean()
                pd_ig = df[df['Rating'].isin(IG_RATINGS)]['PD 1YR'].mean()
                if not pd.isna(pd_hy):
                    print(f"  PD 1YR media HY: {pd_hy:.4f} ({pd_hy*100:.2f}%)")
                if not pd.isna(pd_ig):
                    print(f"  PD 1YR media IG: {pd_ig:.4f} ({pd_ig*100:.2f}%)")
        
        print(f"\nConclusión:")
        print(f"  - {'Alta' if results['ig_pct'] > 80 else 'Baja'} proporción de Investment Grade")
        print(f"  - {'Alta' if results['hy_pct'] > 10 else 'Baja'} exposición a High Yield")
        risk_level = 'BAJO' if results['ig_pct'] > 80 and (pd_1yr is None or pd_1yr < 0.001) else 'MODERADO' if pd_1yr is None or pd_1yr < 0.01 else 'ALTO'
        print(f"  - Riesgo de crédito: {risk_level}")
        print("="*60 + "\n")
    
    return results


def analyze_sectors(df: pd.DataFrame, print_results: bool = True) -> Dict:
    """
    Analiza la distribución de sectores y emisores.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con información de bonos (debe incluir columna 'Industry Sector')
    print_results : bool, default True
        Si es True, imprime los resultados
    
    Returns
    -------
    dict
        Diccionario con estadísticas de sectores
    """
    if 'Industry Sector' not in df.columns:
        raise ValueError("DataFrame debe incluir columna 'Industry Sector'")
    
    sectores = df['Industry Sector'].value_counts(normalize=True) * 100
    emisores = df['Issuer'].nunique() if 'Issuer' in df.columns else 0
    emisores_top = df['Issuer'].value_counts().head(10) if 'Issuer' in df.columns else pd.Series()
    
    # Calcular índice de Herfindahl para concentración sectorial
    herfindahl_sector = ((sectores / 100) ** 2).sum()
    
    top10_emisores_pct = (emisores_top.sum() / len(df) * 100) if len(emisores_top) > 0 else 0
    
    results = {
        'sectors_distribution': sectores.to_dict(),
        'num_issuers': emisores,
        'top_issuers': emisores_top.to_dict(),
        'herfindahl_index': herfindahl_sector,
        'top10_issuers_pct': top10_emisores_pct
    }
    
    if print_results:
        print("="*60)
        print("ANÁLISIS DE SECTORES Y EMISORES")
        print("="*60)
        
        print(f"\n1. Distribución por Sectores (Industry Sector):")
        for sector, pct in sectores.head(10).items():
            count = len(df[df['Industry Sector'] == sector])
            print(f"  {sector}: {count} bonos ({pct:.1f}%)")
        
        if emisores > 0:
            print(f"\n2. Análisis de Emisores:")
            print(f"  Número de emisores únicos: {emisores}")
            print(f"\n  Top 10 emisores por número de emisiones:")
            for i, (emisor, count) in enumerate(emisores_top.items(), 1):
                pct = (count / len(df)) * 100
                print(f"    {i}. {emisor}: {count} emisiones ({pct:.1f}%)")
        
        print(f"\n3. Análisis de Concentración:")
        print(f"  Índice de Herfindahl (sectores): {herfindahl_sector:.3f}")
        if herfindahl_sector > 0.15:
            print(f"    → Concentración ALTA (índice > 0.15)")
        else:
            print(f"    → Concentración BAJA (índice ≤ 0.15)")
        
        if top10_emisores_pct > 0:
            print(f"  Top 10 emisores representan: {top10_emisores_pct:.1f}% del universo")
        
        print(f"\nConclusión:")
        print(f"  - {'Alta' if herfindahl_sector > 0.15 else 'Baja'} concentración sectorial")
        if top10_emisores_pct > 0:
            print(f"  - {'Alta' if top10_emisores_pct > 20 else 'Baja'} concentración en emisores")
        print("="*60 + "\n")
    
    return results


def analyze_liquidity(df: pd.DataFrame, print_results: bool = True) -> Dict:
    """
    Analiza el riesgo de liquidez del universo.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con información de bonos (debe incluir columna 'Bid-Ask Spread' y 'Outstanding Amount')
    print_results : bool, default True
        Si es True, imprime los resultados
    
    Returns
    -------
    dict
        Diccionario con estadísticas de liquidez
    """
    results = {}
    
    # Análisis de Bid-Ask Spread
    if 'Bid-Ask Spread' in df.columns:
        spread_series = pd.to_numeric(df['Bid-Ask Spread'], errors='coerce')
        results['spread_mean'] = spread_series.mean()
        results['spread_median'] = spread_series.median()
        results['spread_std'] = spread_series.std()
        results['spread_min'] = spread_series.min()
        results['spread_max'] = spread_series.max()
        
        # Clasificación de liquidez
        liquidez_alta = (spread_series <= 0.2).sum()
        liquidez_media = ((spread_series > 0.2) & (spread_series <= 0.5)).sum()
        liquidez_baja = (spread_series > 0.5).sum()
        
        results['liquidez_alta_count'] = liquidez_alta
        results['liquidez_alta_pct'] = liquidez_alta / len(df) * 100
        results['liquidez_media_count'] = liquidez_media
        results['liquidez_media_pct'] = liquidez_media / len(df) * 100
        results['liquidez_baja_count'] = liquidez_baja
        results['liquidez_baja_pct'] = liquidez_baja / len(df) * 100
    
    # Análisis de Outstanding Amount
    if 'Outstanding Amount' in df.columns:
        outstanding = pd.to_numeric(df['Outstanding Amount'], errors='coerce')
        results['outstanding_mean'] = outstanding.mean()
        results['outstanding_median'] = outstanding.median()
        results['outstanding_min'] = outstanding.min()
        results['outstanding_max'] = outstanding.max()
        
        # Emisiones grandes (>500M)
        emisiones_grandes = (outstanding > 500000000).sum()
        results['emisiones_grandes_count'] = emisiones_grandes
        results['emisiones_grandes_pct'] = emisiones_grandes / len(df) * 100
        
        # Correlación entre spread y outstanding
        if 'Bid-Ask Spread' in df.columns:
            corr = df[['Bid-Ask Spread', 'Outstanding Amount']].corr().iloc[0, 1]
            results['correlation_spread_outstanding'] = corr
    
    if print_results:
        print("="*60)
        print("ANÁLISIS DE RIESGO DE LIQUIDEZ")
        print("="*60)
        
        if 'spread_mean' in results:
            print(f"\n1. Horquillas Bid-Ask Spread:")
            print(f"  Media: {results['spread_mean']:.4f}")
            print(f"  Mediana: {results['spread_median']:.4f}")
            print(f"  Desviación estándar: {results['spread_std']:.4f}")
            print(f"  Mínimo: {results['spread_min']:.4f}")
            print(f"  Máximo: {results['spread_max']:.4f}")
            print(f"\n  Clasificación:")
            print(f"    Alta liquidez (spread ≤ 0.2): {results['liquidez_alta_count']} bonos ({results['liquidez_alta_pct']:.1f}%)")
            print(f"    Liquidez media (0.2 < spread ≤ 0.5): {results['liquidez_media_count']} bonos ({results['liquidez_media_pct']:.1f}%)")
            print(f"    Baja liquidez (spread > 0.5): {results['liquidez_baja_count']} bonos ({results['liquidez_baja_pct']:.1f}%)")
        
        if 'outstanding_mean' in results:
            print(f"\n2. Nominal Vivo (Outstanding Amount):")
            print(f"  Media: {results['outstanding_mean']:,.0f} €")
            print(f"  Mediana: {results['outstanding_median']:,.0f} €")
            print(f"  Mínimo: {results['outstanding_min']:,.0f} €")
            print(f"  Máximo: {results['outstanding_max']:,.0f} €")
            print(f"  Emisiones > 500M: {results['emisiones_grandes_count']} bonos ({results['emisiones_grandes_pct']:.1f}%)")
        
        if 'correlation_spread_outstanding' in results:
            print(f"\n3. Relación entre Liquidez y Tamaño:")
            corr = results['correlation_spread_outstanding']
            print(f"  Correlación Bid-Ask Spread vs Outstanding Amount: {corr:.3f}")
            if corr < -0.2:
                print(f"    → Correlación negativa: mayor tamaño = mejor liquidez")
            elif corr > 0.2:
                print(f"    → Correlación positiva: mayor tamaño = peor liquidez")
            else:
                print(f"    → Correlación débil: no hay relación clara")
        
        print("="*60 + "\n")
    
    return results


