"""
Paquete de análisis de carteras de renta fija.

Este paquete contiene módulos para:
- utils: Utilidades y carga de datos
- valuation: Valoración de bonos
- metrics: Métricas (YTM, Duración, Convexidad)
- analysis: Análisis de carteras
"""

__version__ = "1.0.0"

from . import utils
from . import valuation
from . import metrics
from . import analysis

__all__ = ['utils', 'valuation', 'metrics', 'analysis']


