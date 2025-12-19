"""
Evaluation module for Choquet aggregation results.
"""

from .aggregate_results import aggregate_multiple_results, compute_statistics
from .plot_results import create_boxplots, create_comparison_plots

__all__ = [
    'aggregate_multiple_results',
    'compute_statistics',
    'create_boxplots',
    'create_comparison_plots'
]
