"""
Analysis module initialization
"""

from src.analysis.compare_capacities import (
    load_capacity_metrics,
    load_all_metrics,
    generate_comparison_report,
    plot_metrics_comparison,
)

__all__ = [
    "load_capacity_metrics",
    "load_all_metrics",
    "generate_comparison_report",
    "plot_metrics_comparison",
]
