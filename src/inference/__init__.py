"""
Inference module initialization
"""

from src.inference.predict_by_capacity import (
    predict_by_capacity,
    load_all_capacity_models,
    predict_with_all_models,
    get_model_recommendation,
)

__all__ = [
    "predict_by_capacity",
    "load_all_capacity_models",
    "predict_with_all_models",
    "get_model_recommendation",
]
