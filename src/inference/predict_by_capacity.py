"""
Inference Module for Capacity-Specific Prediction

This module routes prediction requests to the appropriate capacity-specific model.
"""

import os
import numpy as np
from src.models.model_factory import load_model_for_capacity, recommend_capacity_model
from src.pipeline.preprocessing import (
    transform_standard_scaler,
    inverse_transform_target,
)
from src.config.config import config


def predict_by_capacity(
    input_data: dict,
    capacity: str = None,
    use_auto_routing: bool = True
):
    """
    Predict using capacity-specific model.
    
    Args:
        input_data: Dictionary containing preprocessed features
        capacity: Capacity category ("450", "900", "1300", "2200", "3500")
                 If None, will auto-detect from input_data
        use_auto_routing: If True, automatically select model based on capacity value
    
    Returns:
        dict with prediction and metadata
    
    Example:
        result = predict_by_capacity(
            input_data={
                "Daya_Listrik_Rumah_VA": 900,
                "Jumlah_Anggota_Keluarga": 4,
                # ... other features
            }
        )
    """
    
    # Auto-detect capacity if not provided
    if capacity is None and use_auto_routing:
        capacity_value = input_data.get("Daya_Listrik_Rumah_VA", 900)
        
        # Handle special cases
        if isinstance(capacity_value, str):
            if capacity_value == "Tidak tahu":
                capacity_value = 900  # Default to 900VA
            elif capacity_value == "> 5500":
                capacity_value = 7700
            else:
                capacity_value = int(capacity_value)
        
        capacity = recommend_capacity_model(capacity_value)
    
    if capacity is None:
        capacity = "900"  # Final fallback
    
    # Get model path
    if capacity not in config["capacity_configs"]:
        raise ValueError(f"Capacity '{capacity}' not found. Available: {list(config['capacity_configs'].keys())}")
    
    cfg = config["capacity_configs"][capacity]
    model_path = cfg["model_path"]
    
    # Check if model exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model for {capacity}VA not found at {model_path}. "
            f"Please train the model first using train_single_capacity('{capacity}')"
        )
    
    # Load model
    model, metadata = load_model_for_capacity(model_path, capacity)
    
    # Extract feature columns and scalers from metadata
    feature_columns = metadata.get("feature_columns", [])
    x_scaler = metadata.get("x_scaler")
    y_scaler = metadata.get("y_scaler")
    
    # Prepare input features in correct order
    x_raw = [input_data.get(col, 0.0) for col in feature_columns]
    
    # Scale features
    x_scaled = transform_standard_scaler([x_raw], x_scaler)[0]
    
    # Predict
    x_input = np.array(x_scaled, dtype=np.float32).reshape(1, -1)
    prediction_scaled = model.predict(x_input)[0][0]
    
    # Inverse transform to original scale
    prediction_original = inverse_transform_target(float(prediction_scaled), y_scaler)
    
    return {
        "prediction": prediction_original,
        "capacity_used": f"{capacity}VA",
        "model_path": model_path,
        "metadata": {
            "model_type": metadata.get("model_type"),
            "training_samples": metadata.get("training_samples"),
            "architecture": metadata.get("layer_sizes"),
        }
    }


def load_all_capacity_models():
    """
    Load all trained capacity-specific models.
    
    Returns:
        dict mapping capacity to (model, metadata)
    """
    models = {}
    
    for capacity in config["capacity_configs"].keys():
        cfg = config["capacity_configs"][capacity]
        model_path = cfg["model_path"]
        
        if os.path.exists(model_path):
            try:
                model, metadata = load_model_for_capacity(model_path, capacity)
                models[capacity] = (model, metadata)
            except Exception as e:
                print(f"Warning: Failed to load {capacity}VA model: {e}")
        else:
            print(f"Warning: Model for {capacity}VA not found at {model_path}")
    
    return models


def predict_with_all_models(input_data: dict):
    """
    Make prediction using all available capacity models.
    Useful for comparison or ensemble prediction.
    
    Args:
        input_data: Dictionary containing preprocessed features
    
    Returns:
        dict mapping capacity to prediction result
    """
    results = {}
    
    for capacity in config["capacity_configs"].keys():
        try:
            result = predict_by_capacity(
                input_data=input_data,
                capacity=capacity,
                use_auto_routing=False
            )
            results[capacity] = result
        except Exception as e:
            results[capacity] = {"error": str(e)}
    
    return results


def get_model_recommendation(capacity_value: int) -> dict:
    """
    Get recommendation for which model to use based on capacity value.
    
    Args:
        capacity_value: Actual capacity in VA
    
    Returns:
        dict with recommendation details
    """
    recommended = recommend_capacity_model(capacity_value)
    cfg = config["capacity_configs"][recommended]
    
    return {
        "input_capacity": capacity_value,
        "recommended_model": f"{recommended}VA",
        "model_path": cfg["model_path"],
        "dataset_samples": "See training metrics",
        "reason": f"Capacity {capacity_value}VA is closest to {recommended}VA category"
    }


if __name__ == "__main__":
    # Example usage
    sample_input = {
        "Daya_Listrik_Rumah_VA": 900,
        "Jumlah_Anggota_Keluarga": 4,
        "Nominal_Token_Terakhir_Rp": 100000,
        # ... add all required features
    }
    
    # Single prediction
    result = predict_by_capacity(sample_input)
    print(f"Prediction: {result['prediction']:.2f} hari")
    print(f"Model used: {result['capacity_used']}")
