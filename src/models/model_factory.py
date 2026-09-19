"""
Model Factory for Capacity-Specific Models
Routes to appropriate model class based on capacity.
"""

from src.models.prabayar import PrabayarModel
from src.models.prabayar_450 import Prabayar450Model
from src.models.prabayar_900 import Prabayar900Model
from src.models.prabayar_1300 import Prabayar1300Model
from src.models.prabayar_2200 import Prabayar2200Model
from src.models.prabayar_3500 import Prabayar3500Model


CAPACITY_MODEL_MAP = {
    "450": Prabayar450Model,
    "900": Prabayar900Model,
    "1300": Prabayar1300Model,
    "2200": Prabayar2200Model,
    "3500": Prabayar3500Model,
    "base": PrabayarModel,  # Original unified model
}


def create_model_for_capacity(
    capacity: str,
    layer_sizes: list[int],
    **kwargs
) -> PrabayarModel:
    """
    Create model instance for specific capacity.
    
    Args:
        capacity: Capacity category ("450", "900", "1300", "2200", "3500", "base")
        layer_sizes: Network architecture
        **kwargs: Additional model parameters (clip_value, l2_lambda, etc.)
    
    Returns:
        Capacity-specific model instance
    
    Example:
        model = create_model_for_capacity("900", [100, 64, 32, 1], clip_value=5.0)
    """
    ModelClass = CAPACITY_MODEL_MAP.get(capacity, PrabayarModel)
    return ModelClass(layer_sizes=layer_sizes, **kwargs)


def load_model_for_capacity(path: str, capacity: str = None):
    """
    Load model from file with automatic capacity detection.
    
    Args:
        path: Path to model JSON file
        capacity: Optional capacity hint
    
    Returns:
        (model, metadata) tuple
    
    Example:
        model, metadata = load_model_for_capacity("results/prabayar_900/models/model.json")
    """
    import json
    
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Detect capacity from model_class or metadata
    model_class_name = data.get("model_class", "PrabayarModel")
    detected_capacity = data.get("capacity", "")
    
    # Extract capacity from metadata if available
    if capacity:
        ModelClass = CAPACITY_MODEL_MAP.get(capacity, PrabayarModel)
    elif detected_capacity:
        # Extract numeric part from "450VA" -> "450"
        capacity_key = detected_capacity.replace("VA", "")
        ModelClass = CAPACITY_MODEL_MAP.get(capacity_key, PrabayarModel)
    else:
        # Fallback based on class name
        if "450" in model_class_name:
            ModelClass = Prabayar450Model
        elif "900" in model_class_name:
            ModelClass = Prabayar900Model
        elif "1300" in model_class_name:
            ModelClass = Prabayar1300Model
        elif "2200" in model_class_name:
            ModelClass = Prabayar2200Model
        elif "3500" in model_class_name:
            ModelClass = Prabayar3500Model
        else:
            ModelClass = PrabayarModel
    
    return ModelClass.load(path)


def get_available_capacities() -> list[str]:
    """Return list of available capacity categories."""
    return ["450", "900", "1300", "2200", "3500"]


def get_model_class_for_capacity(capacity: str):
    """
    Get model class for specific capacity without instantiating.
    
    Args:
        capacity: Capacity category
    
    Returns:
        Model class
    """
    return CAPACITY_MODEL_MAP.get(capacity, PrabayarModel)


def recommend_capacity_model(capacity_value: int) -> str:
    """
    Recommend appropriate capacity model based on actual capacity value.
    
    Args:
        capacity_value: Actual capacity in VA (e.g., 900, 1300, etc.)
    
    Returns:
        Recommended capacity key for model selection
    
    Example:
        capacity_key = recommend_capacity_model(1200)  # Returns "1300"
    """
    if capacity_value <= 450:
        return "450"
    elif capacity_value <= 900:
        return "900"
    elif capacity_value <= 1300:
        return "1300"
    elif capacity_value <= 2200:
        return "2200"
    else:
        return "2200"  # Use 2200 as fallback for larger capacities
