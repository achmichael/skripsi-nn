from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any
from contextlib import asynccontextmanager
import numpy as np

from src.pipeline.preprocessing import transform_standard_scaler, inverse_transform_target
from src.models.pascabayar import PascabayarModel
from src.models.prabayar import PrabayarModel

MODEL_PATHS = {
    "prabayar": "results/prabayar/models/model_prabayar.json",
    "pascabayar": "results/pascabayar/models/model_pascabayar.json",
}

models = {}
metadatas = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load models on startup
    try:
        models["prabayar"], metadatas["prabayar"] = PrabayarModel.load(MODEL_PATHS["prabayar"])
        print("Model Prabayar loaded successfully")
    except Exception as e:
        print(f"Failed to load Prabayar model: {e}")
        
    try:
        models["pascabayar"], metadatas["pascabayar"] = PascabayarModel.load(MODEL_PATHS["pascabayar"])
        print("Model Pascabayar loaded successfully")
    except Exception as e:
        print(f"Failed to load Pascabayar model: {e}")
        
    yield
    # Cleanup on shutdown
    models.clear()
    metadatas.clear()

app = FastAPI(title="Prediksi Listrik API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict/prepaid")
async def predict_prepaid(data: Dict[str, Any]):
    if "prabayar" not in models:
        raise HTTPException(status_code=500, detail="Model prabayar not loaded")
        
    model = models["prabayar"]
    metadata = metadatas["prabayar"]
    feature_columns = metadata["feature_columns"]
    x_scaler = metadata["x_scaler"]
    y_scaler = metadata["y_scaler"]
    
    input_values = []
    for col in feature_columns:
        val = data.get(col, 0)
        # Convert boolean to int if necessary, or parse string
        try:
            val = float(val) if not isinstance(val, bool) else float(int(val))
        except (ValueError, TypeError):
            val = 0.0
        input_values.append(val)
        
    x_scaled = transform_standard_scaler([input_values], x_scaler)
    # Predict expects a 2D numpy array and returns a 2D numpy array
    prediction_scaled = float(model.predict(np.array(x_scaled))[0][0])
    prediction = inverse_transform_target(prediction_scaled, y_scaler)
    
    # Ensure no negative prediction for days
    import math
    if math.isinf(prediction):
        prediction = 0.0 # Or max value cap
    prediction = max(0.0, float(prediction))
    
    return {
        "success": True,
        "prediction": round(prediction)
    }

@app.post("/predict/postpaid")
async def predict_postpaid(data: Dict[str, Any]):
    if "pascabayar" not in models:
        raise HTTPException(status_code=500, detail="Model pascabayar not loaded")
        
    model = models["pascabayar"]
    metadata = metadatas["pascabayar"]
    feature_columns = metadata["feature_columns"]
    x_scaler = metadata["x_scaler"]
    y_scaler = metadata["y_scaler"]
    
    input_values = []
    for col in feature_columns:
        val = data.get(col, 0)
        try:
            val = float(val) if not isinstance(val, bool) else float(int(val))
        except (ValueError, TypeError):
            val = 0.0
        input_values.append(val)
        
    x_scaled = transform_standard_scaler([input_values], x_scaler)
    # Predict expects a 2D numpy array and returns a 2D numpy array
    prediction_scaled = float(model.predict(np.array(x_scaled))[0][0])
    prediction = inverse_transform_target(prediction_scaled, y_scaler)
    
    # Ensure no negative prediction for cost
    import math
    if math.isinf(prediction):
        prediction = 0.0 # Or max value cap
    prediction = max(0.0, float(prediction))
    
    return {
        "success": True,
        "prediction": round(prediction)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
