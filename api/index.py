from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any
from contextlib import asynccontextmanager
import numpy as np
import pandas as pd
import os
import math

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from src.pipeline.preprocessing import preprocess, transform_standard_scaler, inverse_transform_target
from src.pipeline.feature_extraction import extract_features_and_target
from src.models.pascabayar import PascabayarModel
from src.models.prabayar import PrabayarModel
from src.config.config import config

MODEL_PATHS = {
    "prabayar": os.path.join(BASE_DIR, "results", "prabayar", "models", "model_prabayar.json"),
    "pascabayar": os.path.join(BASE_DIR, "results", "pascabayar", "models", "model_pascabayar.json"),
}

models = {}
metadatas = {}

load_errors = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load models on startup
    try:
        models["prabayar"], metadatas["prabayar"] = PrabayarModel.load(MODEL_PATHS["prabayar"])
        print("Model Prabayar loaded successfully")
    except Exception as e:
        load_errors["prabayar"] = str(e)
        print(f"Failed to load Prabayar model: {e}")
        
    try:
        models["pascabayar"], metadatas["pascabayar"] = PascabayarModel.load(MODEL_PATHS["pascabayar"])
        print("Model Pascabayar loaded successfully")
    except Exception as e:
        load_errors["pascabayar"] = str(e)
        print(f"Failed to load Pascabayar model: {e}")
        
    yield
    # Cleanup on shutdown
    models.clear()
    metadatas.clear()
    load_errors.clear()

app = FastAPI(title="Prediksi Listrik API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {
        "message": "API Is Running",
        "status": "success"
    }

def process_inference_data(data: Dict[str, Any], model_type: str, minmax_scaler_params: dict):
    df_raw = pd.DataFrame([data])
    
    if "Daya_Listrik_Rumah_VA" in df_raw.columns:
        df_raw["Daya_Listrik_Rumah_VA"] = df_raw["Daya_Listrik_Rumah_VA"].replace({
            "Tidak tahu": 900,
            "> 5500": 7700,
        })
        df_raw["Daya_Listrik_Rumah_VA"] = pd.to_numeric(
            df_raw["Daya_Listrik_Rumah_VA"], errors="coerce"
        ).fillna(900)

    if "Status_Subsidi_Listrik" in df_raw.columns:
        df_raw["Status_Subsidi_Listrik"] = df_raw["Status_Subsidi_Listrik"].map({
            "Subsidi": 0,
            "Non Subsidi": 1,
        }).fillna(1).astype(float)

    if "Alat_Lain_Ada" in df_raw.columns:
        df_raw["Alat_Lain_Ada"] = df_raw["Alat_Lain_Ada"].map({
            "Tidak": 0,
            "Ya": 1,
        }).fillna(0).astype(float)

    df_processed, _ = preprocess(df_raw, scaler_params=minmax_scaler_params)

    feature_columns = config["features"][model_type]
    
    input_values = []
    for col in feature_columns:
        if col in df_processed.columns:
            val = df_processed[col].iloc[0]
            val = float(val) if pd.notna(val) else 0.0
        else:
            val = 0.0
        input_values.append(val)
        
    return input_values

@app.post("/predict/prepaid")
async def predict_prepaid(data: Dict[str, Any]):
    if "prabayar" not in models:
        error_msg = load_errors.get("prabayar", "Unknown error loading model")
        raise HTTPException(status_code=500, detail=f"Model prabayar not loaded. Error: {error_msg}")
        
    model = models["prabayar"]
    metadata = metadatas["prabayar"]
    x_scaler = metadata["x_scaler"]
    y_scaler = metadata["y_scaler"]
    minmax_scaler_params = metadata.get("minmax_scaler_params", {})
    
    # Preprocess
    input_values = process_inference_data(data, "prabayar", minmax_scaler_params)
        
    # Scale
    x_scaled = transform_standard_scaler([input_values], x_scaler)
    
    # Predict
    prediction_scaled = float(model.predict(np.array(x_scaled))[0][0])
    
    # Clip/bound output logis agar exponensial tidak meledak (rentang log1p wajar)
    # Max log untuk target prabayar misalnya 6.0 (karena ~400 hari maks log1p(400) = 5.99)
    if y_scaler.get("use_log", False):
        prediction_scaled = max(0.0, min(prediction_scaled, 1.5)) # Asumsikan skala [0,1] sedikit overshoot boleh
    
    prediction = inverse_transform_target(prediction_scaled, y_scaler)
    
    if math.isinf(prediction) or math.isnan(prediction):
        prediction = 0.0
        
    prediction = max(0.0, float(prediction))
    
    return {
        "success": True,
        "prediction": round(prediction)
    }

@app.post("/predict/postpaid")
async def predict_postpaid(data: Dict[str, Any]):
    if "pascabayar" not in models:
        error_msg = load_errors.get("pascabayar", "Unknown error loading model")
        raise HTTPException(status_code=500, detail=f"Model pascabayar not loaded. Error: {error_msg}")
        
    model = models["pascabayar"]
    metadata = metadatas["pascabayar"]
    x_scaler = metadata["x_scaler"]
    y_scaler = metadata["y_scaler"]
    minmax_scaler_params = metadata.get("minmax_scaler_params", {})
    
    # Preprocess
    input_values = process_inference_data(data, "pascabayar", minmax_scaler_params)
        
    # Scale
    x_scaled = transform_standard_scaler([input_values], x_scaler)
    
    # Predict
    prediction_scaled = float(model.predict(np.array(x_scaled))[0][0])
    
    # Clip/bound output logis
    # Target pascabayar Rp 10rb - 1.5jt. Log1p rentang [9.2, 14.2].
    if y_scaler.get("use_log", False):
         prediction_scaled = max(0.0, min(prediction_scaled, 1.5))
         
    prediction = inverse_transform_target(prediction_scaled, y_scaler)
    
    if math.isinf(prediction) or math.isnan(prediction):
        prediction = 0.0
        
    prediction = max(0.0, float(prediction))
    
    return {
        "success": True,
        "prediction": round(prediction)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("index:app", host="0.0.0.0", port=8000, reload=True)
