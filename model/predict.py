# model/predict.py
import joblib
import pandas as pd

def load_model(model_path):
    """Carga un modelo entrenado"""
    return joblib.load(model_path)

def predict_online(model, features):
    """Predicción online con parámetros"""
    df = pd.DataFrame([features])
    return model.predict(df)[0]

def predict_offline(model, filepath):
    """Predicción batch (offline) a partir de un archivo CSV"""
    df = pd.read_csv(filepath)
    predictions = model.predict(df)
    return predictions.tolist()
