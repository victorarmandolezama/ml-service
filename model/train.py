# model/train.py
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import joblib
from .utils import load_dataset, split_data

def train_model(filepath, model_type='rf', hyperparams={}):
    df = load_dataset(filepath)
    X_train, X_test, y_train, y_test = split_data(df)
    if model_type == 'rf':
        model = RandomForestClassifier(**hyperparams)
    elif model_type == 'xgb':
        model = XGBClassifier(
            **hyperparams, use_label_encoder=False, eval_metric='logloss')
    else:
        raise ValueError("Modelo no soportado. Usa 'rf' o 'xgb'.")
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    # Guardar modelo entrenado
    joblib.dump(model, f'saved_models/{model_type}_model.pkl')
    return {"accuracy": accuracy}
