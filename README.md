# MLService API — Diagnóstico de Cáncer de Mama

Este proyecto implementa una API REST y una app interactiva para entrenar y utilizar modelos de Machine Learning basados en el conjunto de datos de diagnóstico de cáncer de mama (Breast Cancer Wisconsin). Utiliza `Flask` para la API y `Streamlit` para una futura interfaz gráfica.

---

## Estructura del Proyecto

```
ml_service/
├── app.py                  # API principal (Flask)
├── model/
│   ├── train.py            # Entrenamiento de modelos
│   ├── predict.py          # Predicciones online y offline
│   └── utils.py            # Carga y partición de datos
├── data/
│   ├── breast_cancer.csv   # Dataset base en CSV
│   └── test_batch.csv      # Prueba batch de predicción
├── saved_models/           # Modelos entrenados (.pkl)
├── export_dataset.py       # Exportar dataset desde scikit-learn
├── requirements.txt
└── README.md
```

---

## Requisitos

Recomendado: usar [Conda](https://docs.conda.io/) para entorno virtual.

```bash
conda create -n mlservice python=3.10 -y
conda activate mlservice

# Instalar paquetes principales
conda install flask scikit-learn pandas numpy -y
pip install xgboost streamlit joblib
```

---

## Uso

### 1. Exportar el dataset

```bash
python export_dataset.py
```

Esto genera `data/breast_cancer.csv`.

---

### 2. Iniciar la API Flask

```bash
python app.py
```

La API estará disponible en:  
`http://127.0.0.1:5000/`

---

### 3. Entrenamiento de modelo

```bash
curl -X POST http://127.0.0.1:5000/train \
-H "Content-Type: application/json" \
-d '{
  "filepath": "data/breast_cancer.csv",
  "model_type": "rf",
  "hyperparams": {
    "n_estimators": 100,
    "max_depth": 5
  }
}'
```

---

### 4. Predicción online

```bash
curl -X POST http://127.0.0.1:5000/predict-online \
-H "Content-Type: application/json" \
-d '{
  "model_type": "rf",
  "features": {
    "mean radius": 14.0,
    "mean texture": 20.0,
    ...
    "worst fractal dimension": 0.09
  }
}'
```

---

### 5. Predicción offline (batch)

```bash
curl -X POST http://127.0.0.1:5000/predict-offline \
-H "Content-Type: application/json" \
-d '{
  "model_type": "rf",
  "filepath": "data/test_batch.csv"
}'
```

---

## Próximamente

- App visual en `Streamlit` para cargar datos, entrenar modelos y obtener predicciones fácilmente.
- Posibilidad de usar otros datasets como `Heart Disease` o `HIGGS`.

---

## Autor

Proyecto base creado por Victor Armando Lezama, como parte de una arquitectura de experimentación en ML y APIs ligeras.
