# export_dataset.py
import pandas as pd
from sklearn.datasets import load_breast_cancer
print('starting')
# Cargar dataset
data = load_breast_cancer()
print('data loaded')
# Convertir a DataFrame
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target  # 0 = benigno, 1 = maligno
# Guardar en archivo CSV
df.to_csv('data/breast_cancer.csv', index=False)
print("Dataset exportado exitosamente a 'data/breast_cancer.csv'")
