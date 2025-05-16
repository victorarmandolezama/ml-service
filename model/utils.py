# model/utils.py
import pandas as pd
from sklearn.model_selection import train_test_split

def load_dataset(filepath):
    """Carga el dataset desde un archivo CSV"""
    return pd.read_csv(filepath)

def split_data(df, target_column='target'):
    """Divide el dataset en features (X) y target (y)"""
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    return train_test_split(X, y, test_size=0.2, random_state=42)
