"""
WQI Classification model
"""

import numpy as np
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from src.config import Config


class WQIClassifier:
    """XGBoost classifier for WQI classes"""
    
    def __init__(self):
        self.model = None
        self.encoder = LabelEncoder()
        self.is_trained = False
    
    def train(self, X_train: np.ndarray, y_train: list):
        """Train classifier"""
        print("🎯 Training classifier...")
        
        y_encoded = self.encoder.fit_transform(y_train)
        
        self.model = XGBClassifier(**Config.XGB_CLS_PARAMS)
        self.model.fit(X_train, y_encoded)
        
        self.is_trained = True
        print(f"✅ Classifier trained ({len(self.encoder.classes_)} classes)")
        return self
    
    def predict(self, X: np.ndarray) -> list:
        if not self.is_trained:
            raise ValueError("Classifier not trained!")
        y_encoded = self.model.predict(X)
        return self.encoder.inverse_transform(y_encoded)
    
    def predict_proba(self, X: np.ndarray):
        return self.model.predict_proba(X)