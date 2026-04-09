"""
Production prediction API
"""

import numpy as np
from src.persistence import ModelPersistence
from src.config import Config


class WaterQualityPredictor:
    """Production-ready predictor"""
    
    def __init__(self, models_dir: str = None):
        if models_dir is None:
            models_dir = Config.MODELS_DIR
        self.models_dir = models_dir
        self.models = None
        self.is_loaded = False
    
    def load(self):
        """Load models"""
        self.models = ModelPersistence.load(self.models_dir)
        self.is_loaded = True
        return self
    
    def predict(self, sample: dict) -> dict:
        """Predict WQI for a single sample"""
        if not self.is_loaded:
            self.load()
        
        # Prepare input
        features = self.models['config']['features']
        X = np.array([[sample.get(f, 0) for f in features]])
        
        # Preprocess
        X_scaled = self.models['scaler'].transform(X)
        cluster = self.models['kmeans'].predict(X_scaled)[0]
        X_enhanced = np.column_stack([X_scaled, [[cluster]]])
        
        # Predict
        xgb_pred = self.models['xgb'].predict(X_enhanced)[0]
        
        X_lstm = X_enhanced.reshape((1, 1, X_enhanced.shape[1]))
        lstm_pred = self.models['lstm'].predict(X_lstm, verbose=0)[0][0]
        
        # Ensemble
        stacked = np.array([[xgb_pred, lstm_pred]])
        ensemble_pred = self.models['meta'].predict(stacked)[0]
        
        # Classify
        cls_encoded = self.models['classifier'].predict(X_enhanced)[0]
        wqi_class = self.models['encoder'].inverse_transform([cls_encoded])[0]
        cls_proba = self.models['classifier'].predict_proba(X_enhanced)[0]
        
        return {
            'WQI': round(float(ensemble_pred), 2),
            'WQI_Class': wqi_class,
            'Confidence': round(float(np.max(cls_proba)), 3),
            'XGBoost_WQI': round(float(xgb_pred), 2),
            'LSTM_WQI': round(float(lstm_pred), 2),
            'Cluster': int(cluster)
        }
    
    def predict_batch(self, samples: list) -> list:
        """Predict for multiple samples"""
        return [self.predict(s) for s in samples]