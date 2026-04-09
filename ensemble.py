"""
Hybrid Ensemble: Stacking XGBoost + LSTM
"""

import numpy as np
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from src.config import Config
from src.models.lstm_model import LSTMModel


class HybridEnsemble:
    """Stacking ensemble with meta-learner"""
    
    def __init__(self):
        self.xgb = None
        self.lstm = None
        self.meta_learner = None
        self.is_trained = False
        
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray = None, y_val: np.ndarray = None):
        """Train base models and meta-learner"""
        
        print("\n" + "="*60)
        print("TRAINING HYBRID ENSEMBLE")
        print("="*60)
        
        # Train XGBoost
        print("🚀 Training XGBoost...")
        self.xgb = XGBRegressor(**Config.XGB_REG_PARAMS)
        self.xgb.fit(X_train, y_train)
        
        # Train LSTM
        self.lstm = LSTMModel(input_dim=X_train.shape[1])
        self.lstm.build()
        self.lstm.train(X_train, y_train, X_val, y_val)
        
        # Generate predictions for meta-learner
        print("🔀 Training meta-learner...")
        xgb_pred = self.xgb.predict(X_train)
        lstm_pred = self.lstm.predict(X_train)
        stacked = np.column_stack([xgb_pred, lstm_pred])
        
        # Ridge regression as meta-learner
        self.meta_learner = Ridge(alpha=1.0)
        self.meta_learner.fit(stacked, y_train)
        
        self.is_trained = True
        print("✅ Ensemble training complete")
        return self
    
    def predict(self, X: np.ndarray, return_individual: bool = False):
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Ensemble not trained!")
        
        xgb_pred = self.xgb.predict(X)
        lstm_pred = self.lstm.predict(X)
        stacked = np.column_stack([xgb_pred, lstm_pred])
        ensemble_pred = self.meta_learner.predict(stacked)
        
        if return_individual:
            return {
                'ensemble': ensemble_pred,
                'xgboost': xgb_pred,
                'lstm': lstm_pred
            }
        return ensemble_pred