"""
Model evaluation utilities
"""

from sklearn.metrics import (r2_score, mean_absolute_error, mean_squared_error,
                             accuracy_score, confusion_matrix, classification_report)
import numpy as np


class Evaluator:
    """Evaluate model performance"""
    
    @staticmethod
    def evaluate_regression(y_true, y_pred):
        """Calculate regression metrics"""
        return {
            'R2': r2_score(y_true, y_pred),
            'MAE': mean_absolute_error(y_true, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'MAPE': np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        }
    
    @staticmethod
    def evaluate_classification(y_true, y_pred):
        """Calculate classification metrics"""
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'confusion_matrix': confusion_matrix(y_true, y_pred),
            'classification_report': classification_report(y_true, y_pred)
        }
    
    @staticmethod
    def print_results(results: dict, model_name: str):
        """Print formatted results"""
        print(f"\n{model_name}:")
        for metric, value in results.items():
            if isinstance(value, (int, float)):
                print(f"   {metric}: {value:.4f}")