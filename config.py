"""
Configuration settings for the Hybrid AI Water Quality Model
"""

import os
from typing import Dict, Any


class Config:
    """Central configuration class"""
    
    RANDOM_STATE = 42
    
    # Clean professional path
    DATA_PATH = 'data/water_parameters.csv'
    MODELS_DIR = 'models'
    RESULTS_DIR = 'results'
    LOGS_DIR = 'logs'
    
    TEST_SIZE = 0.2
    
    N_CLUSTERS_MIN = 2
    N_CLUSTERS_MAX = 6
    
    XGB_REG_PARAMS = {
        'n_estimators': 200,
        'learning_rate': 0.05,
        'max_depth': 4,
        'min_child_weight': 3,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': RANDOM_STATE,
        'n_jobs': -1
    }
    
    XGB_CLS_PARAMS = {
        'n_estimators': 200,
        'learning_rate': 0.05,
        'max_depth': 4,
        'random_state': RANDOM_STATE,
        'n_jobs': -1
    }
    
    LSTM_PARAMS = {
        'units_layer1': 64,
        'units_layer2': 32,
        'dense_units': 16,
        'dropout_rate': 0.2,
        'l2_reg': 0.001,
        'learning_rate': 0.001,
        'batch_size': 8,
        'max_epochs': 200,
        'patience': 20
    }
    
    STANDARDS = {
    # pH: Special symmetric handling around 7.0
    # Lower bound: 6.5, Ideal: 7.0, Upper bound: 8.5
    # Range below 7.0: 0.5 units, Range above 7.0: 1.5 units
    # We use absolute deviation from ideal, normalized by total range
    "pH": {"Sn": 1.5, "V_ideal": 7.0, "weight": 0.219, "symmetric": True, "range": 1.5},
    
    # Others remain normal
    "EC": {"Sn": 1500, "V_ideal": 0, "weight": 0.037},
    "TDS": {"Sn": 1000, "V_ideal": 0, "weight": 0.056},
    "NO3": {"Sn": 50, "V_ideal": 0, "weight": 0.222},
    "Cl": {"Sn": 250, "V_ideal": 0, "weight": 0.044},
    "SO4": {"Sn": 250, "V_ideal": 0, "weight": 0.044},
    "Ca": {"Sn": 75, "V_ideal": 0, "weight": 0.148},
    "Mg": {"Sn": 50, "V_ideal": 0, "weight": 0.222},
    "Na": {"Sn": 200, "V_ideal": 0, "weight": 0.056},
    "Iron": {"Sn": 0.3, "V_ideal": 0, "weight": 0.037}
    }
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories"""
        for dir_name in [cls.MODELS_DIR, cls.RESULTS_DIR, cls.LOGS_DIR, 'data']:
            os.makedirs(dir_name, exist_ok=True)