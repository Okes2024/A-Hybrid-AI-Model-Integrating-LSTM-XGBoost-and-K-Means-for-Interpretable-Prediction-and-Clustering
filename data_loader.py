"""
Data loading and validation module
"""

import os
import pandas as pd
import numpy as np
from typing import Tuple
from src.config import Config


class DataLoader:
    """Handle water quality data loading with validation"""
    
    def __init__(self, filepath: str = Config.DATA_PATH):
        self.filepath = filepath
        self.df = None
        self.features = list(Config.STANDARDS.keys())
        
    def load(self) -> pd.DataFrame:
        """Load and validate data"""
        try:
            print(f"📂 Loading data from: {self.filepath}")
            
            if not os.path.exists(self.filepath):
                raise FileNotFoundError(f"Data file not found: {self.filepath}")
            
            self.df = pd.read_csv(self.filepath)
            print(f"✅ Loaded {self.df.shape[0]} rows × {self.df.shape[1]} columns")
            
            self._validate_columns()
            self._validate_quality()
            self._calculate_wqi()
            
            return self.df
            
        except Exception as e:
            print(f"❌ Data loading error: {e}")
            raise
    
    def _validate_columns(self):
        """Check required columns exist"""
        missing = [col for col in self.features if col not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        print(f"✅ All {len(self.features)} chemical parameters present")
    
    def _validate_quality(self):
        """Check data quality"""
        if self.df[self.features].isnull().sum().sum() > 0:
            print("⚠️  Filling missing values with median")
            self.df[self.features] = self.df[self.features].fillna(
                self.df[self.features].median()
            )
        
        negatives = (self.df[self.features] < 0).sum().sum()
        if negatives > 0:
            print(f"⚠️  Found {negatives} negative values - converting to absolute")
            self.df[self.features] = self.df[self.features].abs()
    
    def _calculate_wqi(self):
        """Calculate Water Quality Index with SYMMETRIC pH handling"""
        qn_values = pd.DataFrame()
        
        for param, limits in Config.STANDARDS.items():
            V_ideal = limits["V_ideal"]
            
            # SYMMETRIC handling for pH: treat acidic and alkaline equally
            if param == "pH":
                # Absolute distance from ideal (7.0), regardless of direction
                deviation = np.abs(self.df[param] - V_ideal)
                # Maximum acceptable deviation (to reach 8.5 or 5.5)
                max_deviation = limits.get("range", 1.5)
                # Calculate sub-index: 0% = perfect (7.0), 100% = worst (5.5 or 8.5)
                qi = (deviation / max_deviation) * 100
                
            else:
                # Normal asymmetric calculation for other parameters
                Sn = limits["Sn"]
                qi = ((self.df[param] - V_ideal) / (Sn - V_ideal)) * 100
            
            # Cap at 100%
            qn_values[param] = qi.clip(upper=100)
        
        # Weighted average
        total_weight = sum(l["weight"] for l in Config.STANDARDS.values())
        weights = {p: l["weight"]/total_weight for p, l in Config.STANDARDS.items()}
        
        self.df["WQI"] = sum(qn_values[p] * weights[p] for p in qn_values.columns)
        self.df["WQI_Class"] = self.df["WQI"].apply(self._classify_wqi)
        
        if 'Town' not in self.df.columns:
            self.df['Town'] = [f"Town_{i}" for i in range(len(self.df))]
        
        print(f"✅ WQI calculated (SYMMETRIC pH): {self.df['WQI_Class'].value_counts().to_dict()}")
    
    @staticmethod
    def _classify_wqi(wqi: float) -> str:
        if wqi <= 25: return "Excellent"
        elif wqi <= 50: return "Good"
        elif wqi <= 75: return "Fair"
        elif wqi <= 100: return "Poor"
        return "Unsuitable"
    
    def get_features_target(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get X, y_reg, y_cls"""
        X = self.df[self.features].values
        y_reg = self.df['WQI'].values
        y_cls = self.df['WQI_Class'].values
        return X, y_reg, y_cls