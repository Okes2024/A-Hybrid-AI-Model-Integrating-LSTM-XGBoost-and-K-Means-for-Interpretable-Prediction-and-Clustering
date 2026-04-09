"""
Data preprocessing without leakage
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from src.config import Config


class Preprocessor:
    """Preprocess data with proper train/test separation"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.kmeans = None
        self.optimal_k = None
        self.is_fitted = False
    
    def fit(self, X_train: np.ndarray):
        """Fit on training data only"""
        print("🔧 Fitting preprocessor...")
        
        # Fit scaler
        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)
        
        # Find optimal clusters
        self.optimal_k = self._find_optimal_k(X_train_scaled)
        
        # Fit K-Means
        self.kmeans = KMeans(
            n_clusters=self.optimal_k,
            random_state=Config.RANDOM_STATE,
            n_init=10
        )
        self.kmeans.fit(X_train_scaled)
        
        self.is_fitted = True
        print(f"✅ Preprocessor fitted (k={self.optimal_k})")
        return self
    
    def _find_optimal_k(self, X: np.ndarray) -> int:
        """Find optimal k using silhouette score"""
        best_k, best_score = 2, -1
        
        for k in range(Config.N_CLUSTERS_MIN, Config.N_CLUSTERS_MAX + 1):
            kmeans_temp = KMeans(n_clusters=k, random_state=Config.RANDOM_STATE, n_init=10)
            labels = kmeans_temp.fit_predict(X)
            score = silhouette_score(X, labels)
            if score > best_score:
                best_k, best_score = k, score
        
        return best_k
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data"""
        if not self.is_fitted:
            raise ValueError("Preprocessor not fitted!")
        
        X_scaled = self.scaler.transform(X)
        clusters = self.kmeans.predict(X_scaled)
        return np.column_stack([X_scaled, clusters.reshape(-1, 1)])
    
    def fit_transform(self, X_train: np.ndarray) -> np.ndarray:
        self.fit(X_train)
        return self.transform(X_train)