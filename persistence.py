"""
Model persistence utilities
"""

import os
import json
import joblib
from datetime import datetime
from src.config import Config

# Handle Keras import compatibility
try:
    # Keras 3.x
    from keras.saving import load_model
except ImportError:
    try:
        # TensorFlow Keras
        from tensorflow.keras.models import load_model
    except ImportError:
        # Fallback
        load_model = None


class ModelPersistence:
    """Save and load models"""
    
    @staticmethod
    def save(ensemble, classifier, preprocessor, prefix: str = None):
        """Save all models"""
        if prefix is None:
            prefix = Config.MODELS_DIR
        
        os.makedirs(prefix, exist_ok=True)
        print(f"\n💾 Saving models to {prefix}...")
        
        # Save XGBoost
        joblib.dump(ensemble.xgb, f'{prefix}/xgb_regressor.pkl')
        
        # Save LSTM - use native Keras format instead of HDF5
        if hasattr(ensemble.lstm.model, 'save'):
            try:
                # Try native Keras format (recommended)
                ensemble.lstm.model.save(f'{prefix}/lstm_model.keras')
            except:
                # Fallback to HDF5 with warning
                ensemble.lstm.model.save(f'{prefix}/lstm_model.h5')
        
        # Save meta-learner
        joblib.dump(ensemble.meta_learner, f'{prefix}/meta_learner.pkl')
        
        # Save classifier
        joblib.dump(classifier.model, f'{prefix}/xgb_classifier.pkl')
        joblib.dump(classifier.encoder, f'{prefix}/label_encoder.pkl')
        
        # Save preprocessor
        joblib.dump(preprocessor.scaler, f'{prefix}/scaler.pkl')
        joblib.dump(preprocessor.kmeans, f'{prefix}/kmeans.pkl')
        joblib.dump(preprocessor.optimal_k, f'{prefix}/optimal_k.pkl')
        
        # Save config
        config = {
            'features': list(Config.STANDARDS.keys()),
            'n_clusters': preprocessor.optimal_k,
            'classes': list(classifier.encoder.classes_),
            'timestamp': datetime.now().isoformat(),
            'lstm_format': 'keras'  # Track format for loading
        }
        with open(f'{prefix}/config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        print("✅ All models saved")
    
    @staticmethod
    def load(prefix: str = None):
        """Load all models"""
        if prefix is None:
            prefix = Config.MODELS_DIR
        
        print(f"📂 Loading models from {prefix}...")
        
        # Check which LSTM format was saved
        lstm_path = f'{prefix}/lstm_model.keras'
        if not os.path.exists(lstm_path):
            lstm_path = f'{prefix}/lstm_model.h5'
        
        models = {
            'xgb': joblib.load(f'{prefix}/xgb_regressor.pkl'),
            'lstm': load_model(lstm_path, compile=False),  # compile=False to avoid metric issues
            'meta': joblib.load(f'{prefix}/meta_learner.pkl'),
            'classifier': joblib.load(f'{prefix}/xgb_classifier.pkl'),
            'encoder': joblib.load(f'{prefix}/label_encoder.pkl'),
            'scaler': joblib.load(f'{prefix}/scaler.pkl'),
            'kmeans': joblib.load(f'{prefix}/kmeans.pkl'),
            'optimal_k': joblib.load(f'{prefix}/optimal_k.pkl')
        }
        
        # Recompile LSTM manually to avoid metric serialization issues
        if models['lstm'] is not None:
            from tensorflow.keras.optimizers import Adam
            models['lstm'].compile(
                optimizer=Adam(learning_rate=0.001),
                loss='mse'
            )
        
        with open(f'{prefix}/config.json', 'r') as f:
            models['config'] = json.load(f)
        
        print("✅ All models loaded")
        return models