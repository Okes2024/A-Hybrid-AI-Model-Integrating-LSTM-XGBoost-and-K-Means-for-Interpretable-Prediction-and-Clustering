"""
LSTM model with regularization
"""

import numpy as np  
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from src.config import Config


class LSTMModel:
    """LSTM with BatchNorm, Dropout, and EarlyStopping"""
    
    def __init__(self, input_dim: int):
        self.input_dim = input_dim
        self.model = None
        self.history = None
        self.params = Config.LSTM_PARAMS
    
    def build(self):
        """Build model architecture"""
        p = self.params
        
        model = Sequential([
            LSTM(p['units_layer1'], activation='relu', return_sequences=True,
                 input_shape=(1, self.input_dim), kernel_regularizer=l2(p['l2_reg'])),
            BatchNormalization(),
            Dropout(p['dropout_rate']),
            
            LSTM(p['units_layer2'], activation='relu', kernel_regularizer=l2(p['l2_reg'])),
            BatchNormalization(),
            Dropout(p['dropout_rate']),
            
            Dense(p['dense_units'], activation='relu'),
            Dense(1)
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=p['learning_rate']),
            loss='mse',
            metrics=['mae']
        )
        
        self.model = model
        return model
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: np.ndarray = None, y_val: np.ndarray = None):
        """Train with callbacks"""
        
        X_train_lstm = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
        
        val_data = None
        if X_val is not None and y_val is not None:
            X_val_lstm = X_val.reshape((X_val.shape[0], 1, X_val.shape[1]))
            val_data = (X_val_lstm, y_val)
        
        # FIX: Use self.params instead of undefined p
        callbacks = [
            EarlyStopping(
                monitor='val_loss' if val_data else 'loss',
                patience=self.params['patience'],
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss' if val_data else 'loss',
                factor=0.5,
                patience=10,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        print("🧠 Training LSTM...")
        self.history = self.model.fit(
            X_train_lstm, y_train,
            epochs=self.params['max_epochs'],
            batch_size=self.params['batch_size'],
            validation_data=val_data,
            callbacks=callbacks,
            verbose=1
        )
        
        print(f"✅ LSTM trained ({len(self.history.history['loss'])} epochs)")
        return self.history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        X_lstm = X.reshape((X.shape[0], 1, X.shape[1]))
        return self.model.predict(X_lstm, verbose=0).flatten()