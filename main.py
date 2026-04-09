"""
Main training pipeline for Hybrid AI Water Quality Model
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from src.config import Config
from src.data_loader import DataLoader
from src.preprocessor import Preprocessor
from src.models.ensemble import HybridEnsemble
from src.models.classifier import WQIClassifier
from src.persistence import ModelPersistence
from src.predictor import WaterQualityPredictor
from src.visualization import Visualizer
from src.evaluation import Evaluator


def set_seeds(seed=42):
    """Set all random seeds for reproducibility"""
    np.random.seed(seed)
    tf.random.set_seed(seed)
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def main():
    """Main training pipeline"""
    
    print("="*80)
    print("HYBRID AI MODEL - WATER QUALITY PREDICTION")
    print("LSTM + XGBoost + K-Means with Stacking Ensemble")
    print("="*80)
    
    # Setup
    set_seeds(Config.RANDOM_STATE)
    Config.create_directories()
    
    # 1. Load data
    loader = DataLoader()
    df = loader.load()
    X, y_reg, y_cls = loader.get_features_target()
    
    # 2. Split data
    X_train, X_test, y_train_reg, y_test_reg, y_train_cls, y_test_cls = train_test_split(
        X, y_reg, y_cls, 
        test_size=Config.TEST_SIZE, 
        random_state=Config.RANDOM_STATE,
        stratify=y_cls
    )
    
    print(f"\n📐 Split: Train={len(X_train)}, Test={len(X_test)}")
    
    # 3. Preprocess (NO LEAKAGE)
    preprocessor = Preprocessor()
    X_train_enhanced = preprocessor.fit_transform(X_train)
    X_test_enhanced = preprocessor.transform(X_test)
    
    print(f"   Features: {X_train_enhanced.shape[1]} (original {X_train.shape[1]} + cluster)")
    
    # 4. Train ensemble
    ensemble = HybridEnsemble()
    ensemble.train(X_train_enhanced, y_train_reg, X_test_enhanced, y_test_reg)
    
    # 5. Evaluate regression
    predictions = ensemble.predict(X_test_enhanced, return_individual=True)
    
    print("\n" + "="*60)
    print("REGRESSION RESULTS")
    print("="*60)
    
    for model_name, y_pred in predictions.items():
        if model_name == 'weights':
            continue
        metrics = Evaluator.evaluate_regression(y_test_reg, y_pred)
        Evaluator.print_results(metrics, model_name.upper())
    
    # 6. Train classifier
    classifier = WQIClassifier()
    classifier.train(X_train_enhanced, y_train_cls)
    y_pred_cls = classifier.predict(X_test_enhanced)
    cls_metrics = Evaluator.evaluate_classification(y_test_cls, y_pred_cls)
    
    print("\n" + "="*60)
    print("CLASSIFICATION RESULTS")
    print("="*60)
    print(f"Accuracy: {cls_metrics['accuracy']:.4f}")
    print(f"\nClassification Report:\n{cls_metrics['classification_report']}")
    
    # 7. Visualize
    visualizer = Visualizer()
    visualizer.create_dashboard(df, ensemble, X_test_enhanced, y_test_reg, preprocessor)
    # In main.py, after training and dashboard creation
    visualizer.save_readme_charts(df, ensemble, X_test_enhanced, y_test_reg, preprocessor)
    
    # 8. Save models
    ModelPersistence.save(ensemble, classifier, preprocessor)
    
    # 9. Demo prediction API
    print("\n" + "="*60)
    print("PREDICTION API DEMO")
    print("="*60)
    
    predictor = WaterQualityPredictor()
    
    test_samples = [
        {'pH': 6.5, 'EC': 400, 'TDS': 200, 'NO3': 0.2, 'Cl': 30, 
         'SO4': 2.0, 'Ca': 15, 'Mg': 4.0, 'Na': 8.0, 'Iron': 0.3},
        {'pH': 5.8, 'EC': 800, 'TDS': 400, 'NO3': 0.4, 'Cl': 60, 
         'SO4': 5.0, 'Ca': 30, 'Mg': 8.0, 'Na': 15.0, 'Iron': 0.6},
    ]
    
    for i, sample in enumerate(test_samples, 1):
        result = predictor.predict(sample)
        print(f"\nSample {i}: WQI={result['WQI']} ({result['WQI_Class']}), "
              f"Confidence={result['Confidence']:.1%}")
    
    print("\n" + "="*80)
    print("✅ PIPELINE COMPLETE - Models saved to models/")
    print("="*80)


if __name__ == "__main__":
    main()