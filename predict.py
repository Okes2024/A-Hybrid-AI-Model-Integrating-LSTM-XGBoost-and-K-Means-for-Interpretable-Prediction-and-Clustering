"""
Standalone prediction script using trained models
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import argparse
import json
from src.predictor import WaterQualityPredictor


def main():
    parser = argparse.ArgumentParser(description='Predict Water Quality Index')
    parser.add_argument('--input', '-i', type=str, help='JSON file with water parameters')
    parser.add_argument('--interactive', '-t', action='store_true', help='Interactive mode')
    
    args = parser.parse_args()
    
    predictor = WaterQualityPredictor()
    
    if args.interactive:
        # Interactive mode
        print("\n🔬 Water Quality Prediction - Interactive Mode")
        print("Enter water parameters (press Enter to use default):\n")
        
        defaults = {
            'pH': 7.0, 'EC': 300, 'TDS': 150, 'NO3': 0.2, 'Cl': 25, 
            'SO4': 2.0, 'Ca': 12, 'Mg': 3.5, 'Na': 7.0, 'Iron': 0.2
        }
        
        sample = {}
        for param, default in defaults.items():
            val = input(f"{param} [{default}]: ").strip()
            sample[param] = float(val) if val else default
        
        result = predictor.predict(sample)
        
        print("\n" + "="*50)
        print("PREDICTION RESULTS")
        print("="*50)
        for key, value in result.items():
            print(f"{key:15}: {value}")
    
    elif args.input:
        # Batch mode from JSON file
        with open(args.input, 'r') as f:
            sample = json.load(f)
        result = predictor.predict(sample)
        print(json.dumps(result, indent=2))
    
    else:
        # Demo mode
        print("Running demo prediction...")
        sample = {
            'pH': 6.5, 'EC': 400, 'TDS': 200, 'NO3': 0.2, 'Cl': 30, 
            'SO4': 2.0, 'Ca': 15, 'Mg': 4.0, 'Na': 8.0, 'Iron': 0.3
        }
        result = predictor.predict(sample)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()