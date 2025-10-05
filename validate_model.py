#!/usr/bin/env python3
"""
Model Validation and Environment Consistency Checker
Helps debug differences between local and Render predictions.
"""

import joblib
import json
import hashlib
import os
import numpy as np
import pandas as pd
from datetime import datetime

def validate_model_file(filename):
    """Validate that a model file is properly formatted and accessible."""
    
    print(f"🔍 Validating model file: {filename}")
    
    if not os.path.exists(filename):
        print(f"❌ Model file not found: {filename}")
        return False
    
    try:
        # Load model
        model_data = joblib.load(filename)
        
        # Check required components
        required_keys = ['model', 'label_encoders', 'feature_columns']
        missing_keys = [key for key in required_keys if key not in model_data]
        
        if missing_keys:
            print(f"❌ Missing required keys: {missing_keys}")
            return False
        
        # Check model signature if available
        if 'model_signature' in model_data:
            signature = model_data['model_signature']
            print(f"✅ Model signature found:")
            print(f"   Version: {signature.get('model_version', 'Unknown')}")
            print(f"   Features: {signature.get('n_features', 'Unknown')}")
            print(f"   Accuracy: {signature.get('accuracy', 'Unknown')}")
            print(f"   Training Date: {signature.get('training_date', 'Unknown')}")
        
        # Calculate file hash for integrity checking
        with open(filename, 'rb') as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        
        print(f"✅ Model file valid")
        print(f"   File size: {os.path.getsize(filename) / (1024*1024):.2f} MB")
        print(f"   File hash: {file_hash[:12]}...")
        print(f"   Features: {len(model_data['feature_columns'])}")
        
        return True, file_hash, model_data
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

def test_prediction_consistency(model_data, test_cases=None):
    """Test that predictions are consistent across multiple runs."""
    
    print(f"\n🧪 Testing prediction consistency...")
    
    if test_cases is None:
        test_cases = [
            {
                'Age': 55.0,
                'Gender': 'Male',
                'Blood Pressure': 150.0,
                'Cholesterol Level': 240.0,
                'BMI': 28.5,
                'Exercise Habits': 'Medium',
                'Alcohol Consumption': 'Light',
                'Stress Level': 'Medium',
                'Sleep Hours': 7.0,
                'Sugar Consumption': 'Medium',
                'Triglyceride Level': 180.0,
                'Fasting Blood Sugar': 95.0,
                'CRP Level': 2.5,
                'Homocysteine Level': 12.0
            }
        ]
    
    try:
        from enhanced_model_wrapper import EnhancedHeartDiseasePredictor
        
        predictor = EnhancedHeartDiseasePredictor()
        predictor.model_data = model_data
        predictor.is_loaded = True
        
        results = []
        
        for i, test_case in enumerate(test_cases):
            print(f"🔬 Running test case {i+1}...")
            
            # Run the same prediction multiple times to check consistency
            predictions = []
            for run in range(3):
                try:
                    result = predictor.predict(test_case)
                    predictions.append({
                        'prediction': result['prediction'],
                        'risk_probability': result['risk_probability'], 
                        'risk_percentage': result['risk_percentage'],
                        'confidence': result['confidence']
                    })
                except Exception as e:
                    print(f"❌ Prediction run {run+1} failed: {e}")
                    return False
            
            # Check consistency
            first_pred = predictions[0]
            consistent = all(
                pred['prediction'] == first_pred['prediction'] and
                abs(pred['risk_probability'] - first_pred['risk_probability']) < 0.001
                for pred in predictions
            )
            
            if consistent:
                print(f"✅ Test case {i+1} consistent: {first_pred['prediction']} ({first_pred['risk_percentage']})")
                results.append(first_pred)
            else:
                print(f"❌ Test case {i+1} inconsistent predictions:")
                for j, pred in enumerate(predictions):
                    print(f"   Run {j+1}: {pred['prediction']} ({pred['risk_percentage']})")
                return False
        
        print("✅ All prediction consistency tests passed!")
        return True, results
        
    except Exception as e:
        print(f"❌ Consistency testing failed: {e}")
        return False

def create_environment_report():
    """Create a report of the current environment for debugging."""
    
    print(f"\n📋 Creating environment report...")
    
    import sys
    import platform
    
    try:
        import sklearn
        sklearn_version = sklearn.__version__
    except:
        sklearn_version = "Not installed"
    
    try:
        import pandas
        pandas_version = pandas.__version__
    except:
        pandas_version = "Not installed"
    
    try:
        import numpy
        numpy_version = numpy.__version__
    except:
        numpy_version = "Not installed"
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'platform': {
            'system': platform.system(),
            'version': platform.version(),
            'machine': platform.machine(),
            'processor': platform.processor()
        },
        'python': {
            'version': sys.version,
            'executable': sys.executable
        },
        'packages': {
            'scikit-learn': sklearn_version,
            'pandas': pandas_version,
            'numpy': numpy_version,
            'joblib': joblib.__version__ if hasattr(joblib, '__version__') else "Unknown"
        },
        'working_directory': os.getcwd(),
        'model_files': [f for f in os.listdir('.') if f.endswith('.pkl')]
    }
    
    # Save report
    with open('environment_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print("✅ Environment report saved: environment_report.json")
    
    return report

def main():
    """Main validation pipeline."""
    
    print("🔧 Model Validation and Environment Check")
    print("=" * 50)
    
    # Check for model files
    model_files = [
        'heart_disease_render_optimized.pkl',
        'heart_disease_production_unified.pkl',
        'heart_disease_model_with_gender.pkl'
    ]
    
    valid_models = []
    
    for model_file in model_files:
        if os.path.exists(model_file):
            result = validate_model_file(model_file)
            if result and result != False:
                if len(result) == 3:  # Returns (True, hash, model_data)
                    valid_models.append((model_file, result[1], result[2]))
                    
                    # Test consistency for the first valid model
                    if len(valid_models) == 1:
                        consistency_result = test_prediction_consistency(result[2])
                        if consistency_result and consistency_result != False:
                            if len(consistency_result) == 2:
                                print(f"📊 Baseline predictions from {model_file}:")
                                for i, pred in enumerate(consistency_result[1]):
                                    print(f"   Test {i+1}: {pred['prediction']} - {pred['risk_percentage']} (Confidence: {pred['confidence']})")
    
    # Create environment report
    env_report = create_environment_report()
    
    print(f"\n" + "=" * 50)
    if valid_models:
        print(f"✅ Found {len(valid_models)} valid model(s)")
        print(f"🎯 Primary model: {valid_models[0][0]}")
        print(f"🔗 File hash: {valid_models[0][1][:16]}...")
        print(f"\n📋 Environment: {env_report['platform']['system']} {env_report['platform']['machine']}")
        print(f"🐍 Python: {env_report['python']['version'].split()[0]}")
        print(f"📦 Scikit-learn: {env_report['packages']['scikit-learn']}")
    else:
        print("❌ No valid models found!")
        print("🔧 Please run create_unified_model.py first")
    
    return len(valid_models) > 0

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)