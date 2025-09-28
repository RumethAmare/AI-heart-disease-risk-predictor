#!/usr/bin/env python3
"""
DIAGNOSTIC SCRIPT - Find Out What's Wrong!
This will check every component and tell you exactly what to fix.
"""

import os
import pandas as pd
import joblib
from flask import Flask

def check_files():
    """Check if all required files exist"""
    print("🔍 CHECKING FILES...")
    print("=" * 50)
    
    required_files = [
        'heart_disease.csv',
        'heart_disease_extended.csv', 
        'heart_disease_model_FIXED.pkl',
        'app.py',
        'templates/index.html',
        'static/css/style.css',
        'static/js/app.js'
    ]
    
    all_good = True
    for file in required_files:
        if os.path.exists(file):
            size = os.path.getsize(file)
            print(f"✅ {file} - {size:,} bytes")
        else:
            print(f"❌ {file} - MISSING!")
            all_good = False
    
    return all_good

def check_dataset():
    """Check the dataset"""
    print("\n📊 CHECKING DATASET...")
    print("=" * 50)
    
    try:
        df = pd.read_csv('heart_disease.csv')
        print(f"✅ Dataset loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")
        
        if 'Heart Disease Status' in df.columns:
            dist = df['Heart Disease Status'].value_counts()
            print(f"✅ Target distribution: {dist.to_dict()}")
            
            balance_ratio = dist.max() / dist.min()
            if balance_ratio <= 2:
                print(f"✅ Well balanced: {balance_ratio:.1f}:1 ratio")
            else:
                print(f"⚠️ Imbalanced: {balance_ratio:.1f}:1 ratio")
        else:
            print("❌ Target column 'Heart Disease Status' not found!")
            return False
            
        return True
    except Exception as e:
        print(f"❌ Dataset error: {e}")
        return False

def check_model():
    """Check the trained model"""
    print("\n🤖 CHECKING MODEL...")
    print("=" * 50)
    
    model_files = [
        'heart_disease_model_FIXED.pkl',
        'heart_disease_model.pkl'
    ]
    
    for model_file in model_files:
        try:
            if os.path.exists(model_file):
                model_data = joblib.load(model_file)
                print(f"✅ {model_file} loaded successfully")
                
                if 'model' in model_data:
                    print(f"   Model type: {type(model_data['model']).__name__}")
                
                if 'training_info' in model_data:
                    info = model_data['training_info']
                    if 'original_f1_score' in info:
                        f1 = info['original_f1_score']
                        print(f"   F1-Score: {f1:.4f}")
                        if f1 > 0.5:
                            print("   ✅ Good performance!")
                        elif f1 > 0.1:
                            print("   ⚠️ Moderate performance")
                        else:
                            print("   ❌ Poor performance")
                
                return True
            else:
                print(f"❌ {model_file} not found")
        except Exception as e:
            print(f"❌ {model_file} error: {e}")
    
    return False

def test_prediction():
    """Test making a prediction"""
    print("\n🧪 TESTING PREDICTION...")
    print("=" * 50)
    
    try:
        # Load the best available model
        model_data = None
        if os.path.exists('heart_disease_model_FIXED.pkl'):
            model_data = joblib.load('heart_disease_model_FIXED.pkl')
            print("✅ Using FIXED model")
        elif os.path.exists('heart_disease_model.pkl'):
            model_data = joblib.load('heart_disease_model.pkl')
            print("⚠️ Using fallback model")
        else:
            print("❌ No model found!")
            return False
        
        # Test sample
        sample = {
            'Age': 55.0, 'Gender': 'Male', 'Blood Pressure': 140.0,
            'Cholesterol Level': 220.0, 'Exercise Habits': 'Low', 'Smoking': 'Yes',
            'Family Heart Disease': 'Yes', 'Diabetes': 'No', 'BMI': 28.5,
            'High Blood Pressure': 'Yes', 'Low HDL Cholesterol': 'No',
            'High LDL Cholesterol': 'Yes', 'Alcohol Consumption': 'Medium',
            'Stress Level': 'High', 'Sleep Hours': 6.0, 'Sugar Consumption': 'High',
            'Triglyceride Level': 180.0, 'Fasting Blood Sugar': 110.0,
            'CRP Level': 3.5, 'Homocysteine Level': 12.0
        }
        
        # Make prediction using the loaded model components
        df = pd.DataFrame([sample])
        
        # Apply encoders if available
        if 'label_encoders' in model_data:
            encoders = model_data['label_encoders']
            for col, encoder in encoders.items():
                if col in df.columns:
                    try:
                        df[col] = encoder.transform([str(df[col].iloc[0])])[0]
                    except:
                        df[col] = 0  # Default value
        
        # Make prediction
        model = model_data['model']
        prediction = model.predict(df)[0]
        probabilities = model.predict_proba(df)[0]
        
        # Decode result
        if 'target_encoder' in model_data:
            result = model_data['target_encoder'].inverse_transform([prediction])[0]
        else:
            result = "Yes" if prediction == 1 else "No"
        
        confidence = max(probabilities) * 100
        
        print(f"✅ Prediction successful!")
        print(f"   Result: {result}")
        print(f"   Confidence: {confidence:.1f}%")
        print(f"   Probabilities: {probabilities}")
        
        return True
        
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        return False

def test_flask_app():
    """Test if Flask app can start"""
    print("\n🌐 TESTING FLASK APP...")
    print("=" * 50)
    
    try:
        # Import and check app
        from app import app, initialize_model
        print("✅ Flask app imported successfully")
        
        # Try to initialize model
        initialize_model()
        print("✅ Model initialization successful")
        
        print("✅ Flask app ready to run!")
        return True
        
    except Exception as e:
        print(f"❌ Flask app error: {e}")
        return False

def provide_solution():
    """Provide step-by-step solution"""
    print("\n🔧 STEP-BY-STEP SOLUTION:")
    print("=" * 50)
    
    print("1. 🚀 START THE WEB APPLICATION:")
    print("   Command: python start_system.py")
    print("   Or: python app.py")
    print("")
    
    print("2. 🌐 OPEN YOUR WEB BROWSER:")
    print("   URL: http://localhost:5000")
    print("")
    
    print("3. 📝 FILL OUT THE FORM:")
    print("   • Enter Age (e.g., 55)")
    print("   • Select Gender (Male/Female)")
    print("   • Enter Blood Pressure (e.g., 140)")
    print("   • Enter Cholesterol (e.g., 220)")
    print("   • Select Smoking (Yes/No)")
    print("   • Enter BMI (e.g., 28.5)")
    print("")
    
    print("4. 🎯 CLICK 'ANALYZE RISK':")
    print("   • Wait for processing (button shows 'Processing...')")
    print("   • Results will show below the form")
    print("")
    
    print("5. 📊 EXPECTED OUTPUT:")
    print("   • Risk Level: Low/Medium/High")
    print("   • Risk Percentage: XX.X%")
    print("   • Confidence Level: XX.X%")
    print("   • Health Recommendations")

def main():
    """Run complete diagnostic"""
    print("🩺 HEART DISEASE PREDICTION SYSTEM - DIAGNOSTIC")
    print("=" * 70)
    print("Let's find out what's wrong and fix it!")
    
    # Run all checks
    files_ok = check_files()
    dataset_ok = check_dataset() 
    model_ok = check_model()
    prediction_ok = test_prediction()
    flask_ok = test_flask_app()
    
    # Summary
    print("\n📋 DIAGNOSTIC SUMMARY:")
    print("=" * 50)
    print(f"Files: {'✅' if files_ok else '❌'}")
    print(f"Dataset: {'✅' if dataset_ok else '❌'}")
    print(f"Model: {'✅' if model_ok else '❌'}")
    print(f"Prediction: {'✅' if prediction_ok else '❌'}")
    print(f"Flask App: {'✅' if flask_ok else '❌'}")
    
    if all([files_ok, dataset_ok, model_ok, prediction_ok, flask_ok]):
        print("\n🎉 EVERYTHING LOOKS GOOD!")
        print("Your system should work perfectly.")
        provide_solution()
    else:
        print("\n⚠️ ISSUES FOUND!")
        print("Follow the solution steps below to fix them.")
        provide_solution()

if __name__ == "__main__":
    main()