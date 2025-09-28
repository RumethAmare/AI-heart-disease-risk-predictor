#!/usr/bin/env python3
"""
Percentage Generation Trace
Shows exactly where and how the heart disease risk percentage is calculated
"""

import pandas as pd
import numpy as np
from simple_model_wrapper import SimpleHeartDiseasePredictor
import joblib

def trace_percentage_generation():
    """Trace step-by-step how the percentage is generated"""
    
    print("🔍 TRACING PERCENTAGE GENERATION")
    print("=" * 60)
    print("Following the complete flow from input to percentage output...\n")
    
    # Load the predictor to examine its components
    predictor = SimpleHeartDiseasePredictor()
    if not predictor.load_model('heart_disease_model.pkl'):
        print("❌ Could not load model")
        return
    
    # Example input data
    sample_input = {
        'Age': 55,
        'Gender': 'Male',
        'Blood Pressure': 140,
        'Cholesterol Level': 200,
        'Exercise Habits': 'Medium',
        'Smoking': 'No',
        'Family Heart Disease': 'No',
        'Diabetes': 'No',
        'BMI': 28,
        'High Blood Pressure': 'No',
        'Low HDL Cholesterol': 'No',
        'High LDL Cholesterol': 'No',
        'Alcohol Consumption': 'Medium',
        'Stress Level': 'Medium',
        'Sleep Hours': 7,
        'Sugar Consumption': 'Medium',
        'Triglyceride Level': 150,
        'Fasting Blood Sugar': 100,
        'CRP Level': 2,
        'Homocysteine Level': 10
    }
    
    print("📋 STEP-BY-STEP PERCENTAGE GENERATION:")
    print("=" * 50)
    
    print("🔹 STEP 1: Input Data")
    print("   Raw input from user (age, gender, blood pressure, etc.)")
    for key, value in list(sample_input.items())[:6]:
        print(f"   • {key}: {value}")
    print("   • ... (and 14 more attributes)")
    
    print("\n🔹 STEP 2: Data Preprocessing")
    print("   Converting input to DataFrame and encoding categorical variables...")
    
    # Simulate the preprocessing steps
    df = pd.DataFrame([sample_input])
    model_data = predictor.model_data
    model = model_data['model']
    label_encoders = model_data.get('label_encoders', {})
    
    print("   • Converting to DataFrame: ✓")
    print("   • Encoding categorical variables:")
    
    for col, encoder in label_encoders.items():
        if col in df.columns:
            original_val = df[col].iloc[0]
            try:
                encoded_val = encoder.transform([str(original_val)])[0]
                df[col] = encoded_val
                print(f"     - {col}: '{original_val}' → {encoded_val}")
            except:
                df[col] = 0
                print(f"     - {col}: '{original_val}' → 0 (default)")
    
    print("\n🔹 STEP 3: Machine Learning Model Prediction")
    print("   Feeding processed data into trained RandomForest model...")
    
    # Get the raw probabilities from the ML model
    expected_features = model_data.get('feature_columns', df.columns)
    df_model = df[expected_features]
    
    print(f"   • Input features: {len(expected_features)} numerical values")
    print(f"   • Model type: {type(model).__name__}")
    
    # This is the KEY step where percentages are generated
    # Make prediction
    probabilities = model.predict_proba(df)[0]
    risk_probability = probabilities[1]  # Probability of "Yes"
    
    print(f"\n🎯 STEP 4: RAW PROBABILITY GENERATION")
    print("   *** THIS IS WHERE THE PERCENTAGE COMES FROM ***")
    print(f"   • ML Model Output: {probabilities}")
    print(f"     - Probability of 'No':  {probabilities[0]:.6f} ({probabilities[0]*100:.2f}%)")
    print(f"     - Probability of 'Yes': {probabilities[1]:.6f} ({probabilities[1]*100:.2f}%)")
    print(f"   • Selected Risk Probability: {risk_probability:.6f}")
    
    print(f"\n🔹 STEP 5: Percentage Conversion")
    print("   Converting probability to percentage format...")
    
    risk_percentage_raw = risk_probability * 100
    risk_percentage_formatted = f"{risk_probability * 100:.1f}%"
    
    print(f"   • Raw calculation: {risk_probability:.6f} × 100 = {risk_percentage_raw:.2f}")
    print(f"   • Formatted result: '{risk_percentage_formatted}'")
    
    print(f"\n🔹 STEP 6: Threshold Classification")
    print("   Applying thresholds to determine risk level...")
    
    if risk_probability >= 0.65:
        risk_level = "High"
        prediction = "Yes"
    elif risk_probability >= 0.45:
        risk_level = "Medium-High"
        prediction = "Yes"
    elif risk_probability >= 0.35:
        risk_level = "Medium"
        prediction = "No"
    else:
        risk_level = "Low"
        prediction = "No"
    
    print(f"   • Probability: {risk_probability:.3f}")
    print(f"   • Threshold check:")
    print(f"     - ≥0.65 (High): {'✓' if risk_probability >= 0.65 else '✗'}")
    print(f"     - ≥0.45 (Med-High): {'✓' if risk_probability >= 0.45 else '✗'}")
    print(f"     - ≥0.35 (Medium): {'✓' if risk_probability >= 0.35 else '✗'}")
    print(f"   • Final Classification: {risk_level} → {prediction}")
    
    print(f"\n🔹 STEP 7: Final Output")
    print("   Packaging results for return...")
    
    result = {
        'prediction': prediction,
        'risk_probability': float(risk_probability),
        'risk_percentage': risk_percentage_formatted,
        'risk_level': risk_level
    }
    
    for key, value in result.items():
        print(f"   • {key}: {value}")
    
    return risk_probability, probabilities

def examine_ml_model_internals():
    """Examine what's inside the ML model that generates probabilities"""
    
    print(f"\n🔬 EXAMINING ML MODEL INTERNALS")
    print("=" * 60)
    
    try:
        # Load model data directly
        model_data = joblib.load('heart_disease_model.pkl')
        model = model_data['model']
        
        print(f"📊 Model Details:")
        print(f"   • Type: {type(model).__name__}")
        print(f"   • Number of trees: {model.n_estimators}")
        print(f"   • Classes: {model.classes_}")
        print(f"   • Number of features: {model.n_features_in_}")
        
        print(f"\n🌳 How RandomForest Generates Probabilities:")
        print("   1. Each tree in the forest makes a prediction (0 or 1)")
        print("   2. Count how many trees predict each class")
        print("   3. Probability = (trees predicting class) / (total trees)")
        print("   4. Example: If 30/100 trees predict 'Yes' → 30% probability")
        
        print(f"\n📈 Feature Importance (Top 10):")
        if hasattr(model, 'feature_importances_'):
            feature_names = model_data.get('feature_columns', [f'Feature_{i}' for i in range(len(model.feature_importances_))])
            importance_pairs = list(zip(feature_names, model.feature_importances_))
            importance_pairs.sort(key=lambda x: x[1], reverse=True)
            
            for i, (feature, importance) in enumerate(importance_pairs[:10]):
                print(f"   {i+1:2d}. {feature}: {importance:.4f}")
        
    except Exception as e:
        print(f"❌ Error examining model: {str(e)}")

def show_percentage_sources():
    """Show all the places where percentages are used"""
    
    print(f"\n📍 WHERE PERCENTAGES APPEAR IN THE SYSTEM")
    print("=" * 60)
    
    sources = [
        {
            "Location": "simple_model_wrapper.py (Line ~97)",
            "Code": "'risk_percentage': f\"{risk_probability * 100:.1f}%\"",
            "Description": "Main percentage calculation - converts ML probability to %"
        },
        {
            "Location": "simple_model_wrapper.py (Line ~98)", 
            "Code": "'confidence': f\"{confidence * 100:.1f}%\"",
            "Description": "Confidence percentage - based on max probability"
        },
        {
            "Location": "app.py (Flask API)",
            "Code": "result['risk_percentage']",
            "Description": "API endpoint returns percentage to frontend"
        },
        {
            "Location": "Frontend JavaScript",
            "Code": "response.risk_percentage",
            "Description": "Displays percentage in web interface"
        }
    ]
    
    for i, source in enumerate(sources, 1):
        print(f"\n📌 Source {i}: {source['Location']}")
        print(f"   Code: {source['Code']}")
        print(f"   Purpose: {source['Description']}")

def main():
    """Main function to trace percentage generation"""
    
    # Trace the complete flow
    probability, raw_probs = trace_percentage_generation()
    
    # Examine model internals
    examine_ml_model_internals()
    
    # Show where percentages are used
    show_percentage_sources()
    
    # Summary
    print(f"\n🎯 SUMMARY: WHERE THE PERCENTAGE COMES FROM")
    print("=" * 60)
    print("The percentage is generated through this exact process:")
    print()
    print("1. 🤖 RandomForest ML model processes input features")
    print("2. 🌳 Each tree votes: 'Yes' or 'No' for heart disease")  
    print("3. 📊 Probability = (trees voting 'Yes') / (total trees)")
    print("4. 🔢 Percentage = probability × 100")
    print("5. 📱 Result: \"35.4%\" means 35.4% of trees predicted heart disease")
    print()
    print("🔑 KEY INSIGHT:")
    print(f"   The {probability*100:.1f}% you see is the RAW output from the")
    print("   trained RandomForest algorithm, converted to percentage format.")
    print("   It represents the model's confidence in heart disease prediction.")

if __name__ == "__main__":
    main()