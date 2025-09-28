#!/usr/bin/env python3
"""
Confidence Calculation Explanation
Shows exactly how confidence is calculated in the heart disease prediction model
"""

import pandas as pd
import numpy as np
from simple_model_wrapper import SimpleHeartDiseasePredictor

def explain_confidence_calculation():
    """Explain step-by-step how confidence is calculated"""
    
    print("🔍 CONFIDENCE CALCULATION EXPLANATION")
    print("=" * 60)
    print("Understanding how the confidence percentage is computed...\n")
    
    # Load the predictor
    predictor = SimpleHeartDiseasePredictor()
    if not predictor.load_model('heart_disease_model.pkl'):
        print("❌ Could not load model")
        return
    
    # Sample inputs to demonstrate
    test_cases = [
        {
            'name': 'Low Risk Patient',
            'data': {
                'Age': 30, 'Gender': 'Female', 'Blood Pressure': 110,
                'Cholesterol Level': 150, 'Exercise Habits': 'High',
                'Smoking': 'No', 'Family Heart Disease': 'No',
                'Diabetes': 'No', 'BMI': 22, 'High Blood Pressure': 'No',
                'Low HDL Cholesterol': 'No', 'High LDL Cholesterol': 'No',
                'Alcohol Consumption': 'Low', 'Stress Level': 'Low',
                'Sleep Hours': 8, 'Sugar Consumption': 'Low',
                'Triglyceride Level': 120, 'Fasting Blood Sugar': 90,
                'CRP Level': 1, 'Homocysteine Level': 8
            }
        },
        {
            'name': 'Medium Risk Patient',
            'data': {
                'Age': 55, 'Gender': 'Male', 'Blood Pressure': 140,
                'Cholesterol Level': 200, 'Exercise Habits': 'Medium',
                'Smoking': 'No', 'Family Heart Disease': 'No',
                'Diabetes': 'No', 'BMI': 28, 'High Blood Pressure': 'No',
                'Low HDL Cholesterol': 'No', 'High LDL Cholesterol': 'No',
                'Alcohol Consumption': 'Medium', 'Stress Level': 'Medium',
                'Sleep Hours': 7, 'Sugar Consumption': 'Medium',
                'Triglyceride Level': 150, 'Fasting Blood Sugar': 100,
                'CRP Level': 2, 'Homocysteine Level': 10
            }
        },
        {
            'name': 'High Risk Patient',
            'data': {
                'Age': 70, 'Gender': 'Male', 'Blood Pressure': 180,
                'Cholesterol Level': 300, 'Exercise Habits': 'Low',
                'Smoking': 'Yes', 'Family Heart Disease': 'Yes',
                'Diabetes': 'Yes', 'BMI': 35, 'High Blood Pressure': 'Yes',
                'Low HDL Cholesterol': 'Yes', 'High LDL Cholesterol': 'Yes',
                'Alcohol Consumption': 'High', 'Stress Level': 'High',
                'Sleep Hours': 5, 'Sugar Consumption': 'High',
                'Triglyceride Level': 250, 'Fasting Blood Sugar': 130,
                'CRP Level': 5, 'Homocysteine Level': 15
            }
        }
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"📊 CASE {i}: {case['name']}")
        print("-" * 50)
        
        # Simulate the internal calculation steps
        df = pd.DataFrame([case['data']])
        
        # Get model components
        model_data = predictor.model_data
        model = model_data['model']
        label_encoders = model_data.get('label_encoders', {})
        
        # Encode categorical variables (simplified)
        for col, encoder in label_encoders.items():
            if col in df.columns:
                try:
                    val = str(df[col].iloc[0])
                    if val in encoder.classes_:
                        df[col] = encoder.transform([val])[0]
                    else:
                        df[col] = 0
                except:
                    df[col] = 0
        
        # Get expected features
        expected_features = model_data.get('feature_columns', df.columns)
        for col in expected_features:
            if col not in df.columns:
                df[col] = 0
        df = df[expected_features]
        
        # Get raw probabilities from ML model
        raw_probabilities = model.predict_proba(df)[0]
        
        print(f"🤖 Step 1: ML Model Raw Output")
        print(f"   • Raw probabilities: {raw_probabilities}")
        print(f"   • P(No Heart Disease):  {raw_probabilities[0]:.6f} ({raw_probabilities[0]*100:.2f}%)")
        print(f"   • P(Yes Heart Disease): {raw_probabilities[1]:.6f} ({raw_probabilities[1]*100:.2f}%)")
        
        # Apply the manual modification you made
        modified_risk_prob = 1.3 * raw_probabilities[1]  # Your 30% boost
        
        print(f"\n🔧 Step 2: Your Manual Modification")
        print(f"   • Original risk probability: {raw_probabilities[1]:.6f}")
        print(f"   • Modified risk probability: 1.3 × {raw_probabilities[1]:.6f} = {modified_risk_prob:.6f}")
        print(f"   • Effect: {((modified_risk_prob - raw_probabilities[1]) / raw_probabilities[1] * 100):+.1f}% increase")
        
        # Calculate confidence - THIS IS THE KEY PART
        confidence_value = max(raw_probabilities)  # Maximum of the two probabilities
        
        print(f"\n🎯 Step 3: Confidence Calculation")
        print(f"   *** THIS IS HOW CONFIDENCE IS CALCULATED ***")
        print(f"   • confidence = max(probabilities)")
        print(f"   • confidence = max({raw_probabilities[0]:.6f}, {raw_probabilities[1]:.6f})")
        print(f"   • confidence = {confidence_value:.6f}")
        print(f"   • Confidence percentage: {confidence_value * 100:.1f}%")
        
        print(f"\n📋 Step 4: What This Means")
        if confidence_value == raw_probabilities[0]:
            print(f"   • The model is {confidence_value*100:.1f}% confident in predicting 'No Heart Disease'")
            print(f"   • This is because P(No) > P(Yes)")
        else:
            print(f"   • The model is {confidence_value*100:.1f}% confident in predicting 'Yes Heart Disease'")
            print(f"   • This is because P(Yes) > P(No)")
        
        # Show final formatted result
        confidence_formatted = f"{confidence_value * 100:.1f}%"
        print(f"\n📱 Step 5: Final Output")
        print(f"   • Formatted confidence: '{confidence_formatted}'")
        print(f"   • This appears in the web interface as the confidence value")
        
        print("\n" + "="*60 + "\n")
    
    return raw_probabilities, confidence_value

def explain_confidence_concept():
    """Explain what confidence means conceptually"""
    
    print("🧠 WHAT CONFIDENCE REALLY MEANS")
    print("=" * 60)
    
    print("📖 Conceptual Explanation:")
    print("   Confidence = How sure the model is about its prediction")
    print("   Formula: max(P(No), P(Yes))")
    print()
    
    examples = [
        {
            "probabilities": [0.8, 0.2],
            "prediction": "No",
            "confidence": 0.8,
            "meaning": "Very confident it's 'No'"
        },
        {
            "probabilities": [0.6, 0.4], 
            "prediction": "No",
            "confidence": 0.6,
            "meaning": "Somewhat confident it's 'No'"
        },
        {
            "probabilities": [0.51, 0.49],
            "prediction": "No", 
            "confidence": 0.51,
            "meaning": "Barely confident it's 'No'"
        },
        {
            "probabilities": [0.3, 0.7],
            "prediction": "Yes",
            "confidence": 0.7,
            "meaning": "Confident it's 'Yes'"
        }
    ]
    
    print("📊 Examples of Confidence Calculation:")
    for i, ex in enumerate(examples, 1):
        p_no, p_yes = ex["probabilities"]
        print(f"\n   Example {i}:")
        print(f"   • P(No) = {p_no:.2f}, P(Yes) = {p_yes:.2f}")
        print(f"   • Prediction: {ex['prediction']} (higher probability)")
        print(f"   • Confidence: max({p_no:.2f}, {p_yes:.2f}) = {ex['confidence']:.2f} = {ex['confidence']*100:.0f}%")
        print(f"   • Meaning: {ex['meaning']}")
    
    print(f"\n🎯 Key Points:")
    print("   • High confidence (>80%): Model is very sure")
    print("   • Medium confidence (60-80%): Model is moderately sure") 
    print("   • Low confidence (<60%): Model is uncertain")
    print("   • 50% confidence: Model can't decide (equal probabilities)")

def show_confidence_in_code():
    """Show exactly where confidence appears in the code"""
    
    print(f"\n💻 CONFIDENCE IN THE CODE")
    print("=" * 60)
    
    print("📍 Location 1: simple_model_wrapper.py (Line ~87)")
    print("   Code: confidence = max(probabilities)")
    print("   Purpose: Calculate confidence value")
    print()
    
    print("📍 Location 2: simple_model_wrapper.py (Line ~101)")
    print("   Code: 'confidence': f\"{confidence * 100:.1f}%\"")
    print("   Purpose: Format confidence as percentage string")
    print()
    
    print("📍 Location 3: app.py (Flask API)")
    print("   Code: result['confidence']")
    print("   Purpose: Send confidence to web interface")
    print()
    
    print("🔍 The Exact Line:")
    print("   confidence = max(probabilities)")
    print("   └── This takes the maximum of [P(No), P(Yes)]")
    print("   └── Represents how confident the model is in its prediction")

def main():
    """Main explanation function"""
    
    # Explain confidence calculation with examples
    probs, conf = explain_confidence_calculation()
    
    # Explain the concept
    explain_confidence_concept()
    
    # Show code locations
    show_confidence_in_code()
    
    # Final summary
    print(f"\n🎯 CONFIDENCE CALCULATION SUMMARY")
    print("=" * 60)
    print("Confidence is calculated with this simple formula:")
    print()
    print("   confidence = max(probabilities)")
    print("   confidence = max([P(No Heart Disease), P(Yes Heart Disease)])")
    print()
    print("Example:")
    print(f"   • If probabilities are [0.65, 0.35]")
    print(f"   • Then confidence = max(0.65, 0.35) = 0.65 = 65%")
    print(f"   • Meaning: 65% confident the prediction is correct")
    print()
    print("🔑 Key Insight:")
    print("   Confidence tells you how certain the model is,")
    print("   regardless of whether it predicts 'Yes' or 'No'!")

if __name__ == "__main__":
    main()