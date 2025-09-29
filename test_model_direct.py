#!/usr/bin/env python3
"""
Direct Model Wrapper Test - Check raw model probabilities
"""

from simple_model_wrapper import SimpleHeartDiseasePredictor
import pandas as pd

def test_model_directly():
    """Test the model wrapper directly to see raw probabilities"""
    
    print("🔍 Direct Model Wrapper Test")
    print("=" * 50)
    
    # Load the model
    predictor = SimpleHeartDiseasePredictor()
    success = predictor.load_model('heart_disease_model.pkl')
    
    if not success:
        print("❌ Failed to load model")
        return
    
    print("✅ Model loaded successfully")
    
    # High-risk patient test case
    high_risk_data = {
        'Age': 65,
        'Gender': 'Male',
        'Blood Pressure': 160,
        'Cholesterol Level': 280,
        'Exercise Habits': 'Low',
        'Smoking': 'Yes',
        'Family Heart Disease': 'Yes',
        'Diabetes': 'Yes',
        'BMI': 32.0,
        'High Blood Pressure': 'Yes',
        'Low HDL Cholesterol': 'Yes',
        'High LDL Cholesterol': 'Yes',
        'Alcohol Consumption': 'Heavy',
        'Stress Level': 'High',
        'Sleep Hours': 5.0,
        'Sugar Consumption': 'High',
        'Triglyceride Level': 200,
        'Fasting Blood Sugar': 130,
        'CRP Level': 5.0,
        'Homocysteine Level': 15.0
    }
    
    print(f"\n📊 Testing High-Risk Patient:")
    for key, value in high_risk_data.items():
        print(f"   {key}: {value}")
    
    # Make prediction
    result = predictor.predict(high_risk_data)
    
    print(f"\n📋 Model Wrapper Results:")
    for key, value in result.items():
        print(f"   {key}: {value}")
    
    # Calculate what the raw probabilities would be
    raw_prob = result['risk_probability'] / 1.3
    print(f"\n🔍 Analysis:")
    print(f"   Raw model probability (before 1.3x boost): {raw_prob:.3f} ({raw_prob*100:.1f}%)")
    print(f"   Boosted probability: {result['risk_probability']:.3f} ({result['risk_probability']*100:.1f}%)")
    
    if raw_prob < 0.25:
        print(f"   ❌ Issue: Raw model probability is very low for high-risk patient")
    elif raw_prob < 0.4:
        print(f"   ⚠️ Warning: Raw model probability is lower than expected")
    else:
        print(f"   ✅ Raw model probability seems reasonable")
    
    # Test low-risk patient
    print(f"\n" + "=" * 50)
    low_risk_data = {
        'Age': 25,
        'Gender': 'Female',
        'Blood Pressure': 110,
        'Cholesterol Level': 180,
        'Exercise Habits': 'High',
        'Smoking': 'Never',
        'Family Heart Disease': 'No',
        'Diabetes': 'No',
        'BMI': 22.0,
        'High Blood Pressure': 'No',
        'Low HDL Cholesterol': 'No',
        'High LDL Cholesterol': 'No',
        'Alcohol Consumption': 'None',
        'Stress Level': 'Low',
        'Sleep Hours': 8.0,
        'Sugar Consumption': 'Low',
        'Triglyceride Level': 100,
        'Fasting Blood Sugar': 85,
        'CRP Level': 0.5,
        'Homocysteine Level': 8.0
    }
    
    print(f"📊 Testing Low-Risk Patient:")
    result_low = predictor.predict(low_risk_data)
    raw_prob_low = result_low['risk_probability'] / 1.3
    
    print(f"   Raw probability: {raw_prob_low:.3f} ({raw_prob_low*100:.1f}%)")
    print(f"   Boosted probability: {result_low['risk_probability']:.3f} ({result_low['risk_probability']*100:.1f}%)")
    print(f"   Prediction: {result_low['prediction']}")
    
    # Check if there's good separation between high and low risk
    prob_difference = result['risk_probability'] - result_low['risk_probability']
    print(f"\n📈 Model Discrimination:")
    print(f"   Probability difference: {prob_difference:.3f} ({prob_difference*100:.1f} percentage points)")
    
    if prob_difference > 0.2:
        print(f"   ✅ Good separation between high and low risk patients")
    else:
        print(f"   ❌ Poor separation - model may need retraining or different approach")

if __name__ == "__main__":
    test_model_directly()