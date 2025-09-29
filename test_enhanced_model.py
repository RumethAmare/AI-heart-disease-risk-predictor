#!/usr/bin/env python3
"""
Test the Enhanced Model with High-Risk and Low-Risk Cases
"""

from enhanced_model_wrapper import EnhancedHeartDiseasePredictor
import json

# Initialize the enhanced predictor
predictor = EnhancedHeartDiseasePredictor()

# Load the model
if predictor.load_model('heart_disease_model.pkl'):
    print("✅ Enhanced model loaded successfully")
else:
    print("❌ Failed to load model - continuing with clinical-only predictions")

print("\n" + "="*80)
print("TESTING ENHANCED HEART DISEASE PREDICTION MODEL")
print("="*80)

# Test Case 1: High-Risk Patient
print("\n🔴 HIGH-RISK PATIENT TEST:")
print("65-year-old male, diabetes, smoking, high BP, high cholesterol, BMI 32")

high_risk_patient = {
    'Age': 65,
    'Gender': 'Male',
    'Blood Pressure': 165,
    'Cholesterol Level': 280,
    'BMI': 32,
    'Smoking': 'Yes',
    'Diabetes': 'Yes',
    'Family Heart Disease': 'Yes',
    'Exercise Habits': 'Low',
    'High Blood Pressure': 'Yes',
    'High LDL Cholesterol': 'Yes',
    'Low HDL Cholesterol': 'Yes',
    'Stress Level': 'High',
    'Alcohol Consumption': 'Moderate',
    'Sugar Consumption': 'High',
    'Sleep Hours': 5,
    'CRP Level': 4.2,
    'Homocysteine Level': 16.0,
    'Triglyceride Level': 220,
    'Fasting Blood Sugar': 140
}

result_high = predictor.predict(high_risk_patient)
print(f"Prediction: {result_high['prediction']}")
print(f"Risk Probability: {result_high['risk_percentage']}")
print(f"Risk Level: {result_high['risk_level']}")
print(f"Clinical Score: {result_high['clinical_score']}/100")
print(f"Confidence: {result_high['confidence']}")
print(f"Recommendation: {result_high['recommendation']}")

print("\n" + "-"*80)

# Test Case 2: Low-Risk Patient  
print("\n🟢 LOW-RISK PATIENT TEST:")
print("25-year-old female, no risk factors, excellent health")

low_risk_patient = {
    'Age': 25,
    'Gender': 'Female',
    'Blood Pressure': 110,
    'Cholesterol Level': 160,
    'BMI': 22,
    'Smoking': 'No',
    'Diabetes': 'No',
    'Family Heart Disease': 'No',
    'Exercise Habits': 'High',
    'High Blood Pressure': 'No',
    'High LDL Cholesterol': 'No',
    'Low HDL Cholesterol': 'No',
    'Stress Level': 'Low',
    'Alcohol Consumption': 'None',
    'Sugar Consumption': 'Low',
    'Sleep Hours': 8,
    'CRP Level': 0.5,
    'Homocysteine Level': 8.0,
    'Triglyceride Level': 80,
    'Fasting Blood Sugar': 85
}

result_low = predictor.predict(low_risk_patient)
print(f"Prediction: {result_low['prediction']}")
print(f"Risk Probability: {result_low['risk_percentage']}")
print(f"Risk Level: {result_low['risk_level']}")
print(f"Clinical Score: {result_low['clinical_score']}/100")
print(f"Confidence: {result_low['confidence']}")
print(f"Recommendation: {result_low['recommendation']}")

print("\n" + "="*80)
print("ENHANCED MODEL PERFORMANCE ANALYSIS")
print("="*80)

high_prob = float(result_high['risk_percentage'].replace('%', ''))
low_prob = float(result_low['risk_percentage'].replace('%', ''))
discrimination = high_prob - low_prob

print(f"\nHigh-Risk Patient Risk: {high_prob:.1f}%")
print(f"Low-Risk Patient Risk: {low_prob:.1f}%")
print(f"Risk Discrimination: {discrimination:.1f} percentage points")

if discrimination > 30:
    print("✅ EXCELLENT discrimination between high and low risk patients")
elif discrimination > 15:
    print("✅ GOOD discrimination between high and low risk patients")  
elif discrimination > 5:
    print("⚠️ MODERATE discrimination - could be improved")
else:
    print("❌ POOR discrimination - model needs significant improvement")

print(f"\nHigh-Risk Clinical Expectation: Should be >60% risk")
print(f"Low-Risk Clinical Expectation: Should be <20% risk")

if high_prob >= 60 and low_prob <= 20:
    print("✅ Enhanced model meets clinical expectations!")
else:
    print("⚠️ Enhanced model may need further calibration")

print(f"\n📊 Model Improvement Summary:")
print(f"   - Uses clinical risk scoring based on medical guidelines")  
print(f"   - Combines clinical assessment with ML predictions")
print(f"   - Provides realistic risk stratification")
print(f"   - Includes detailed recommendations")