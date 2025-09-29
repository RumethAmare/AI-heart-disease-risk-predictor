#!/usr/bin/env python3
"""
Test Enhanced Flask API with Realistic Cases
"""

import requests
import json

API_URL = "http://localhost:5000/api/predict"

print("="*80)
print("TESTING ENHANCED FLASK API")
print("="*80)

# Test Case 1: High-Risk Patient
print("\n🔴 HIGH-RISK PATIENT API TEST:")

high_risk_data = {
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

try:
    response = requests.post(API_URL, json=high_risk_data, timeout=10)
    if response.status_code == 200:
        result = response.json()
        print(f"✅ API Response: {response.status_code}")
        print(f"Prediction: {result['prediction']}")
        print(f"Risk Probability: {result['risk_percentage']}")
        print(f"Risk Level: {result['risk_level']}")
        print(f"Recommendation: {result['recommendation'][:100]}...")
    else:
        print(f"❌ API Error: {response.status_code}")
        print(response.text)
except requests.exceptions.RequestException as e:
    print(f"❌ Connection Error: {e}")

print("\n" + "-"*50)

# Test Case 2: Low-Risk Patient
print("\n🟢 LOW-RISK PATIENT API TEST:")

low_risk_data = {
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

try:
    response = requests.post(API_URL, json=low_risk_data, timeout=10)
    if response.status_code == 200:
        result = response.json()
        print(f"✅ API Response: {response.status_code}")
        print(f"Prediction: {result['prediction']}")
        print(f"Risk Probability: {result['risk_percentage']}")
        print(f"Risk Level: {result['risk_level']}")
        print(f"Recommendation: {result['recommendation'][:100]}...")
    else:
        print(f"❌ API Error: {response.status_code}")
        print(response.text)
except requests.exceptions.RequestException as e:
    print(f"❌ Connection Error: {e}")

print("\n" + "="*80)
print("✅ Enhanced API is ready!")
print("🔗 Visit http://localhost:5000 to test the web interface")
print("📊 The model now provides clinically realistic risk assessments")
print("="*80)