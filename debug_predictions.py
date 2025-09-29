#!/usr/bin/env python3
"""
Debug Prediction Output - Test what the model is actually returning
"""

import json
import requests
import sys

def test_prediction_output():
    """Test the prediction API with sample data"""
    
    # Sample high-risk patient data
    test_data = {
        "age": 65,
        "gender": "Male",
        "blood_pressure": 160,
        "cholesterol_level": 280,
        "smoking": "Yes",
        "bmi": 32.0,
        "exercise_habits": "Low",
        "family_heart_disease": "Yes",
        "diabetes": "Yes",
        "high_blood_pressure": "Yes",
        "low_hdl_cholesterol": "Yes",
        "high_ldl_cholesterol": "Yes",
        "alcohol_consumption": "Heavy",
        "stress_level": "High",
        "sugar_consumption": "High",
        "sleep_hours": 5.0,
        "triglyceride_level": 200,
        "fasting_blood_sugar": 130,
        "crp_level": 5.0,
        "homocysteine_level": 15.0
    }
    
    print("🧪 Testing Prediction API Output")
    print("=" * 50)
    print(f"📊 Input Data:")
    for key, value in test_data.items():
        print(f"   {key}: {value}")
    
    try:
        # Make API request
        url = "http://localhost:5000/api/predict"
        response = requests.post(url, json=test_data)
        
        if response.status_code == 200:
            result = response.json()
            
            print(f"\n✅ API Response (Status: {response.status_code})")
            print("=" * 50)
            
            # Pretty print the result
            for key, value in result.items():
                if key != 'input_data':  # Skip the echoed input data
                    print(f"📋 {key}: {value}")
            
            print(f"\n🔍 Analysis:")
            print(f"   • Risk Prediction: {result.get('prediction', 'N/A')}")
            print(f"   • Risk Probability: {result.get('risk_probability', 'N/A')}")
            print(f"   • Risk Percentage: {result.get('risk_percentage', 'N/A')}")
            print(f"   • Risk Level: {result.get('risk_level', 'N/A')}")
            print(f"   • Confidence: {result.get('confidence', 'N/A')}")
            print(f"   • Recommendation: {result.get('recommendation', 'N/A')}")
            
            # Check if this seems reasonable for a high-risk patient
            risk_prob = result.get('risk_probability', 0)
            if risk_prob > 0.7:
                print(f"\n✅ Result looks reasonable for high-risk patient (>{risk_prob:.1%})")
            elif risk_prob > 0.4:
                print(f"\n⚠️ Result seems moderate for high-risk patient ({risk_prob:.1%})")
            else:
                print(f"\n❌ Result seems too low for high-risk patient ({risk_prob:.1%})")
                
        else:
            print(f"\n❌ API Error (Status: {response.status_code})")
            print(f"Response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Connection Error: Make sure Flask server is running on localhost:5000")
    except Exception as e:
        print(f"❌ Error: {str(e)}")

def test_low_risk_patient():
    """Test with a low-risk patient"""
    
    test_data = {
        "age": 25,
        "gender": "Female",
        "blood_pressure": 110,
        "cholesterol_level": 180,
        "smoking": "Never",
        "bmi": 22.0,
        "exercise_habits": "High",
        "family_heart_disease": "No",
        "diabetes": "No",
        "high_blood_pressure": "No",
        "low_hdl_cholesterol": "No",
        "high_ldl_cholesterol": "No",
        "alcohol_consumption": "None",
        "stress_level": "Low",
        "sugar_consumption": "Low",
        "sleep_hours": 8.0,
        "triglyceride_level": 100,
        "fasting_blood_sugar": 85,
        "crp_level": 0.5,
        "homocysteine_level": 8.0
    }
    
    print("\n🧪 Testing Low-Risk Patient")
    print("=" * 50)
    
    try:
        url = "http://localhost:5000/api/predict"
        response = requests.post(url, json=test_data)
        
        if response.status_code == 200:
            result = response.json()
            risk_prob = result.get('risk_probability', 0)
            
            print(f"📋 Low-Risk Patient Result:")
            print(f"   • Risk Prediction: {result.get('prediction', 'N/A')}")
            print(f"   • Risk Probability: {risk_prob:.1%}")
            print(f"   • Risk Level: {result.get('risk_level', 'N/A')}")
            
            if risk_prob < 0.3:
                print(f"✅ Good - Low risk patient shows low probability")
            else:
                print(f"❌ Issue - Low risk patient shows high probability")
                
    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    test_prediction_output()
    test_low_risk_patient()