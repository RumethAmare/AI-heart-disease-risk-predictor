#!/usr/bin/env python3
"""
Test script to validate all fallback scenarios work correctly
"""

import requests
import json

def test_api_prediction():
    """Test the API prediction endpoint"""
    url = "http://localhost:5000/api/predict"
    
    test_data = {
        "age": 45,
        "gender": "Male", 
        "blood_pressure": 140,
        "cholesterol_level": 220,
        "bmi": 28,
        "exercise_habits": "Medium"
    }
    
    try:
        response = requests.post(url, json=test_data, timeout=10)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
        
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                print(f"✅ Prediction successful: {data['prediction']} ({data['risk_percentage']})")
                return True
            else:
                print(f"❌ API returned error: {data.get('error')}")
                return False
        else:
            print(f"❌ HTTP error: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Request failed: {e}")
        return False

def test_health_endpoint():
    """Test the health check endpoint"""
    try:
        response = requests.get("http://localhost:5000/api/health", timeout=5)
        print(f"Health Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Model Loaded: {data.get('model_loaded')}")
            return data.get('model_loaded', False)
        return False
    except Exception as e:
        print(f"Health check failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing Heart Disease Prediction API...")
    print("=" * 50)
    
    # Test health endpoint
    print("\n1. Testing Health Endpoint...")
    health_ok = test_health_endpoint()
    
    # Test prediction
    print("\n2. Testing Prediction Endpoint...")  
    prediction_ok = test_api_prediction()
    
    print("\n" + "=" * 50)
    if health_ok and prediction_ok:
        print("✅ All tests passed! API is working correctly.")
    else:
        print("❌ Some tests failed. Check the logs above.")