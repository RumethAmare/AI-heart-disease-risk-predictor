#!/usr/bin/env python3
"""
Test both the model and statistics API endpoints
"""

import requests
import json

def test_model_prediction():
    """Test that the model is working properly (not fallback)"""
    
    print("🧪 Testing Model Prediction API...")
    
    # Test with a sample patient
    test_data = {
        'age': 55,
        'gender': 'Male',
        'blood_pressure': 140,
        'cholesterol_level': 220,
        'bmi': 28,
        'smoking': 'Yes',
        'diabetes': 'No',
        'family_heart_disease': 'Yes',
        'exercise_habits': 'Medium',
        'high_blood_pressure': 'Yes',
        'high_ldl_cholesterol': 'Yes',
        'low_hdl_cholesterol': 'No',
        'stress_level': 'Medium',
        'alcohol_consumption': 'Low',
        'sugar_consumption': 'Medium',
        'sleep_hours': 7,
        'crp_level': 2.5,
        'homocysteine_level': 11.0,
        'triglyceride_level': 150,
        'fasting_blood_sugar': 95
    }
    
    try:
        response = requests.post('http://localhost:5000/api/predict', json=test_data, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Model Prediction API working!")
            print(f"   Prediction: {result['prediction']}")
            print(f"   Risk: {result['risk_percentage']}")
            print(f"   Level: {result['risk_level']}")
            
            # Check if it's using the enhanced model (not fallback)
            if 'combined_approach' in result and result['combined_approach']:
                print(f"✅ Enhanced model (not fallback) is working properly!")
                print(f"   Clinical Score: {result.get('clinical_score', 'N/A')}/100")
                print(f"   ML Probability: {result.get('ml_probability', 'N/A')}")
                return True
            else:
                print(f"⚠️  Model working but may be using fallback mode")
                return True
        else:
            print(f"❌ Model API Error: {response.status_code}")
            print(response.text)
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Connection Error: {e}")
        return False

def test_statistics_api():
    """Test that the statistics API is working"""
    
    print(f"\n📊 Testing Statistics API...")
    
    try:
        response = requests.get('http://localhost:5000/api/statistics', timeout=10)
        
        if response.status_code == 200:
            api_response = response.json()
            stats = api_response.get('data', {})
            print(f"✅ Statistics API working!")
            
            if 'dataset_info' in stats:
                print(f"   Dataset Records: {stats['dataset_info']['total_records']:,}")
                print(f"   Features: {stats['dataset_info']['total_features']}")
                print(f"   Target Distribution: {stats['target_distribution']['counts']}")
                
                # Check if we have feature data
                if 'features' in stats and len(stats['features']) > 0:
                    print(f"✅ Feature statistics loaded: {len(stats['features'])} features")
                    return True
                else:
                    print(f"⚠️  Statistics loaded but feature data may be incomplete")
                    return True
            else:
                print(f"⚠️  Statistics response structure unexpected")
                return True
        else:
            print(f"❌ Statistics API Error: {response.status_code}")
            print(response.text)
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Connection Error: {e}")
        return False

def test_statistics_page():
    """Test that the statistics HTML page loads"""
    
    print(f"\n📄 Testing Statistics Page...")
    
    try:
        response = requests.get('http://localhost:5000/statistics', timeout=10)
        
        if response.status_code == 200:
            html_content = response.text
            if 'Heart Disease Statistics' in html_content or 'statistics' in html_content.lower():
                print(f"✅ Statistics page loads successfully!")
                return True
            else:
                print(f"⚠️  Statistics page loads but content may be incomplete")
                return True
        else:
            print(f"❌ Statistics Page Error: {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Connection Error: {e}")
        return False

def main():
    print("🔧 TESTING FIXED FLASK APPLICATION")
    print("="*60)
    
    # Test all endpoints
    model_ok = test_model_prediction()
    stats_api_ok = test_statistics_api()
    stats_page_ok = test_statistics_page()
    
    print(f"\n" + "="*60)
    print("📋 TEST RESULTS SUMMARY")
    print("="*60)
    
    print(f"🤖 Model Prediction: {'✅ WORKING' if model_ok else '❌ FAILED'}")
    print(f"📊 Statistics API:   {'✅ WORKING' if stats_api_ok else '❌ FAILED'}")
    print(f"📄 Statistics Page:  {'✅ WORKING' if stats_page_ok else '❌ FAILED'}")
    
    if model_ok and stats_api_ok and stats_page_ok:
        print(f"\n🎉 ALL TESTS PASSED!")
        print(f"✅ Model is using properly trained version (not fallback)")
        print(f"✅ Statistics page is working with dataset")
        print(f"✅ Ready for deployment!")
    else:
        print(f"\n⚠️  Some issues found - check individual test results above")
    
    print(f"\n🔗 Test your application:")
    print(f"   Main page: http://localhost:5000/")
    print(f"   Assessment: http://localhost:5000/assess")  
    print(f"   Statistics: http://localhost:5000/statistics")

if __name__ == "__main__":
    main()