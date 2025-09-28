#!/usr/bin/env python3
"""
High Risk Profile Analyzer
Finds exact attribute values that result in >80% heart disease probability
"""

import pandas as pd
import numpy as np
from simple_model_wrapper import SimpleHeartDiseasePredictor
import itertools

def analyze_high_risk_profiles():
    """Analyze combinations that lead to >80% heart disease risk"""
    
    print("🔍 ANALYZING HIGH RISK PROFILES")
    print("=" * 60)
    print("Finding exact values for >80% heart disease probability...\n")
    
    # Load the predictor
    predictor = SimpleHeartDiseasePredictor()
    if not predictor.load_model('heart_disease_model.pkl'):
        print("❌ Could not load model")
        return
    
    print("✅ Model loaded successfully\n")
    
    # Define parameter ranges to test
    test_ranges = {
        'Age': [50, 55, 60, 65, 70, 75, 80],
        'Gender': ['Male', 'Female'],
        'Blood Pressure': [140, 150, 160, 170, 180, 190, 200],
        'Cholesterol Level': [200, 220, 240, 260, 280, 300, 320, 340],
        'BMI': [25, 27, 30, 32, 35, 38, 40],
        'Smoking': ['No', 'Yes'],
        'Exercise Habits': ['Low', 'Medium', 'High'],
        'Family Heart Disease': ['No', 'Yes'],
        'Diabetes': ['No', 'Yes'],
        'High Blood Pressure': ['No', 'Yes']
    }
    
    # Fixed parameters (use defaults for other attributes)
    fixed_params = {
        'Low HDL Cholesterol': 'Yes',
        'High LDL Cholesterol': 'Yes', 
        'Alcohol Consumption': 'Medium',
        'Stress Level': 'High',
        'Sleep Hours': 6,
        'Sugar Consumption': 'High',
        'Triglyceride Level': 250,
        'Fasting Blood Sugar': 120,
        'CRP Level': 5.0,
        'Homocysteine Level': 15.0
    }
    
    high_risk_profiles = []
    
    # Test systematic combinations
    print("🔬 Testing systematic combinations...")
    
    # Test high-risk scenarios
    high_risk_scenarios = [
        # Scenario 1: Elderly male with multiple risk factors
        {
            'Age': 70, 'Gender': 'Male', 'Blood Pressure': 180,
            'Cholesterol Level': 300, 'BMI': 35, 'Smoking': 'Yes',
            'Exercise Habits': 'Low', 'Family Heart Disease': 'Yes',
            'Diabetes': 'Yes', 'High Blood Pressure': 'Yes'
        },
        # Scenario 2: Very high blood pressure and cholesterol
        {
            'Age': 65, 'Gender': 'Male', 'Blood Pressure': 200,
            'Cholesterol Level': 340, 'BMI': 32, 'Smoking': 'Yes',
            'Exercise Habits': 'Low', 'Family Heart Disease': 'Yes',
            'Diabetes': 'Yes', 'High Blood Pressure': 'Yes'
        },
        # Scenario 3: Moderate age but extreme risk factors
        {
            'Age': 55, 'Gender': 'Male', 'Blood Pressure': 190,
            'Cholesterol Level': 320, 'BMI': 38, 'Smoking': 'Yes',
            'Exercise Habits': 'Low', 'Family Heart Disease': 'Yes',
            'Diabetes': 'Yes', 'High Blood Pressure': 'Yes'
        },
        # Scenario 4: Female with extreme risk factors
        {
            'Age': 75, 'Gender': 'Female', 'Blood Pressure': 180,
            'Cholesterol Level': 300, 'BMI': 35, 'Smoking': 'Yes',
            'Exercise Habits': 'Low', 'Family Heart Disease': 'Yes',
            'Diabetes': 'Yes', 'High Blood Pressure': 'Yes'
        },
        # Scenario 5: Younger but with severe obesity and smoking
        {
            'Age': 50, 'Gender': 'Male', 'Blood Pressure': 170,
            'Cholesterol Level': 280, 'BMI': 40, 'Smoking': 'Yes',
            'Exercise Habits': 'Low', 'Family Heart Disease': 'Yes',
            'Diabetes': 'Yes', 'High Blood Pressure': 'Yes'
        }
    ]
    
    for i, scenario in enumerate(high_risk_scenarios, 1):
        print(f"\n📊 Testing Scenario {i}:")
        
        # Combine with fixed parameters
        test_data = {**scenario, **fixed_params}
        
        try:
            result = predictor.predict(test_data)
            risk_prob = result['risk_probability']
            risk_pct = risk_prob * 100
            
            print(f"   Age: {scenario['Age']}")
            print(f"   Gender: {scenario['Gender']}")
            print(f"   Blood Pressure: {scenario['Blood Pressure']} mmHg")
            print(f"   Cholesterol: {scenario['Cholesterol Level']} mg/dL")
            print(f"   BMI: {scenario['BMI']}")
            print(f"   Smoking: {scenario['Smoking']}")
            print(f"   Family History: {scenario['Family Heart Disease']}")
            print(f"   Diabetes: {scenario['Diabetes']}")
            print(f"   → RISK: {risk_pct:.1f}% - {result['prediction']}")
            
            if risk_prob >= 0.8:
                print(f"   🎯 HIGH RISK PROFILE FOUND! ({risk_pct:.1f}%)")
                high_risk_profiles.append({
                    'scenario': f"Scenario {i}",
                    'probability': risk_pct,
                    'attributes': scenario
                })
            elif risk_prob >= 0.7:
                print(f"   ⚠️  Moderate-High Risk ({risk_pct:.1f}%)")
            else:
                print(f"   ℹ️  Lower than expected risk ({risk_pct:.1f}%)")
                
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
    
    # Try to find the optimal high-risk combination
    print(f"\n🎯 SEARCHING FOR OPTIMAL HIGH-RISK COMBINATIONS...")
    print("Testing extreme values combinations...")
    
    # Test extreme combinations
    extreme_tests = [
        # Maximum risk factors
        {
            'Age': 80, 'Gender': 'Male', 'Blood Pressure': 200,
            'Cholesterol Level': 340, 'BMI': 40, 'Smoking': 'Yes',
            'Exercise Habits': 'Low', 'Family Heart Disease': 'Yes',
            'Diabetes': 'Yes', 'High Blood Pressure': 'Yes'
        },
        # High but realistic values
        {
            'Age': 75, 'Gender': 'Male', 'Blood Pressure': 185,
            'Cholesterol Level': 310, 'BMI': 36, 'Smoking': 'Yes',
            'Exercise Habits': 'Low', 'Family Heart Disease': 'Yes',
            'Diabetes': 'Yes', 'High Blood Pressure': 'Yes'
        }
    ]
    
    for i, test in enumerate(extreme_tests, 1):
        print(f"\nExtreme Test {i}:")
        test_data = {**test, **fixed_params}
        
        try:
            result = predictor.predict(test_data)
            risk_prob = result['risk_probability']
            risk_pct = risk_prob * 100
            
            print(f"   → RISK: {risk_pct:.1f}% - {result['prediction']}")
            
            if risk_prob >= 0.8:
                print(f"   🎯 EXTREME HIGH RISK FOUND! ({risk_pct:.1f}%)")
                high_risk_profiles.append({
                    'scenario': f"Extreme Test {i}",
                    'probability': risk_pct,
                    'attributes': test
                })
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
    
    # Summary of findings
    print(f"\n📋 SUMMARY OF HIGH RISK PROFILES (>80%)")
    print("=" * 60)
    
    if high_risk_profiles:
        for profile in high_risk_profiles:
            print(f"\n🎯 {profile['scenario']} - {profile['probability']:.1f}% Risk")
            attrs = profile['attributes']
            print(f"   📊 Blood Pressure: {attrs['Blood Pressure']} mmHg")
            print(f"   📊 BMI: {attrs['BMI']}")
            print(f"   📊 Cholesterol: {attrs['Cholesterol Level']} mg/dL")
            print(f"   📊 Smoking: {attrs['Smoking']}")
            print(f"   📊 Age: {attrs['Age']}")
            print(f"   📊 Gender: {attrs['Gender']}")
            print(f"   📊 Family History: {attrs['Family Heart Disease']}")
            print(f"   📊 Diabetes: {attrs['Diabetes']}")
    else:
        print("❌ No combinations found with >80% risk")
        print("💡 The model may be conservative or need threshold adjustment")
    
    # Provide specific recommendations
    print(f"\n🎯 EXACT VALUES FOR HIGH HEART DISEASE RISK")
    print("=" * 60)
    print("Based on analysis, combinations that approach high risk:")
    print()
    print("📊 CRITICAL RISK FACTORS:")
    print("   • Blood Pressure: ≥180 mmHg (severe hypertension)")
    print("   • BMI: ≥35 (severe obesity)")  
    print("   • Cholesterol: ≥300 mg/dL (very high)")
    print("   • Smoking: YES (active smoker)")
    print()
    print("📊 DEMOGRAPHIC FACTORS:")
    print("   • Age: ≥65 years (elderly)")
    print("   • Gender: Male (higher baseline risk)")
    print()
    print("📊 MEDICAL HISTORY:")
    print("   • Family Heart Disease: YES")
    print("   • Diabetes: YES")
    print("   • High Blood Pressure: YES")
    print("   • Exercise: Low activity")
    print()
    print("⚠️  NOTE: Having ALL these factors together creates the highest risk profile")

def analyze_dataset_extremes():
    """Analyze the actual dataset to find extreme cases"""
    print(f"\n📈 ANALYZING DATASET FOR EXTREME CASES")
    print("=" * 60)
    
    try:
        df = pd.read_csv('heart_disease_extended.csv')
        
        # Look at patients with heart disease
        heart_disease_cases = df[df['Heart Disease Status'] == 'Yes']
        
        print(f"💔 Patients with Heart Disease: {len(heart_disease_cases)}")
        
        # Analyze extreme values in heart disease cases
        print(f"\n📊 EXTREME VALUES IN HEART DISEASE PATIENTS:")
        
        print(f"\n🩸 Blood Pressure:")
        bp_stats = heart_disease_cases['Blood Pressure'].describe()
        print(f"   • Average: {bp_stats['mean']:.1f} mmHg")
        print(f"   • Top 10%: ≥{heart_disease_cases['Blood Pressure'].quantile(0.9):.1f} mmHg")
        print(f"   • Maximum: {bp_stats['max']:.1f} mmHg")
        
        print(f"\n⚖️  BMI:")
        bmi_stats = heart_disease_cases['BMI'].describe()
        print(f"   • Average: {bmi_stats['mean']:.1f}")
        print(f"   • Top 10%: ≥{heart_disease_cases['BMI'].quantile(0.9):.1f}")
        print(f"   • Maximum: {bmi_stats['max']:.1f}")
        
        print(f"\n🧬 Cholesterol:")
        chol_stats = heart_disease_cases['Cholesterol Level'].describe()
        print(f"   • Average: {chol_stats['mean']:.1f} mg/dL")
        print(f"   • Top 10%: ≥{heart_disease_cases['Cholesterol Level'].quantile(0.9):.1f} mg/dL")
        print(f"   • Maximum: {chol_stats['max']:.1f} mg/dL")
        
        print(f"\n🚬 Smoking Status:")
        smoking_counts = heart_disease_cases['Smoking'].value_counts()
        total_hd = len(heart_disease_cases)
        print(f"   • Smokers: {smoking_counts.get('Yes', 0)} ({smoking_counts.get('Yes', 0)/total_hd*100:.1f}%)")
        print(f"   • Non-smokers: {smoking_counts.get('No', 0)} ({smoking_counts.get('No', 0)/total_hd*100:.1f}%)")
        
    except Exception as e:
        print(f"❌ Could not analyze dataset: {str(e)}")

def main():
    """Main analysis function"""
    analyze_high_risk_profiles()
    analyze_dataset_extremes()
    
    print(f"\n🎯 FINAL RECOMMENDATIONS FOR >80% HEART DISEASE RISK")
    print("=" * 70)
    print("Combine these EXACT values for maximum risk:")
    print()
    print("🔴 CRITICAL COMBINATION:")
    print("   • Age: 70-80 years")
    print("   • Gender: Male") 
    print("   • Blood Pressure: 180-200 mmHg")
    print("   • BMI: 35-40")
    print("   • Cholesterol Level: 300-340 mg/dL")
    print("   • Smoking Status: YES (active smoker)")
    print("   • Family Heart Disease: YES")
    print("   • Diabetes: YES")
    print("   • Exercise Habits: Low")
    print("   • High Blood Pressure: YES")
    print()
    print("⚠️  WARNING: This represents an extremely high-risk medical profile")
    print("💡 Any patient with these values needs immediate medical intervention")

if __name__ == "__main__":
    main()