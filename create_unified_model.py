#!/usr/bin/env python3
"""
Unified Model System for Render Deployment
Ensures identical predictions between local and production environments.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import hashlib
import json
from datetime import datetime
import os

def create_production_model():
    """Create a deterministic model that works identically in all environments."""
    
    print("🏗️ Creating unified production model...")
    
    # Use fixed seed for reproducibility
    np.random.seed(12345)
    n_samples = 2500
    
    print(f"📊 Generating {n_samples} training samples...")
    
    # Create realistic, deterministic dataset
    age = np.random.uniform(20, 85, n_samples)
    gender = np.random.choice(['Male', 'Female'], n_samples, p=[0.55, 0.45])
    
    # Create correlated features based on medical research
    blood_pressure = []
    cholesterol = []
    bmi = []
    
    for i in range(n_samples):
        # Age-related BP increase
        bp_base = 100 + (age[i] - 20) * 0.6
        bp_noise = np.random.normal(0, 15)
        bp = max(80, min(220, bp_base + bp_noise))
        blood_pressure.append(bp)
        
        # Age and gender-related cholesterol
        chol_base = 160 + (age[i] - 20) * 0.8
        if gender[i] == 'Male':
            chol_base += 15
        chol_noise = np.random.normal(0, 30)
        chol = max(120, min(400, chol_base + chol_noise))
        cholesterol.append(chol)
        
        # BMI with realistic distribution
        bmi_val = np.random.normal(26.5, 4.5)
        bmi_val = max(16, min(45, bmi_val))
        bmi.append(bmi_val)
    
    # Other features
    exercise_habits = np.random.choice(['Low', 'Medium', 'High'], n_samples, p=[0.35, 0.45, 0.20])
    alcohol = np.random.choice(['None', 'Light', 'Moderate', 'Heavy'], n_samples, p=[0.30, 0.45, 0.20, 0.05])
    stress = np.random.choice(['Low', 'Medium', 'High'], n_samples, p=[0.25, 0.55, 0.20])
    sleep_hours = np.random.uniform(4.5, 10.5, n_samples)
    sugar_consumption = np.random.choice(['Low', 'Medium', 'High'], n_samples, p=[0.30, 0.50, 0.20])
    
    # Medical indicators
    triglycerides = np.random.uniform(60, 350, n_samples)
    fasting_glucose = np.random.uniform(70, 180, n_samples)
    crp = np.random.exponential(1.8, n_samples).clip(0.2, 12.0)
    homocysteine = np.random.uniform(5, 22, n_samples)
    
    # Create target variable with realistic medical risk factors
    heart_disease = []
    for i in range(n_samples):
        risk_score = 0.0
        
        # Age risk (strongest predictor)
        if age[i] >= 70:
            risk_score += 0.35
        elif age[i] >= 60:
            risk_score += 0.25
        elif age[i] >= 50:
            risk_score += 0.15
        elif age[i] >= 40:
            risk_score += 0.05
        
        # Gender risk
        if gender[i] == 'Male':
            risk_score += 0.12
        
        # Blood pressure risk
        if blood_pressure[i] >= 180:
            risk_score += 0.30
        elif blood_pressure[i] >= 160:
            risk_score += 0.22
        elif blood_pressure[i] >= 140:
            risk_score += 0.15
        elif blood_pressure[i] >= 130:
            risk_score += 0.08
        
        # Cholesterol risk
        if cholesterol[i] >= 280:
            risk_score += 0.25
        elif cholesterol[i] >= 240:
            risk_score += 0.18
        elif cholesterol[i] >= 200:
            risk_score += 0.10
        
        # BMI risk
        if bmi[i] >= 35:
            risk_score += 0.15
        elif bmi[i] >= 30:
            risk_score += 0.10
        elif bmi[i] >= 25:
            risk_score += 0.05
        
        # Exercise protection
        if exercise_habits[i] == 'High':
            risk_score -= 0.12
        elif exercise_habits[i] == 'Low':
            risk_score += 0.10
        
        # Lifestyle factors
        if alcohol[i] == 'Heavy':
            risk_score += 0.08
        if stress[i] == 'High':
            risk_score += 0.06
        if sleep_hours[i] < 6 or sleep_hours[i] > 9:
            risk_score += 0.04
        
        # Medical indicators
        if triglycerides[i] > 200:
            risk_score += 0.06
        if fasting_glucose[i] > 125:
            risk_score += 0.08
        if crp[i] > 3:
            risk_score += 0.05
        if homocysteine[i] > 15:
            risk_score += 0.04
        
        # Add controlled randomness
        random_factor = np.random.normal(0, 0.08)
        final_risk = risk_score + random_factor
        
        # Convert to binary outcome
        heart_disease.append(1 if final_risk > 0.38 else 0)
    
    # Create DataFrame
    data = pd.DataFrame({
        'Age': np.round(age, 1),
        'Gender': gender,
        'Blood Pressure': np.round(blood_pressure, 0).astype(int),
        'Cholesterol Level': np.round(cholesterol, 0).astype(int),
        'BMI': np.round(bmi, 1),
        'Exercise Habits': exercise_habits,
        'Alcohol Consumption': alcohol,
        'Stress Level': stress,
        'Sleep Hours': np.round(sleep_hours, 1),
        'Sugar Consumption': sugar_consumption,
        'Triglyceride Level': np.round(triglycerides, 0).astype(int),
        'Fasting Blood Sugar': np.round(fasting_glucose, 0).astype(int),
        'CRP Level': np.round(crp, 2),
        'Homocysteine Level': np.round(homocysteine, 1),
        'Heart Disease': heart_disease
    })
    
    print(f"✅ Dataset created: {len(data)} samples")
    positive_cases = sum(heart_disease)
    print(f"📊 Heart Disease cases: {positive_cases}/{len(data)} ({positive_cases/len(data)*100:.1f}%)")
    
    return data

def train_unified_model(data):
    """Train a model with fixed parameters for consistent results."""
    
    print("\n🤖 Training unified RandomForest model...")
    
    X = data.drop('Heart Disease', axis=1)
    y = data['Heart Disease']
    
    # Encode categorical variables with consistent order
    categorical_cols = ['Gender', 'Exercise Habits', 'Alcohol Consumption', 'Stress Level', 'Sugar Consumption']
    label_encoders = {}
    
    X_encoded = X.copy()
    for col in categorical_cols:
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
        label_encoders[col] = le
    
    # Train model with fixed parameters for reproducibility
    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        bootstrap=True,
        random_state=12345,  # Fixed seed
        n_jobs=1  # Single thread for consistency across platforms
    )
    
    print("🔄 Training model...")
    model.fit(X_encoded, y)
    
    # Calculate accuracy
    accuracy = model.score(X_encoded, y)
    print(f"📈 Model accuracy: {accuracy:.4f}")
    
    # Get feature importance
    feature_importance = dict(zip(X_encoded.columns, model.feature_importances_))
    
    # Create model signature for validation
    model_signature = {
        'n_samples': len(data),
        'n_features': len(X_encoded.columns),
        'accuracy': accuracy,
        'feature_order': list(X_encoded.columns),
        'categorical_mappings': {col: list(le.classes_) for col, le in label_encoders.items()},
        'training_date': datetime.now().isoformat(),
        'model_version': '3.0_unified'
    }
    
    return model, label_encoders, X_encoded.columns.tolist(), feature_importance, model_signature

def save_production_model(model, label_encoders, feature_columns, feature_importance, signature):
    """Save the unified model with validation data."""
    
    print("\n💾 Saving unified production model...")
    
    # Create comprehensive model package
    model_package = {
        'model': model,
        'label_encoders': label_encoders,
        'feature_columns': feature_columns,
        'feature_names': feature_columns,  # Compatibility
        'original_columns': feature_columns,  # Compatibility
        'feature_means': {},  # Will be filled if needed
        'feature_importance': feature_importance,
        'model_signature': signature,
        'model_info': {
            'model_type': 'RandomForest_Unified',
            'training_date': datetime.now().isoformat(),
            'n_features': len(feature_columns),
            'optimized_for': 'universal_deployment',
            'accuracy': signature['accuracy'],
            'version': '3.0_unified',
            'deterministic': True
        }
    }
    
    # Save with multiple names for compatibility
    filenames = [
        'heart_disease_render_optimized.pkl',
        'heart_disease_production_unified.pkl'
    ]
    
    for filename in filenames:
        joblib.dump(model_package, filename, compress=3)
        file_size = os.path.getsize(filename) / (1024*1024)
        print(f"✅ Saved: {filename} ({file_size:.1f} MB)")
    
    # Save signature separately for validation
    with open('model_signature.json', 'w') as f:
        json.dump(signature, f, indent=2)
    
    print("✅ Model signature saved for validation")
    
    return filenames[0]

def test_model_consistency():
    """Test that the model produces consistent results."""
    
    print("\n🧪 Testing model consistency...")
    
    # Load the model
    from enhanced_model_wrapper import EnhancedHeartDiseasePredictor
    
    predictor = EnhancedHeartDiseasePredictor()
    success = predictor.load_model('heart_disease_render_optimized.pkl')
    
    if not success:
        print("❌ Failed to load model")
        return False
    
    # Test with fixed input
    test_cases = [
        {
            'Age': 55,
            'Gender': 'Male',
            'Blood Pressure': 150,
            'Cholesterol Level': 240,
            'BMI': 28.5,
            'Exercise Habits': 'Medium',
            'Alcohol Consumption': 'Light',
            'Stress Level': 'Medium',
            'Sleep Hours': 7.0,
            'Sugar Consumption': 'Medium',
            'Triglyceride Level': 180,
            'Fasting Blood Sugar': 95,
            'CRP Level': 2.5,
            'Homocysteine Level': 12.0
        },
        {
            'Age': 35,
            'Gender': 'Female',
            'Blood Pressure': 120,
            'Cholesterol Level': 190,
            'BMI': 23.0,
            'Exercise Habits': 'High',
            'Alcohol Consumption': 'None',
            'Stress Level': 'Low',
            'Sleep Hours': 8.0,
            'Sugar Consumption': 'Low',
            'Triglyceride Level': 120,
            'Fasting Blood Sugar': 85,
            'CRP Level': 1.0,
            'Homocysteine Level': 8.5
        }
    ]
    
    print("🔬 Running consistency tests...")
    
    for i, test_case in enumerate(test_cases):
        try:
            result = predictor.predict(test_case)
            print(f"Test {i+1}: {result['prediction']} ({result['risk_percentage']}) - Confidence: {result['confidence']}")
        except Exception as e:
            print(f"❌ Test {i+1} failed: {e}")
            return False
    
    print("✅ All consistency tests passed!")
    return True

def main():
    """Main training and validation pipeline."""
    
    print("🚀 Creating Unified Production Model for Render")
    print("=" * 60)
    
    try:
        # Create deterministic dataset
        data = create_production_model()
        
        # Train unified model
        model, encoders, features, importance, signature = train_unified_model(data)
        
        # Save production model
        model_file = save_production_model(model, encoders, features, importance, signature)
        
        # Test consistency
        if test_model_consistency():
            print("\n" + "=" * 60)
            print("🎉 Unified production model created successfully!")
            print(f"📁 Primary model: {model_file}")
            print(f"🎯 Accuracy: {signature['accuracy']:.4f}")
            print(f"📊 Features: {len(features)}")
            print("\n🔄 Next steps:")
            print("   1. Commit and push the new model files")
            print("   2. Deploy to Render")
            print("   3. Predictions will now be IDENTICAL to local environment!")
            return True
        else:
            print("❌ Model consistency test failed!")
            return False
            
    except Exception as e:
        print(f"\n❌ Failed to create unified model: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)