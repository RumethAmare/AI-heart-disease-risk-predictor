#!/usr/bin/env python3
"""
Quick Model Training for Render Deployment
Creates a fast, optimized model for immediate deployment.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib
from datetime import datetime
import os

def create_optimized_synthetic_dataset():
    """Create a high-quality synthetic dataset optimized for quick training."""
    
    print("🧬 Creating optimized synthetic dataset...")
    
    np.random.seed(42)
    n_samples = 2000  # Smaller for faster training
    
    # Generate realistic medical data
    age = np.random.normal(55, 15, n_samples).clip(18, 90)
    gender = np.random.choice(['Male', 'Female'], n_samples, p=[0.6, 0.4])
    
    # Blood pressure - influenced by age and gender
    bp_base = 110 + (age - 30) * 0.5 + np.where(gender == 'Male', 5, 0)
    blood_pressure = np.random.normal(bp_base, 15, n_samples).clip(80, 220)
    
    # Cholesterol - influenced by age and gender
    chol_base = 180 + (age - 30) * 0.8 + np.where(gender == 'Male', 10, 0)
    cholesterol = np.random.normal(chol_base, 40, n_samples).clip(120, 400)
    
    # BMI - realistic distribution
    bmi = np.random.normal(26, 4, n_samples).clip(16, 45)
    
    # Other factors
    exercise_habits = np.random.choice(['Low', 'Medium', 'High'], n_samples, p=[0.3, 0.5, 0.2])
    alcohol = np.random.choice(['None', 'Light', 'Moderate', 'Heavy'], n_samples, p=[0.3, 0.4, 0.25, 0.05])
    stress = np.random.choice(['Low', 'Medium', 'High'], n_samples, p=[0.2, 0.6, 0.2])
    
    # Sleep hours
    sleep_hours = np.random.normal(7.5, 1.2, n_samples).clip(4, 11)
    
    # Additional medical indicators for better predictions
    triglycerides = np.random.normal(150, 40, n_samples).clip(50, 400)
    fasting_glucose = np.random.normal(95, 20, n_samples).clip(60, 200)
    crp = np.random.exponential(2, n_samples).clip(0.1, 10)
    homocysteine = np.random.normal(10, 3, n_samples).clip(5, 25)
    
    # Create target variable based on realistic risk factors
    risk_scores = []
    for i in range(n_samples):
        score = 0
        
        # Age factor (strong predictor)
        if age[i] > 65: score += 0.30
        elif age[i] > 55: score += 0.20
        elif age[i] > 45: score += 0.10
        
        # Gender factor  
        if gender[i] == 'Male': score += 0.10
        
        # Blood pressure (major risk factor)
        if blood_pressure[i] > 180: score += 0.30
        elif blood_pressure[i] > 160: score += 0.22
        elif blood_pressure[i] > 140: score += 0.15
        elif blood_pressure[i] > 130: score += 0.08
        
        # Cholesterol (major risk factor)
        if cholesterol[i] > 280: score += 0.25
        elif cholesterol[i] > 240: score += 0.18
        elif cholesterol[i] > 200: score += 0.10
        
        # BMI
        if bmi[i] > 35: score += 0.18
        elif bmi[i] > 30: score += 0.12
        elif bmi[i] > 25: score += 0.06
        
        # Exercise protection (strong protective factor)
        if exercise_habits[i] == 'High': score -= 0.15
        elif exercise_habits[i] == 'Low': score += 0.12
        
        # Other factors
        if alcohol[i] == 'Heavy': score += 0.10
        if stress[i] == 'High': score += 0.08
        if sleep_hours[i] < 6 or sleep_hours[i] > 9: score += 0.06
        
        # Advanced indicators
        if triglycerides[i] > 200: score += 0.08
        if fasting_glucose[i] > 125: score += 0.10
        if crp[i] > 3: score += 0.08
        if homocysteine[i] > 15: score += 0.06
        
        risk_scores.append(score)
    
    # Convert to binary outcome with realistic threshold
    risk_probs = np.array(risk_scores)
    # Add some noise to make it more realistic
    risk_probs += np.random.normal(0, 0.03, n_samples)
    target = (risk_probs > 0.35).astype(int)  # Realistic prevalence
    
    # Create DataFrame
    data = pd.DataFrame({
        'Age': age.astype(int),
        'Gender': gender,
        'Blood Pressure': blood_pressure.astype(int),
        'Cholesterol Level': cholesterol.astype(int),
        'BMI': bmi.round(1),
        'Exercise Habits': exercise_habits,
        'Alcohol Consumption': alcohol,
        'Stress Level': stress,
        'Sleep Hours': sleep_hours.round(1),
        'Sugar Consumption': np.random.choice(['Low', 'Medium', 'High'], n_samples),
        'Triglyceride Level': triglycerides.astype(int),
        'Fasting Blood Sugar': fasting_glucose.astype(int),
        'CRP Level': crp.round(2),
        'Homocysteine Level': homocysteine.round(1),
        'Heart Disease': target
    })
    
    print(f"✅ Created optimized dataset: {len(data)} samples")
    print(f"📊 Heart Disease distribution: {data['Heart Disease'].value_counts().to_dict()}")
    
    return data

def train_quick_model(data):
    """Train a fast, optimized Random Forest model."""
    
    print("\n🚀 Training fast optimized Random Forest model...")
    
    # Prepare features and target
    X = data.drop('Heart Disease', axis=1)
    y = data['Heart Disease']
    
    print(f"📊 Training data: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Handle categorical variables
    categorical_cols = ['Gender', 'Exercise Habits', 'Alcohol Consumption', 'Stress Level', 'Sugar Consumption']
    label_encoders = {}
    
    X_processed = X.copy()
    for col in categorical_cols:
        if col in X_processed.columns:
            le = LabelEncoder()
            X_processed[col] = le.fit_transform(X_processed[col].astype(str))
            label_encoders[col] = le
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train optimized Random Forest (fast but effective parameters)
    model = RandomForestClassifier(
        n_estimators=150,  # Good balance of performance and speed
        max_depth=15,      # Prevent overfitting
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )
    
    print("🔄 Training model...")
    model.fit(X_train, y_train)
    
    # Evaluate model
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    y_pred = model.predict(X_test)
    
    print(f"\n📈 Model Performance:")
    print(f"   Training Accuracy: {train_score:.4f}")
    print(f"   Test Accuracy: {test_score:.4f}")
    
    print(f"\n📊 Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X_processed.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n🔍 Top 10 Feature Importance:")
    print(feature_importance.head(10).to_string(index=False))
    
    return model, label_encoders, X_processed.columns.tolist(), feature_importance

def save_render_model(model, label_encoders, feature_columns, feature_importance):
    """Save the model for Render deployment."""
    
    print(f"\n💾 Saving Render-optimized model...")
    
    # Create comprehensive model package
    model_package = {
        'model': model,
        'label_encoders': label_encoders,
        'feature_columns': feature_columns,
        'feature_names': feature_columns,  # Compatibility
        'original_columns': feature_columns,  # Compatibility
        'feature_importance': feature_importance.to_dict('records'),
        'model_info': {
            'model_type': 'RandomForest',
            'training_date': datetime.now().isoformat(),
            'n_features': len(feature_columns),
            'optimized_for': 'render_deployment',
            'version': '2.0'
        }
    }
    
    # Save as standard filename
    filename = "heart_disease_render_optimized.pkl"
    joblib.dump(model_package, filename, compress=3)
    
    print(f"✅ Model saved as: {filename}")
    print(f"📁 File size: {os.path.getsize(filename) / (1024*1024):.1f} MB")
    
    return filename

def main():
    """Quick training pipeline."""
    
    print("🚀 Quick Heart Disease Model Training for Render")
    print("=" * 55)
    
    try:
        # Create synthetic data
        data = create_optimized_synthetic_dataset()
        
        # Train model
        model, encoders, feature_cols, importance = train_quick_model(data)
        
        # Save model
        model_file = save_render_model(model, encoders, feature_cols, importance)
        
        print("\n" + "=" * 55)
        print("🎉 Quick training completed successfully!")
        print(f"📁 Model saved: {model_file}")
        print("\n🔄 Next steps:")
        print("   1. Test the model locally")
        print("   2. Commit and push to repository")
        print("   3. Deploy to Render for high-quality ML predictions!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)