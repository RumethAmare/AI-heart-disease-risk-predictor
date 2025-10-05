#!/usr/bin/env python3
"""
Train Optimized Model for Render Deployment
Creates a lightweight, high-performance model specifically for production deployment.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
from datetime import datetime
import os

def load_and_prepare_data():
    """Load the extended heart disease dataset and prepare it for training."""
    
    print("🔄 Loading extended heart disease dataset...")
    
    # Try to load the extended dataset first
    dataset_files = [
        'heart_disease_extended_dataset.csv',
        'heart_disease_dataset.csv', 
        'heart.csv'
    ]
    
    data = None
    for filename in dataset_files:
        if os.path.exists(filename):
            try:
                data = pd.read_csv(filename)
                print(f"✅ Loaded dataset: {filename} ({len(data)} rows, {len(data.columns)} columns)")
                break
            except Exception as e:
                print(f"❌ Error loading {filename}: {e}")
                continue
    
    if data is None:
        print("❌ No dataset found! Creating synthetic dataset for training...")
        data = create_synthetic_dataset()
    
    return data

def create_synthetic_dataset():
    """Create a high-quality synthetic dataset based on medical research."""
    
    print("🧬 Creating synthetic heart disease dataset...")
    
    np.random.seed(42)
    n_samples = 5000  # Larger dataset for better training
    
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
    smoking = np.random.choice(['Never', 'Former', 'Current'], n_samples, p=[0.5, 0.3, 0.2])
    alcohol = np.random.choice(['None', 'Light', 'Moderate', 'Heavy'], n_samples, p=[0.3, 0.4, 0.25, 0.05])
    stress = np.random.choice(['Low', 'Medium', 'High'], n_samples, p=[0.2, 0.6, 0.2])
    
    # Sleep hours
    sleep_hours = np.random.normal(7.5, 1.2, n_samples).clip(4, 11)
    
    # Create target variable based on realistic risk factors
    risk_scores = []
    for i in range(n_samples):
        score = 0
        
        # Age factor
        if age[i] > 65: score += 0.25
        elif age[i] > 55: score += 0.15
        elif age[i] > 45: score += 0.08
        
        # Gender factor  
        if gender[i] == 'Male': score += 0.08
        
        # Blood pressure
        if blood_pressure[i] > 180: score += 0.25
        elif blood_pressure[i] > 160: score += 0.18
        elif blood_pressure[i] > 140: score += 0.12
        elif blood_pressure[i] > 130: score += 0.06
        
        # Cholesterol
        if cholesterol[i] > 280: score += 0.2
        elif cholesterol[i] > 240: score += 0.15
        elif cholesterol[i] > 200: score += 0.08
        
        # BMI
        if bmi[i] > 35: score += 0.15
        elif bmi[i] > 30: score += 0.1
        elif bmi[i] > 25: score += 0.05
        
        # Exercise protection
        if exercise_habits[i] == 'High': score -= 0.1
        elif exercise_habits[i] == 'Low': score += 0.08
        
        # Smoking risk
        if smoking[i] == 'Current': score += 0.15
        elif smoking[i] == 'Former': score += 0.05
        
        # Other factors
        if alcohol[i] == 'Heavy': score += 0.08
        if stress[i] == 'High': score += 0.06
        if sleep_hours[i] < 6 or sleep_hours[i] > 9: score += 0.04
        
        risk_scores.append(score)
    
    # Convert to binary outcome with some randomness
    risk_probs = np.array(risk_scores)
    # Add some noise to make it more realistic
    risk_probs += np.random.normal(0, 0.05, n_samples)
    target = (risk_probs > 0.35).astype(int)  # Threshold for heart disease
    
    # Create DataFrame
    data = pd.DataFrame({
        'Age': age.astype(int),
        'Gender': gender,
        'Blood Pressure': blood_pressure.astype(int),
        'Cholesterol Level': cholesterol.astype(int),
        'BMI': bmi.round(1),
        'Exercise Habits': exercise_habits,
        'Smoking': smoking,
        'Alcohol Consumption': alcohol,
        'Stress Level': stress,
        'Sleep Hours': sleep_hours.round(1),
        'Heart Disease': target
    })
    
    print(f"✅ Created synthetic dataset: {len(data)} samples")
    print(f"📊 Heart Disease distribution: {data['Heart Disease'].value_counts().to_dict()}")
    
    return data

def train_optimized_model(data):
    """Train an optimized Random Forest model for production deployment."""
    
    print("\n🤖 Training optimized Random Forest model...")
    
    # Prepare features and target
    X = data.drop('Heart Disease', axis=1)
    y = data['Heart Disease']
    
    print(f"📊 Training data: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"🎯 Target distribution: {y.value_counts().to_dict()}")
    
    # Handle categorical variables
    categorical_cols = ['Gender', 'Exercise Habits', 'Smoking', 'Alcohol Consumption', 'Stress Level']
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
    
    # Optimize hyperparameters with GridSearch
    print("🔍 Optimizing hyperparameters...")
    
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 15, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    }
    
    rf_base = RandomForestClassifier(random_state=42, n_jobs=-1)
    grid_search = GridSearchCV(
        rf_base, param_grid, cv=5, scoring='accuracy', 
        n_jobs=-1, verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"🏆 Best parameters: {grid_search.best_params_}")
    print(f"🎯 Best CV score: {grid_search.best_score_:.4f}")
    
    # Train final model with best parameters
    best_model = grid_search.best_estimator_
    
    # Evaluate model
    train_score = best_model.score(X_train, y_train)
    test_score = best_model.score(X_test, y_test)
    
    y_pred = best_model.predict(X_test)
    
    print(f"\n📈 Model Performance:")
    print(f"   Training Accuracy: {train_score:.4f}")
    print(f"   Test Accuracy: {test_score:.4f}")
    print(f"   Cross-validation: {grid_search.best_score_:.4f}")
    
    print(f"\n📊 Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X_processed.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n🔍 Top 10 Feature Importance:")
    print(feature_importance.head(10).to_string(index=False))
    
    return best_model, label_encoders, X_processed.columns.tolist(), feature_importance

def save_model_for_render(model, label_encoders, feature_columns, feature_importance):
    """Save the trained model in a format optimized for Render deployment."""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"heart_disease_render_optimized_{timestamp}.pkl"
    
    print(f"\n💾 Saving optimized model for Render deployment...")
    
    # Create comprehensive model package
    model_package = {
        'model': model,
        'label_encoders': label_encoders,
        'feature_columns': feature_columns,
        'feature_importance': feature_importance.to_dict('records'),
        'model_info': {
            'model_type': 'RandomForest',
            'training_date': datetime.now().isoformat(),
            'n_features': len(feature_columns),
            'optimized_for': 'render_deployment'
        }
    }
    
    # Save the model
    joblib.dump(model_package, model_filename, compress=3)
    
    print(f"✅ Model saved as: {model_filename}")
    print(f"📁 File size: {os.path.getsize(model_filename) / (1024*1024):.1f} MB")
    
    # Also save as the standard name for easy loading
    standard_filename = "heart_disease_render_optimized.pkl"
    joblib.dump(model_package, standard_filename, compress=3)
    print(f"✅ Also saved as: {standard_filename}")
    
    return model_filename, standard_filename

def main():
    """Main training pipeline."""
    
    print("🚀 Starting Heart Disease Model Training for Render Deployment")
    print("=" * 70)
    
    try:
        # Load data
        data = load_and_prepare_data()
        
        # Train model
        model, encoders, feature_cols, importance = train_optimized_model(data)
        
        # Save model
        model_file, standard_file = save_model_for_render(model, encoders, feature_cols, importance)
        
        print("\n" + "=" * 70)
        print("🎉 Model training completed successfully!")
        print(f"📁 Primary model file: {model_file}")
        print(f"📁 Standard model file: {standard_file}")
        print("\n🔄 Next steps:")
        print("   1. Commit and push the new model file to your repository")
        print("   2. Deploy to Render - it will automatically use the optimized model")
        print("   3. Test predictions - should now get high-quality ML predictions!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)