#!/usr/bin/env python3
"""
Heart Disease Model Training Script
Trains a machine learning model using the heart_disease_extended.csv dataset
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_data(csv_path):
    """Load and preprocess the heart disease dataset"""
    
    print("🔄 Loading and preprocessing data...")
    df = pd.read_csv(csv_path)
    
    print(f"📊 Dataset shape: {df.shape}")
    print(f"📊 Target distribution:")
    print(df['Heart Disease Status'].value_counts())
    
    # Handle missing values
    print("\n🧹 Handling missing values...")
    
    # For categorical columns, fill with mode
    categorical_columns = ['Gender', 'Exercise Habits', 'Smoking', 'Family Heart Disease', 
                          'Diabetes', 'High Blood Pressure', 'Low HDL Cholesterol', 
                          'High LDL Cholesterol', 'Alcohol Consumption', 'Stress Level', 
                          'Sugar Consumption']
    
    for col in categorical_columns:
        if col in df.columns and df[col].isnull().sum() > 0:
            mode_value = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
            df[col].fillna(mode_value, inplace=True)
            print(f"   ✓ {col}: filled {df[col].isnull().sum()} missing values with '{mode_value}'")
    
    # For numerical columns, fill with median
    numerical_columns = ['Age', 'Blood Pressure', 'Cholesterol Level', 'BMI', 
                        'Sleep Hours', 'Triglyceride Level', 'Fasting Blood Sugar', 
                        'CRP Level', 'Homocysteine Level']
    
    for col in numerical_columns:
        if col in df.columns and df[col].isnull().sum() > 0:
            median_value = df[col].median()
            df[col].fillna(median_value, inplace=True)
            print(f"   ✓ {col}: filled {df[col].isnull().sum()} missing values with median {median_value:.2f}")
    
    # Prepare features and target
    feature_columns = [col for col in df.columns if col != 'Heart Disease Status']
    X = df[feature_columns].copy()
    y = df['Heart Disease Status'].copy()
    
    print(f"\n📋 Features: {len(feature_columns)} columns")
    print(f"📋 Target: {y.name}")
    
    return X, y, feature_columns

def encode_categorical_features(X_train, X_test, categorical_columns):
    """Encode categorical features using LabelEncoder"""
    
    print("\n🔤 Encoding categorical features...")
    label_encoders = {}
    
    for col in categorical_columns:
        if col in X_train.columns:
            le = LabelEncoder()
            
            # Fit on training data
            X_train[col] = X_train[col].astype(str)
            le.fit(X_train[col])
            X_train[col] = le.transform(X_train[col])
            
            # Transform test data (handle unseen categories)
            X_test[col] = X_test[col].astype(str)
            X_test[col] = X_test[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
            X_test[col] = le.transform(X_test[col])
            
            label_encoders[col] = le
            print(f"   ✓ {col}: {len(le.classes_)} categories")
    
    return X_train, X_test, label_encoders

def train_model(X_train, y_train):
    """Train a Random Forest model"""
    
    print("\n🤖 Training Random Forest model...")
    
    # Configure the model with balanced parameters
    model = RandomForestClassifier(
        n_estimators=100,           # Good balance of performance and speed
        max_depth=15,               # Prevent overfitting
        min_samples_split=5,        # Minimum samples to split
        min_samples_leaf=2,         # Minimum samples in leaf
        class_weight='balanced',    # Handle any residual class imbalance
        random_state=42,           # Reproducible results
        n_jobs=-1                  # Use all CPU cores
    )
    
    # Train the model
    model.fit(X_train, y_train)
    
    print("   ✓ Model training completed")
    return model

def evaluate_model(model, X_train, y_train, X_test, y_test):
    """Evaluate the trained model"""
    
    print("\n📊 Evaluating model performance...")
    
    # Training predictions
    train_pred = model.predict(X_train)
    train_accuracy = accuracy_score(y_train, train_pred)
    
    # Test predictions
    test_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, test_pred)
    
    print(f"   📈 Training Accuracy: {train_accuracy:.3f}")
    print(f"   📈 Test Accuracy: {test_accuracy:.3f}")
    
    # Detailed classification report
    print("\n📋 Classification Report (Test Set):")
    print(classification_report(y_test, test_pred))
    
    # Confusion Matrix
    print("\n📋 Confusion Matrix:")
    cm = confusion_matrix(y_test, test_pred)
    print(f"   True Negative:  {cm[0][0]}")
    print(f"   False Positive: {cm[0][1]}")
    print(f"   False Negative: {cm[1][0]}")
    print(f"   True Positive:  {cm[1][1]}")
    
    return train_accuracy, test_accuracy

def save_model(model, label_encoders, target_encoder, feature_columns, filename):
    """Save the trained model and encoders"""
    
    print(f"\n💾 Saving model to {filename}...")
    
    model_data = {
        'model': model,
        'label_encoders': label_encoders,
        'target_encoder': target_encoder,
        'feature_columns': feature_columns,
        'model_type': 'RandomForestClassifier',
        'training_info': {
            'dataset_size': len(feature_columns),
            'features': feature_columns,
            'balanced_classes': True,
            'preprocessed': True
        }
    }
    
    joblib.dump(model_data, filename)
    print(f"   ✓ Model saved successfully")

def test_model_predictions(model_data, feature_columns):
    """Test the model with sample predictions"""
    
    print("\n🧪 Testing model predictions...")
    
    model = model_data['model']
    label_encoders = model_data['label_encoders']
    
    # Create test cases
    test_cases = [
        {
            'name': 'High Risk Patient',
            'data': {
                'Age': 70, 'Gender': 'Male', 'Blood Pressure': 180, 
                'Cholesterol Level': 300, 'Exercise Habits': 'Low', 
                'Smoking': 'Yes', 'Family Heart Disease': 'Yes', 
                'Diabetes': 'Yes', 'BMI': 32, 'High Blood Pressure': 'Yes',
                'Low HDL Cholesterol': 'Yes', 'High LDL Cholesterol': 'Yes',
                'Alcohol Consumption': 'High', 'Stress Level': 'High',
                'Sleep Hours': 5, 'Sugar Consumption': 'High',
                'Triglyceride Level': 250, 'Fasting Blood Sugar': 130,
                'CRP Level': 5.0, 'Homocysteine Level': 18.0
            }
        },
        {
            'name': 'Low Risk Patient',
            'data': {
                'Age': 30, 'Gender': 'Female', 'Blood Pressure': 110, 
                'Cholesterol Level': 160, 'Exercise Habits': 'High', 
                'Smoking': 'No', 'Family Heart Disease': 'No', 
                'Diabetes': 'No', 'BMI': 22, 'High Blood Pressure': 'No',
                'Low HDL Cholesterol': 'No', 'High LDL Cholesterol': 'No',
                'Alcohol Consumption': 'Low', 'Stress Level': 'Low',
                'Sleep Hours': 8, 'Sugar Consumption': 'Low',
                'Triglyceride Level': 120, 'Fasting Blood Sugar': 90,
                'CRP Level': 1.0, 'Homocysteine Level': 8.0
            }
        }
    ]
    
    for test_case in test_cases:
        # Prepare data
        df_test = pd.DataFrame([test_case['data']])
        
        # Encode categorical features
        for col, encoder in label_encoders.items():
            if col in df_test.columns:
                val = str(df_test[col].iloc[0])
                if val in encoder.classes_:
                    df_test[col] = encoder.transform([val])[0]
                else:
                    df_test[col] = 0
        
        # Ensure all features are present
        for col in feature_columns:
            if col not in df_test.columns:
                df_test[col] = 0
        
        # Reorder columns
        df_test = df_test[feature_columns]
        
        # Make prediction
        prediction = model.predict(df_test)[0]
        probability = model.predict_proba(df_test)[0]
        
        print(f"   {test_case['name']}:")
        print(f"      Prediction: {prediction}")
        print(f"      Probability: No={probability[0]:.3f}, Yes={probability[1]:.3f}")

def main():
    """Main training function"""
    
    print("🚀 HEART DISEASE MODEL TRAINING")
    print("=" * 50)
    
    try:
        # Step 1: Load and preprocess data
        X, y, feature_columns = load_and_preprocess_data('heart_disease_extended.csv')
        
        # Step 2: Split the data
        print("\n🔄 Splitting data...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        print(f"   📊 Training set: {len(X_train)} samples")
        print(f"   📊 Test set: {len(X_test)} samples")
        
        # Step 3: Encode categorical features
        categorical_columns = ['Gender', 'Exercise Habits', 'Smoking', 'Family Heart Disease', 
                              'Diabetes', 'High Blood Pressure', 'Low HDL Cholesterol', 
                              'High LDL Cholesterol', 'Alcohol Consumption', 'Stress Level', 
                              'Sugar Consumption']
        
        X_train, X_test, label_encoders = encode_categorical_features(
            X_train, X_test, categorical_columns
        )
        
        # Step 4: Encode target variable
        target_encoder = LabelEncoder()
        y_train_encoded = target_encoder.fit_transform(y_train)
        y_test_encoded = target_encoder.transform(y_test)
        
        # Step 5: Train the model
        model = train_model(X_train, y_train_encoded)
        
        # Step 6: Evaluate the model
        train_acc, test_acc = evaluate_model(
            model, X_train, y_train_encoded, X_test, y_test_encoded
        )
        
        # Step 7: Save the model
        save_model(model, label_encoders, target_encoder, feature_columns, 
                  'heart_disease_model.pkl')
        
        # Step 8: Test predictions
        model_data = {
            'model': model,
            'label_encoders': label_encoders,
            'target_encoder': target_encoder
        }
        test_model_predictions(model_data, feature_columns)
        
        print("\n🎉 MODEL TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 50)
        print(f"✅ Model saved as: heart_disease_model.pkl")
        print(f"✅ Training accuracy: {train_acc:.3f}")
        print(f"✅ Test accuracy: {test_acc:.3f}")
        print("✅ Ready for use in web application")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        exit(1)