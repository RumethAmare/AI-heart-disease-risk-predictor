#!/usr/bin/env python3
"""
Heart Disease Model Training with Gender Included
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import joblib

def train_model_with_gender():
    """Train model with Gender included and 6 other columns dropped"""
    
    print("HEART DISEASE MODEL TRAINING WITH GENDER INCLUDED")
    print("=" * 60)
    
    # 1. Load data
    print("Loading dataset...")
    df = pd.read_csv('heart_disease_extended.csv')
    print(f"   Original dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # 2. Define columns to drop (keeping Gender)
    print("Column dropping configuration...")
    
    columns_to_drop = [
        'Diabetes',              # Low importance
        'Smoking',               # Low importance  
        'High Blood Pressure',   # Low importance
        'Low HDL Cholesterol',   # Low importance
        'Family Heart Disease',  # Low importance
        'High LDL Cholesterol',  # Low importance
        # Gender kept for medical relevance
    ]
    
    # Filter valid columns
    valid_drops = [col for col in columns_to_drop 
                  if col in df.columns and col != 'Heart Disease Status']
    
    if valid_drops:
        print(f"   Dropping columns: {valid_drops}")
        df = df.drop(columns=valid_drops)
        print(f"   Dataset after dropping: {df.shape[0]} rows, {df.shape[1]} columns")
    else:
        print("   No columns to drop - using all features")
    
    print(f"   Final features ({df.shape[1]-1}): {[col for col in df.columns if col != 'Heart Disease Status']}")
    
    # 3. Prepare features and target
    print("Preprocessing data...")
    X = df.drop('Heart Disease Status', axis=1)
    y = df['Heart Disease Status']
    
    print(f"   Features shape: {X.shape}")
    print(f"   Target shape: {y.shape}")
    
    # 4. Handle missing values
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    categorical_cols = X.select_dtypes(include=['object']).columns
    
    print(f"   Numerical features: {len(numeric_cols)}")
    print(f"   Categorical features: {len(categorical_cols)}")
    
    X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())
    X[categorical_cols] = X[categorical_cols].fillna(X[categorical_cols].mode().iloc[0])
    
    # 5. Encode categorical variables
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le
    
    # 6. Encode target
    target_encoder = LabelEncoder()
    y_encoded = target_encoder.fit_transform(y)
    print("   Categorical encoding completed")
    
    # 7. Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    print(f"   Train/test split: {X_train.shape[0]}/{X_test.shape[0]} samples")
    
    # 8. Apply SMOTE for class balancing
    print("Balancing classes with SMOTE...")
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    
    print(f"   Original training: {np.bincount(y_train)}")
    print(f"   Balanced training: {np.bincount(y_train_balanced)}")
    
    # 9. Train model
    print(f"Training Random Forest with {X.shape[1]} features...")
    rf = RandomForestClassifier(
        n_estimators=150,
        max_depth=20,
        min_samples_split=5,
        class_weight='balanced',
        random_state=42
    )
    
    rf.fit(X_train_balanced, y_train_balanced)
    print("   Model training completed!")
    
    # 10. Evaluate model
    print("Model Evaluation...")
    
    # Training performance
    y_train_pred = rf.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_f1 = f1_score(y_train, y_train_pred)
    
    # Test performance
    y_test_pred = rf.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred)
    
    print(f"   Training Accuracy: {train_accuracy:.4f}")
    print(f"   Test Accuracy: {test_accuracy:.4f}")
    print(f"   Training F1-Score: {train_f1:.4f}")
    print(f"   Test F1-Score: {test_f1:.4f}")
    
    # Feature importance of final model
    print("Top Feature Importance (Final Model):")
    final_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for i, (_, row) in enumerate(final_importance.head(10).iterrows()):
        print(f"   {i+1:2d}. {row['feature']:<25}: {row['importance']:.4f}")
    
    # 11. Save model
    print("Saving model...")
    model_data = {
        'model': rf,
        'label_encoders': label_encoders,
        'target_encoder': target_encoder,
        'feature_columns': X.columns.tolist(),
        'dropped_columns': valid_drops,
        'training_metrics': {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'train_f1_score': train_f1,
            'test_f1_score': test_f1,
            'features_used': X.shape[1],
            'samples_trained': X_train_balanced.shape[0]
        }
    }
    
    filename = 'heart_disease_model_with_gender.pkl'
    joblib.dump(model_data, filename)
    print(f"   Model saved as: {filename}")
    
    # 12. Summary
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"Features used: {X.shape[1]} (dropped {len(valid_drops)} columns)")
    print(f"Test Accuracy: {test_accuracy:.1%}")
    print(f"Test F1-Score: {test_f1:.3f}")
    print(f"Model ready for deployment!")
    
    return model_data

if __name__ == "__main__":
    model_data = train_model_with_gender()
    print("Model training completed successfully!")