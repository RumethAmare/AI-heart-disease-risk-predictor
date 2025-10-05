#!/usr/bin/env python3
"""
Train Heart Disease Model with Low-Importance Column Dropping
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import joblib

def train_with_dropped_columns():
    """Train model with low-importance columns removed"""
    
    print("🗑️ TRAINING WITH COLUMN DROPPING ENABLED")
    print("=" * 60)
    
    # 1. Load data
    print("📊 Loading dataset...")
    df = pd.read_csv('heart_disease_extended.csv')
    print(f"   Original dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # 2. Define columns to drop based on feature importance analysis
    print("\n🔍 Dropping Low-Importance Columns...")
    
    # Columns with importance < 0.02 (from analysis)
    low_importance_columns = [
        'Diabetes',              # 0.0149 importance
        'Smoking',               # 0.0147 importance  
        'High Blood Pressure',   # 0.0141 importance
        'Low HDL Cholesterol',   # 0.0138 importance
        'Family Heart Disease',  # 0.0135 importance
        'High LDL Cholesterol',  # 0.0132 importance
        'Gender',                # 0.0128 importance
    ]
    
    # Additional optional drops (moderate importance 0.02-0.025)
    optional_drops = [
        # 'Exercise Habits',       # 0.0240 importance
        # 'Sugar Consumption',     # 0.0232 importance
        # 'Alcohol Consumption',   # 0.0228 importance
        # 'Stress Level',          # 0.0210 importance
    ]
    
    # Combine drops
    columns_to_drop = low_importance_columns + optional_drops
    
    # Filter valid columns
    valid_drops = [col for col in columns_to_drop 
                  if col in df.columns and col != 'Heart Disease Status']
    
    print(f"   🗑️ Dropping {len(valid_drops)} columns: {valid_drops}")
    df_original_shape = df.shape
    df = df.drop(columns=valid_drops)
    print(f"   📊 Dataset: {df_original_shape} → {df.shape}")
    print(f"   📋 Remaining features: {[col for col in df.columns if col != 'Heart Disease Status']}")
    
    # 3. Prepare data
    print(f"\n🔧 Preprocessing reduced dataset...")
    X = df.drop('Heart Disease Status', axis=1)
    y = df['Heart Disease Status']
    
    print(f"   Features: {X.shape[1]} columns")
    print(f"   Samples: {X.shape[0]} records")
    
    # 4. Handle missing values
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    categorical_cols = X.select_dtypes(include=['object']).columns
    
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
    
    # 7. Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # 8. Balance with SMOTE
    print(f"\n⚖️ Balancing classes...")
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    
    print(f"   Balanced training: {np.bincount(y_train_balanced)}")
    
    # 9. Train model
    print(f"\n🤖 Training RandomForest with {X.shape[1]} features...")
    rf = RandomForestClassifier(
        n_estimators=150,        # Slightly more trees
        max_depth=20,
        min_samples_split=5,
        class_weight='balanced',
        random_state=42
    )
    
    rf.fit(X_train_balanced, y_train_balanced)
    
    # 10. Evaluate
    print(f"\n📊 Evaluating reduced-feature model...")
    
    y_train_pred = rf.predict(X_train)
    y_test_pred = rf.predict(X_test)
    
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    train_f1 = f1_score(y_train, y_train_pred)
    test_f1 = f1_score(y_test, y_test_pred)
    
    print(f"   🎯 Training Accuracy: {train_accuracy:.4f}")
    print(f"   🎯 Test Accuracy: {test_accuracy:.4f}")
    print(f"   🎯 Training F1-Score: {train_f1:.4f}")
    print(f"   🎯 Test F1-Score: {test_f1:.4f}")
    
    # 11. Feature importance of reduced model
    print(f"\n🔍 Feature Importance (Reduced Model):")
    importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for i, (_, row) in enumerate(importance_df.iterrows()):
        print(f"   {i+1:2d}. {row['feature']:<25}: {row['importance']:.4f}")
    
    # 12. Save reduced model
    print(f"\n💾 Saving reduced-feature model...")
    
    model_data = {
        'model': rf,
        'label_encoders': label_encoders,
        'target_encoder': target_encoder,
        'feature_columns': X.columns.tolist(),
        'dropped_columns': valid_drops,
        'model_info': {
            'features_used': len(X.columns),
            'features_dropped': len(valid_drops),
            'original_features': df_original_shape[1] - 1,  # Excluding target
            'reduction_ratio': len(valid_drops) / (df_original_shape[1] - 1)
        },
        'training_metrics': {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'train_f1_score': train_f1,
            'test_f1_score': test_f1
        }
    }
    
    model_filename = 'heart_disease_model_reduced.pkl'
    joblib.dump(model_data, model_filename)
    print(f"   ✅ Saved as: {model_filename}")
    
    # 13. Summary comparison
    print(f"\n🎉 REDUCED MODEL TRAINING COMPLETE!")
    print("=" * 60)
    print(f"📊 FEATURE REDUCTION SUMMARY:")
    print(f"   Original features: {df_original_shape[1]-1}")
    print(f"   Features used: {len(X.columns)}")
    print(f"   Features dropped: {len(valid_drops)}")
    print(f"   Reduction: {len(valid_drops)/(df_original_shape[1]-1)*100:.1f}%")
    print(f"\n📈 PERFORMANCE:")
    print(f"   Test Accuracy: {test_accuracy:.1%}")
    print(f"   Test F1-Score: {test_f1:.3f}")
    print(f"\n🗑️ DROPPED COLUMNS: {valid_drops}")
    
    # 14. Test sample prediction
    print(f"\n🧪 Testing reduced model...")
    sample_data = {
        'Age': 55, 'Blood Pressure': 140, 'Cholesterol Level': 220,
        'Exercise Habits': 'Low', 'BMI': 28.5,
        'Alcohol Consumption': 'Medium', 'Stress Level': 'High',
        'Sleep Hours': 6, 'Sugar Consumption': 'High',
        'Triglyceride Level': 180, 'Fasting Blood Sugar': 110,
        'CRP Level': 3.5, 'Homocysteine Level': 12.0
    }
    
    # Convert to DataFrame and process
    sample_df = pd.DataFrame([sample_data])
    
    # Fill missing columns with defaults
    for col in X.columns:
        if col not in sample_df.columns:
            if col in numeric_cols:
                sample_df[col] = X[col].median()
            else:
                sample_df[col] = X[col].mode().iloc[0] if len(X[col].mode()) > 0 else 0
    
    # Encode categorical variables
    for col in categorical_cols:
        if col in sample_df.columns:
            val = str(sample_df[col].iloc[0])
            if val in label_encoders[col].classes_:
                sample_df[col] = label_encoders[col].transform([val])[0]
            else:
                sample_df[col] = 0
    
    # Ensure correct column order
    sample_df = sample_df[X.columns]
    
    # Make prediction
    prediction = rf.predict(sample_df)[0]
    probabilities = rf.predict_proba(sample_df)[0]
    
    predicted_class = target_encoder.inverse_transform([prediction])[0]
    confidence = max(probabilities) * 100
    
    print(f"   Sample prediction: {predicted_class} (Confidence: {confidence:.1f}%)")
    
    return model_data

if __name__ == "__main__":
    print("🔧 HEART DISEASE MODEL WITH FEATURE REDUCTION")
    print("=" * 70)
    
    model = train_with_dropped_columns()
    
    print(f"\n✅ Reduced model ready!")
    print(f"📁 Saved as: heart_disease_model_reduced.pkl") 
    print(f"🚀 Update Flask app to use the reduced model for faster predictions!")