#!/usr/bin/env python3
"""
Advanced Heart Disease Model Training with Column Dropping and Feature Analysis
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import joblib

def analyze_feature_importance():
    """Analyze feature importance to help decide which columns to drop"""
    
    print("🔍 FEATURE IMPORTANCE ANALYSIS")
    print("=" * 60)
    
    # Load and preprocess data quickly
    df = pd.read_csv('heart_disease_extended.csv')
    X = df.drop('Heart Disease Status', axis=1)
    y = df['Heart Disease Status']
    
    # Quick preprocessing for analysis
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    categorical_cols = X.select_dtypes(include=['object']).columns
    
    X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())
    X[categorical_cols] = X[categorical_cols].fillna(X[categorical_cols].mode().iloc[0])
    
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le
    
    # Encode target
    target_encoder = LabelEncoder()
    y_encoded = target_encoder.fit_transform(y)
    
    # Train quick model for feature importance
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y_encoded)
    
    # Get feature importance
    importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("📊 Feature Importance Ranking:")
    print("-" * 40)
    for i, (_, row) in enumerate(importance_df.iterrows()):
        status = "🟢 Keep" if row['importance'] > 0.05 else "🔴 Consider Dropping" if row['importance'] < 0.02 else "🟡 Maybe Drop"
        print(f"{i+1:2d}. {row['feature']:<25}: {row['importance']:.4f} {status}")
    
    # Suggest columns to drop
    low_importance = importance_df[importance_df['importance'] < 0.02]['feature'].tolist()
    if low_importance:
        print(f"\n💡 Suggested columns to drop (importance < 0.02): {low_importance}")
    else:
        print(f"\n✅ All features have decent importance (>= 0.02)")
    
    return importance_df

def train_with_column_dropping():
    """Train model with configurable column dropping"""
    
    print("\n🚀 TRAINING WITH COLUMN DROPPING")
    print("=" * 60)
    
    # 1. Load data
    print("📊 Loading dataset...")
    df = pd.read_csv('heart_disease_extended.csv')
    print(f"   Original dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # 2. CONFIGURE COLUMNS TO DROP
    print("\n🗑️ Column Dropping Configuration...")
    
    # Option 1: Drop by name (manual selection)
    manual_drops = [
        # Uncomment to drop these columns:
        # 'Sleep Hours',           # Often less predictive
        # 'Sugar Consumption',     # May be captured by other metrics
        # 'Alcohol Consumption',   # Lifestyle factor
        # 'Stress Level',          # Subjective measure
    ]
    
    # Option 2: Drop low importance columns (from analysis)
    # Uncomment to enable automatic dropping of low-importance features:
    drop_low_importance = False  # Set to True to enable
    importance_threshold = 0.02   # Drop features with importance < this value
    
    # Perform feature importance analysis if needed
    if drop_low_importance:
        print("   Analyzing feature importance for automatic dropping...")
        importance_df = analyze_feature_importance()
        auto_drops = importance_df[importance_df['importance'] < importance_threshold]['feature'].tolist()
        print(f"   Auto-drop candidates (importance < {importance_threshold}): {auto_drops}")
    else:
        auto_drops = []
    
    # Combine manual and automatic drops
    columns_to_drop = list(set(manual_drops + auto_drops))
    
    # Filter valid columns
    valid_drops = [col for col in columns_to_drop 
                  if col in df.columns and col != 'Heart Disease Status']
    
    if valid_drops:
        print(f"   🗑️ Dropping columns: {valid_drops}")
        df = df.drop(columns=valid_drops)
        print(f"   📊 Dataset after dropping: {df.shape[0]} rows, {df.shape[1]} columns")
    else:
        print("   ℹ️ No columns to drop - using all features")
    
    print(f"   📋 Final features ({df.shape[1]-1}): {[col for col in df.columns if col != 'Heart Disease Status']}")
    
    # 3. Check class distribution
    target_counts = df['Heart Disease Status'].value_counts()
    print(f"\n📈 Class distribution: {target_counts.to_dict()}")
    
    # 4. Prepare features and target
    print("\n🔧 Preprocessing data...")
    X = df.drop('Heart Disease Status', axis=1)
    y = df['Heart Disease Status']
    
    print(f"   Features shape: {X.shape}")
    print(f"   Target shape: {y.shape}")
    
    # 5. Handle missing values
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    categorical_cols = X.select_dtypes(include=['object']).columns
    
    print(f"   Numerical features: {len(numeric_cols)}")
    print(f"   Categorical features: {len(categorical_cols)}")
    
    X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())
    X[categorical_cols] = X[categorical_cols].fillna(X[categorical_cols].mode().iloc[0])
    
    # 6. Encode categorical variables
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le
    
    # 7. Encode target
    target_encoder = LabelEncoder()
    y_encoded = target_encoder.fit_transform(y)
    print("   ✅ Categorical encoding completed")
    
    # 8. Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    print(f"   📊 Train/test split: {X_train.shape[0]}/{X_test.shape[0]} samples")
    
    # 9. Apply SMOTE for class balancing
    print("\n⚖️ Balancing classes with SMOTE...")
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    
    print(f"   Original training: {np.bincount(y_train)}")
    print(f"   Balanced training: {np.bincount(y_train_balanced)}")
    
    # 10. Train model
    print(f"\n🤖 Training Random Forest with {X.shape[1]} features...")
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        class_weight='balanced',
        random_state=42
    )
    
    rf.fit(X_train_balanced, y_train_balanced)
    print("   ✅ Model training completed!")
    
    # 11. Evaluate model
    print(f"\n📊 Model Evaluation...")
    
    # Training performance
    y_train_pred = rf.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_f1 = f1_score(y_train, y_train_pred)
    
    # Test performance
    y_test_pred = rf.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred)
    
    print(f"   🎯 Training Accuracy: {train_accuracy:.4f}")
    print(f"   🎯 Test Accuracy: {test_accuracy:.4f}")
    print(f"   🎯 Training F1-Score: {train_f1:.4f}")
    print(f"   🎯 Test F1-Score: {test_f1:.4f}")
    
    # Feature importance of final model
    print(f"\n🔍 Top Feature Importance (Final Model):")
    final_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for i, (_, row) in enumerate(final_importance.head(10).iterrows()):
        print(f"   {i+1:2d}. {row['feature']:<25}: {row['importance']:.4f}")
    
    # 12. Save model
    print(f"\n💾 Saving model...")
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
    
    filename = 'heart_disease_model_dropped_features.pkl'
    joblib.dump(model_data, filename)
    print(f"   ✅ Model saved as: {filename}")
    
    # 13. Summary
    print(f"\n🎉 TRAINING COMPLETE!")
    print("=" * 60)
    print(f"✅ Features used: {X.shape[1]} (dropped {len(valid_drops)} columns)")
    print(f"✅ Test Accuracy: {test_accuracy:.1%}")
    print(f"✅ Test F1-Score: {test_f1:.3f}")
    print(f"✅ Model ready for deployment!")
    
    return model_data

if __name__ == "__main__":
    print("🔬 HEART DISEASE MODEL TRAINING WITH FEATURE SELECTION")
    print("=" * 70)
    
    # First analyze feature importance
    importance_analysis = analyze_feature_importance()
    
    # Then train with column dropping
    model_data = train_with_column_dropping()
    
    print(f"\n🔗 Next steps:")
    print(f"1. Update Flask app to use: heart_disease_model_dropped_features.pkl")
    print(f"2. Test the model with reduced features")
    print(f"3. Deploy if performance is satisfactory")