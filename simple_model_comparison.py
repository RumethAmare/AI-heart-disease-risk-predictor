#!/usr/bin/env python3
"""
Simple Model Comparison for Heart Disease Dataset
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# Import models
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

def load_data():
    """Load and preprocess data"""
    print("Loading heart_disease_extended.csv...")
    df = pd.read_csv('heart_disease_extended.csv')
    
    # Drop the same columns as our optimized model
    columns_to_drop = [
        'Diabetes', 'Smoking', 'High Blood Pressure', 
        'Low HDL Cholesterol', 'Family Heart Disease', 'High LDL Cholesterol'
    ]
    
    valid_drops = [col for col in columns_to_drop if col in df.columns]
    df = df.drop(columns=valid_drops)
    
    # Separate features and target
    X = df.drop('Heart Disease Status', axis=1)
    y = df['Heart Disease Status']
    
    # Handle missing values
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    categorical_cols = X.select_dtypes(include=['object']).columns
    
    X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())
    X[categorical_cols] = X[categorical_cols].fillna(X[categorical_cols].mode().iloc[0])
    
    # Encode categorical variables
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
    
    # Encode target
    le_target = LabelEncoder()
    y_encoded = le_target.fit_transform(y)
    
    print(f"Dataset shape: {X.shape}")
    print(f"Features: {list(X.columns)}")
    return X, y_encoded

def test_models():
    """Test different models and compare performance"""
    
    # Load data
    X, y = load_data()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Apply SMOTE
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    
    # Scale data for some models
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_balanced)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Training samples: {X_train_balanced.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")
    print()
    
    # Define models
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'SVM': SVC(random_state=42),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Naive Bayes': GaussianNB()
    }
    
    results = []
    
    print("TRAINING AND EVALUATING MODELS")
    print("=" * 60)
    
    for name, model in models.items():
        print(f"Training {name}...")
        
        try:
            # Use scaled data for distance-based models
            if name in ['Logistic Regression', 'SVM', 'K-Nearest Neighbors']:
                model.fit(X_train_scaled, y_train_balanced)
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train_balanced, y_train_balanced)
                y_pred = model.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            
            results.append({
                'Model': name,
                'Accuracy': accuracy,
                'F1_Score': f1,
                'Precision': precision,
                'Recall': recall
            })
            
            print(f"  Accuracy: {accuracy:.3f} | F1: {f1:.3f} | Precision: {precision:.3f} | Recall: {recall:.3f}")
            
        except Exception as e:
            print(f"  Error: {str(e)}")
    
    return results

def generate_report(results):
    """Generate comprehensive report"""
    
    # Convert to DataFrame and sort by accuracy
    df = pd.DataFrame(results)
    df = df.sort_values('Accuracy', ascending=False)
    
    print("\n" + "=" * 80)
    print("MODEL PERFORMANCE COMPARISON REPORT")
    print("=" * 80)
    
    print("\nRankings by Accuracy:")
    print("-" * 50)
    for i, (_, row) in enumerate(df.iterrows(), 1):
        print(f"{i}. {row['Model']:<20} - Accuracy: {row['Accuracy']:.3f} ({row['Accuracy']*100:.1f}%)")
    
    print("\nDetailed Metrics:")
    print("-" * 80)
    print(f"{'Model':<20} {'Accuracy':<10} {'F1-Score':<10} {'Precision':<12} {'Recall':<10}")
    print("-" * 80)
    
    for _, row in df.iterrows():
        print(f"{row['Model']:<20} {row['Accuracy']:<10.3f} {row['F1_Score']:<10.3f} "
              f"{row['Precision']:<12.3f} {row['Recall']:<10.3f}")
    
    # Best model analysis
    best_model = df.iloc[0]
    print(f"\nBEST PERFORMING MODEL: {best_model['Model']}")
    print("-" * 40)
    print(f"Accuracy: {best_model['Accuracy']:.3f} ({best_model['Accuracy']*100:.1f}%)")
    print(f"F1-Score: {best_model['F1_Score']:.3f}")
    print(f"Precision: {best_model['Precision']:.3f}")
    print(f"Recall: {best_model['Recall']:.3f}")
    
    # Summary statistics
    print(f"\nSUMMARY STATISTICS:")
    print(f"Average Accuracy: {df['Accuracy'].mean():.3f}")
    print(f"Best Accuracy: {df['Accuracy'].max():.3f}")
    print(f"Worst Accuracy: {df['Accuracy'].min():.3f}")
    print(f"Standard Deviation: {df['Accuracy'].std():.3f}")
    
    # Save results
    df.to_csv('model_comparison_simple.csv', index=False)
    print(f"\nResults saved to 'model_comparison_simple.csv'")
    
    return df

if __name__ == "__main__":
    print("HEART DISEASE DATASET - MODEL COMPARISON")
    print("=" * 50)
    
    results = test_models()
    report_df = generate_report(results)
    
    print("\nComparison completed successfully!")