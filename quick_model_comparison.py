#!/usr/bin/env python3
"""
Robust Model Comparison for Heart Disease Dataset
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# Import models
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

def run_quick_comparison():
    """Quick model comparison with error handling"""
    
    print("HEART DISEASE MODEL COMPARISON")
    print("=" * 50)
    
    # Load data
    print("Loading data...")
    df = pd.read_csv('heart_disease_extended.csv')
    
    # Drop same columns as optimized model
    columns_to_drop = ['Diabetes', 'Smoking', 'High Blood Pressure', 
                      'Low HDL Cholesterol', 'Family Heart Disease', 'High LDL Cholesterol']
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
    
    # Prepare data
    X = df.drop('Heart Disease Status', axis=1)
    y = df['Heart Disease Status']
    
    # Quick preprocessing
    for col in X.select_dtypes(include=['object']).columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))
    
    y_encoded = LabelEncoder().fit_transform(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    
    # Balance data
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Training: {X_train_balanced.shape[0]} balanced samples")
    print(f"Testing: {X_test.shape[0]} samples")
    print()
    
    # Define models with simpler configurations
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=500),
        'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=42),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=3),
        'Naive Bayes': GaussianNB(),
        'SVM (Linear)': SVC(kernel='linear', C=0.1, random_state=42)
    }
    
    results = []
    
    # Scale data once for all models that need it
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_balanced)
    X_test_scaled = scaler.transform(X_test)
    
    print("Training models...")
    print("-" * 50)
    
    for name, model in models.items():
        try:
            print(f"Training {name}...", end=" ")
            
            # Use scaled data for specific models
            if name in ['Logistic Regression', 'K-Nearest Neighbors', 'SVM (Linear)']:
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
            
            print(f"Accuracy: {accuracy:.3f}")
            
        except Exception as e:
            print(f"Failed: {str(e)[:50]}...")
            continue
    
    # Create results DataFrame
    if results:
        df_results = pd.DataFrame(results)
        df_results = df_results.sort_values('Accuracy', ascending=False)
        
        print("\n" + "=" * 70)
        print("MODEL PERFORMANCE RESULTS")
        print("=" * 70)
        
        print(f"{'Rank':<5} {'Model':<20} {'Accuracy':<10} {'F1-Score':<10} {'Precision':<10} {'Recall':<10}")
        print("-" * 70)
        
        for i, (_, row) in enumerate(df_results.iterrows(), 1):
            print(f"{i:<5} {row['Model']:<20} {row['Accuracy']:<10.3f} {row['F1_Score']:<10.3f} "
                  f"{row['Precision']:<10.3f} {row['Recall']:<10.3f}")
        
        # Summary
        best_model = df_results.iloc[0]
        print(f"\nBEST MODEL: {best_model['Model']}")
        print(f"Accuracy: {best_model['Accuracy']:.3f} ({best_model['Accuracy']*100:.1f}%)")
        print(f"F1-Score: {best_model['F1_Score']:.3f}")
        
        print(f"\nSUMMARY:")
        print(f"Models tested: {len(df_results)}")
        print(f"Average accuracy: {df_results['Accuracy'].mean():.3f}")
        print(f"Best accuracy: {df_results['Accuracy'].max():.3f}")
        print(f"Accuracy range: {df_results['Accuracy'].min():.3f} - {df_results['Accuracy'].max():.3f}")
        
        # Save results
        df_results.to_csv('heart_disease_model_comparison.csv', index=False)
        print(f"\nResults saved to 'heart_disease_model_comparison.csv'")
        
        return df_results
    else:
        print("No models completed successfully!")
        return None

if __name__ == "__main__":
    results = run_quick_comparison()
    print("\nModel comparison completed!")