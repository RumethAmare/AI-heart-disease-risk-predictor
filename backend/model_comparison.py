#!/usr/bin/env python3
"""
Comprehensive Model Comparison for Heart Disease Dataset
Training multiple ML algorithms and comparing their performance
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import joblib
import warnings
warnings.filterwarnings('ignore')

# Import different ML models
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

def load_and_preprocess_data():
    """Load and preprocess the heart disease dataset"""
    print("Loading and preprocessing dataset...")
    
    # Load data
    df = pd.read_csv('heart_disease_extended.csv')
    print(f"Dataset shape: {df.shape}")
    
    # Use the same feature set as our optimized model (14 features with Gender)
    columns_to_drop = [
        'Diabetes', 'Smoking', 'High Blood Pressure', 
        'Low HDL Cholesterol', 'Family Heart Disease', 'High LDL Cholesterol'
    ]
    
    valid_drops = [col for col in columns_to_drop 
                  if col in df.columns and col != 'Heart Disease Status']
    
    if valid_drops:
        df = df.drop(columns=valid_drops)
        print(f"Dropped columns: {valid_drops}")
    
    # Separate features and target
    X = df.drop('Heart Disease Status', axis=1)
    y = df['Heart Disease Status']
    
    print(f"Features: {list(X.columns)}")
    print(f"Target distribution: {y.value_counts().to_dict()}")
    
    # Handle missing values
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    categorical_cols = X.select_dtypes(include=['object']).columns
    
    X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())
    X[categorical_cols] = X[categorical_cols].fillna(X[categorical_cols].mode().iloc[0])
    
    # Encode categorical variables
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le
    
    # Encode target
    target_encoder = LabelEncoder()
    y_encoded = target_encoder.fit_transform(y)
    
    return X, y_encoded, label_encoders, target_encoder

def initialize_models():
    """Initialize all models to be tested"""
    models = {
        # Tree-based models
        'Random Forest': RandomForestClassifier(
            n_estimators=150, max_depth=20, min_samples_split=5,
            class_weight='balanced', random_state=42
        ),
        'Extra Trees': ExtraTreesClassifier(
            n_estimators=150, max_depth=20, min_samples_split=5,
            class_weight='balanced', random_state=42
        ),
        'Decision Tree': DecisionTreeClassifier(
            max_depth=20, min_samples_split=5,
            class_weight='balanced', random_state=42
        ),
        
        # Boosting models
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=100, max_depth=10, learning_rate=0.1, random_state=42
        ),
        'AdaBoost': AdaBoostClassifier(
            n_estimators=100, learning_rate=1.0, random_state=42
        ),
        'XGBoost': XGBClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            eval_metric='logloss', random_state=42
        ),
        'LightGBM': LGBMClassifier(
            n_estimators=100, max_depth=10, learning_rate=0.1,
            class_weight='balanced', random_state=42, verbose=-1
        ),
        
        # Linear models
        'Logistic Regression': LogisticRegression(
            class_weight='balanced', max_iter=1000, random_state=42
        ),
        
        # Distance-based models
        'K-Nearest Neighbors': KNeighborsClassifier(
            n_neighbors=5, weights='distance'
        ),
        
        # Probabilistic models
        'Naive Bayes': GaussianNB(),
        
        # Support Vector Machine
        'SVM (RBF)': SVC(
            kernel='rbf', class_weight='balanced', probability=True, random_state=42
        ),
        'SVM (Linear)': SVC(
            kernel='linear', class_weight='balanced', probability=True, random_state=42
        )
    }
    
    return models

def evaluate_model(model, model_name, X_train, X_test, y_train, y_test, X_train_balanced, y_train_balanced):
    """Evaluate a single model and return metrics"""
    print(f"Training {model_name}...")
    
    # Train on balanced data
    model.fit(X_train_balanced, y_train_balanced)
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Probabilities for AUC (if available)
    try:
        y_test_proba = model.predict_proba(X_test)[:, 1]
        auc_score = roc_auc_score(y_test, y_test_proba)
    except:
        auc_score = None
    
    # Cross-validation score
    try:
        cv_scores = cross_val_score(model, X_train_balanced, y_train_balanced, 
                                  cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42), 
                                  scoring='accuracy')
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
    except:
        cv_mean, cv_std = None, None
    
    # Calculate metrics
    metrics = {
        'Model': model_name,
        'Train_Accuracy': accuracy_score(y_train, y_train_pred),
        'Test_Accuracy': accuracy_score(y_test, y_test_pred),
        'Train_F1': f1_score(y_train, y_train_pred),
        'Test_F1': f1_score(y_test, y_test_pred),
        'Test_Precision': precision_score(y_test, y_test_pred),
        'Test_Recall': recall_score(y_test, y_test_pred),
        'Test_AUC': auc_score,
        'CV_Accuracy_Mean': cv_mean,
        'CV_Accuracy_Std': cv_std,
        'Overfitting': accuracy_score(y_train, y_train_pred) - accuracy_score(y_test, y_test_pred)
    }
    
    return metrics, model

def run_model_comparison():
    """Run comprehensive model comparison"""
    print("COMPREHENSIVE MODEL COMPARISON FOR HEART DISEASE PREDICTION")
    print("=" * 80)
    
    # Load and preprocess data
    X, y, label_encoders, target_encoder = load_and_preprocess_data()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features for models that need it
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Apply SMOTE for class balancing
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    X_train_balanced_scaled = scaler.fit_transform(X_train_balanced)
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Balanced training set: {X_train_balanced.shape[0]} samples")
    print()
    
    # Initialize models
    models = initialize_models()
    
    # Models that need scaled data
    scaled_models = ['Logistic Regression', 'K-Nearest Neighbors', 'SVM (RBF)', 'SVM (Linear)']
    
    results = []
    trained_models = {}
    
    # Evaluate each model
    for model_name, model in models.items():
        try:
            if model_name in scaled_models:
                # Use scaled data
                metrics, trained_model = evaluate_model(
                    model, model_name, X_train_scaled, X_test_scaled, 
                    y_train, y_test, X_train_balanced_scaled, y_train_balanced
                )
            else:
                # Use original data
                metrics, trained_model = evaluate_model(
                    model, model_name, X_train, X_test, 
                    y_train, y_test, X_train_balanced, y_train_balanced
                )
            
            results.append(metrics)
            trained_models[model_name] = trained_model
            
        except Exception as e:
            print(f"Error training {model_name}: {str(e)}")
            continue
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Sort by test accuracy
    results_df = results_df.sort_values('Test_Accuracy', ascending=False)
    
    return results_df, trained_models, X_test, y_test

def generate_report(results_df, trained_models, X_test, y_test):
    """Generate comprehensive performance report"""
    print("MODEL PERFORMANCE REPORT")
    print("=" * 80)
    
    # Summary table
    print("ACCURACY COMPARISON (Sorted by Test Accuracy)")
    print("-" * 80)
    print(f"{'Model':<20} {'Train Acc':<10} {'Test Acc':<10} {'Test F1':<10} {'Test AUC':<10} {'Overfitting':<12}")
    print("-" * 80)
    
    for _, row in results_df.iterrows():
        overfitting = row['Overfitting']
        overfitting_status = "High" if overfitting > 0.05 else "Low" if overfitting < 0.02 else "Medium"
        auc_str = f"{row['Test_AUC']:.3f}" if pd.notna(row['Test_AUC']) else "N/A"
        
        print(f"{row['Model']:<20} {row['Train_Accuracy']:<10.3f} {row['Test_Accuracy']:<10.3f} "
              f"{row['Test_F1']:<10.3f} {auc_str:<10} {overfitting_status:<12}")
    
    print("\n" + "=" * 80)
    
    # Top 3 models detailed analysis
    print("TOP 3 MODELS - DETAILED ANALYSIS")
    print("-" * 80)
    
    top_3 = results_df.head(3)
    for i, (_, row) in enumerate(top_3.iterrows(), 1):
        print(f"\n{i}. {row['Model']} - Best {'Performer' if i == 1 else 'Alternative'}")
        print(f"   Test Accuracy: {row['Test_Accuracy']:.3f} ({row['Test_Accuracy']*100:.1f}%)")
        print(f"   Test F1-Score: {row['Test_F1']:.3f}")
        print(f"   Test Precision: {row['Test_Precision']:.3f}")
        print(f"   Test Recall: {row['Test_Recall']:.3f}")
        if pd.notna(row['Test_AUC']):
            print(f"   AUC Score: {row['Test_AUC']:.3f}")
        if pd.notna(row['CV_Accuracy_Mean']):
            print(f"   CV Accuracy: {row['CV_Accuracy_Mean']:.3f} ± {row['CV_Accuracy_Std']:.3f}")
        print(f"   Overfitting: {row['Overfitting']:.3f} ({'Low' if row['Overfitting'] < 0.02 else 'Medium' if row['Overfitting'] < 0.05 else 'High'})")
    
    # Model categories performance
    print("\n" + "=" * 80)
    print("PERFORMANCE BY MODEL CATEGORY")
    print("-" * 80)
    
    # Tree-based models
    tree_models = results_df[results_df['Model'].str.contains('Tree|Forest')]['Test_Accuracy']
    if len(tree_models) > 0:
        print(f"Tree-based Models: Avg = {tree_models.mean():.3f}, Best = {tree_models.max():.3f}")
    
    # Boosting models
    boosting_models = results_df[results_df['Model'].str.contains('Boost|XGB|LightGBM')]['Test_Accuracy']
    if len(boosting_models) > 0:
        print(f"Boosting Models: Avg = {boosting_models.mean():.3f}, Best = {boosting_models.max():.3f}")
    
    # Linear models
    linear_models = results_df[results_df['Model'].str.contains('Logistic|SVM')]['Test_Accuracy']
    if len(linear_models) > 0:
        print(f"Linear Models: Avg = {linear_models.mean():.3f}, Best = {linear_models.max():.3f}")
    
    # Save best model
    best_model_name = results_df.iloc[0]['Model']
    best_model = trained_models[best_model_name]
    
    model_data = {
        'model': best_model,
        'model_name': best_model_name,
        'test_accuracy': results_df.iloc[0]['Test_Accuracy'],
        'test_f1': results_df.iloc[0]['Test_F1'],
        'comparison_results': results_df
    }
    
    joblib.dump(model_data, 'best_model_comparison.pkl')
    
    print(f"\nBest model ({best_model_name}) saved as 'best_model_comparison.pkl'")
    print(f"Best Test Accuracy: {results_df.iloc[0]['Test_Accuracy']:.3f} ({results_df.iloc[0]['Test_Accuracy']*100:.1f}%)")
    
    return results_df

def save_detailed_results(results_df):
    """Save detailed results to CSV"""
    results_df.to_csv('model_comparison_results.csv', index=False)
    print(f"\nDetailed results saved to 'model_comparison_results.csv'")
    
    # Create summary statistics
    summary = {
        'Total Models Tested': len(results_df),
        'Best Test Accuracy': results_df['Test_Accuracy'].max(),
        'Average Test Accuracy': results_df['Test_Accuracy'].mean(),
        'Best Model': results_df.iloc[0]['Model'],
        'Models with >95% Accuracy': len(results_df[results_df['Test_Accuracy'] > 0.95]),
        'Models with Low Overfitting': len(results_df[results_df['Overfitting'] < 0.02])
    }
    
    print("\nSUMMARY STATISTICS")
    print("-" * 40)
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key}: {value:.3f}")
        else:
            print(f"{key}: {value}")

if __name__ == "__main__":
    # Run the comprehensive comparison
    results_df, trained_models, X_test, y_test = run_model_comparison()
    
    # Generate report
    generate_report(results_df, trained_models, X_test, y_test)
    
    # Save results
    save_detailed_results(results_df)
    
    print("\nModel comparison completed successfully!")
    print("Check 'model_comparison_results.csv' for detailed results.")