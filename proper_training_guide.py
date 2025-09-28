#!/usr/bin/env python3
"""
PROPER Heart Disease Dataset Training Guide
FDM Mini Project 2025 - Complete Training Pipeline

This script demonstrates the CORRECT way to train a heart disease prediction model
with proper data preprocessing, class balancing, hyperparameter tuning, and evaluation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                           confusion_matrix, classification_report, roc_auc_score, roc_curve)
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek, SMOTEENN
import joblib
import warnings
warnings.filterwarnings('ignore')

class ProperHeartDiseaseTraining:
    """
    COMPREHENSIVE Heart Disease Model Training with Advanced Techniques
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.best_model = None
        self.evaluation_results = {}
        
    def load_and_explore_data(self, data_path):
        """
        Step 1: Load and thoroughly explore the dataset
        """
        print("🔍 STEP 1: DATA EXPLORATION")
        print("=" * 60)
        
        # Load data
        df = pd.read_csv(data_path)
        print(f"📊 Dataset Shape: {df.shape}")
        print(f"📈 Total Records: {len(df):,}")
        print(f"📋 Total Features: {len(df.columns)}")
        
        # Display basic info
        print(f"\n📋 DATASET COLUMNS:")
        for i, col in enumerate(df.columns, 1):
            dtype = df[col].dtype
            unique_vals = df[col].nunique()
            print(f"{i:2d}. {col:<25} | Type: {str(dtype):<10} | Unique: {unique_vals:>4}")
        
        # Check target variable
        target_col = 'Heart Disease Status'
        if target_col in df.columns:
            print(f"\n🎯 TARGET VARIABLE ANALYSIS:")
            print(f"Target Column: '{target_col}'")
            
            value_counts = df[target_col].value_counts()
            print(f"Class Distribution:")
            for class_name, count in value_counts.items():
                percentage = (count / len(df)) * 100
                print(f"  {class_name}: {count:,} ({percentage:.2f}%)")
            
            # Calculate class imbalance ratio
            minority_class = min(value_counts)
            majority_class = max(value_counts)
            imbalance_ratio = majority_class / minority_class
            print(f"⚖️ Imbalance Ratio: {imbalance_ratio:.2f}:1")
            
            if imbalance_ratio > 2:
                print("⚠️  WARNING: Dataset is imbalanced! Will apply balancing techniques.")
        
        # Check for missing values
        print(f"\n🔍 MISSING VALUES ANALYSIS:")
        missing_data = df.isnull().sum()
        if missing_data.sum() > 0:
            print("Columns with missing values:")
            for col, missing_count in missing_data[missing_data > 0].items():
                percentage = (missing_count / len(df)) * 100
                print(f"  {col}: {missing_count} ({percentage:.2f}%)")
        else:
            print("✅ No missing values found!")
        
        # Data types analysis
        print(f"\n📈 DATA TYPES BREAKDOWN:")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        if target_col in numeric_cols:
            numeric_cols.remove(target_col)
        if target_col in categorical_cols:
            categorical_cols.remove(target_col)
            
        print(f"Numeric Features ({len(numeric_cols)}): {numeric_cols}")
        print(f"Categorical Features ({len(categorical_cols)}): {categorical_cols}")
        
        # Statistical summary
        print(f"\n📊 STATISTICAL SUMMARY:")
        print(df.describe())
        
        return df, numeric_cols, categorical_cols
    
    def advanced_preprocessing(self, df, numeric_cols, categorical_cols, target_col='Heart Disease Status'):
        """
        Step 2: Advanced Data Preprocessing
        """
        print(f"\n🔧 STEP 2: ADVANCED DATA PREPROCESSING")
        print("=" * 60)
        
        # Separate features and target
        X = df.drop(target_col, axis=1)
        y = df[target_col]
        
        # Handle missing values with advanced techniques
        print("🔄 Handling missing values...")
        
        # For numeric columns: Use KNN Imputer (more sophisticated than mean/median)
        if len(numeric_cols) > 0:
            knn_imputer = KNNImputer(n_neighbors=5)
            X[numeric_cols] = knn_imputer.fit_transform(X[numeric_cols])
            print(f"  ✅ KNN imputation applied to {len(numeric_cols)} numeric columns")
        
        # For categorical columns: Use mode imputation
        if len(categorical_cols) > 0:
            cat_imputer = SimpleImputer(strategy='most_frequent')
            X[categorical_cols] = cat_imputer.fit_transform(X[categorical_cols])
            print(f"  ✅ Mode imputation applied to {len(categorical_cols)} categorical columns")
        
        # Advanced categorical encoding
        print("🏷️ Encoding categorical variables...")
        for col in categorical_cols:
            if X[col].nunique() <= 10:  # Use Label Encoding for low cardinality
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                self.encoders[col] = le
                print(f"  ✅ Label encoded: {col} ({X[col].nunique()} unique values)")
            else:  # Use One-Hot Encoding for high cardinality
                dummies = pd.get_dummies(X[col], prefix=col)
                X = pd.concat([X.drop(col, axis=1), dummies], axis=1)
                print(f"  ✅ One-hot encoded: {col} ({len(dummies.columns)} new columns)")
        
        # Encode target variable
        target_encoder = LabelEncoder()
        y_encoded = target_encoder.fit_transform(y)
        self.encoders['target'] = target_encoder
        
        print(f"🎯 Target encoded: {dict(zip(target_encoder.classes_, range(len(target_encoder.classes_))))}")
        
        return X, y_encoded
    
    def handle_class_imbalance(self, X, y):
        """
        Step 3: Handle Class Imbalance with Multiple Techniques
        """
        print(f"\n⚖️ STEP 3: HANDLING CLASS IMBALANCE")
        print("=" * 60)
        
        # Check current class distribution
        unique, counts = np.unique(y, return_counts=True)
        print(f"Original distribution: {dict(zip(unique, counts))}")
        
        imbalance_ratio = max(counts) / min(counts)
        
        if imbalance_ratio <= 1.5:
            print("✅ Dataset is relatively balanced. No rebalancing needed.")
            return {'original': (X, y)}
        
        print(f"⚠️ Imbalance ratio: {imbalance_ratio:.2f}:1 - Applying rebalancing techniques...")
        
        balanced_datasets = {}
        
        # 1. SMOTE (Synthetic Minority Oversampling)
        try:
            smote = SMOTE(random_state=42)
            X_smote, y_smote = smote.fit_resample(X, y)
            balanced_datasets['SMOTE'] = (X_smote, y_smote)
            print("  ✅ SMOTE applied")
        except Exception as e:
            print(f"  ❌ SMOTE failed: {e}")
        
        # 2. ADASYN (Adaptive Synthetic Sampling)
        try:
            adasyn = ADASYN(random_state=42)
            X_adasyn, y_adasyn = adasyn.fit_resample(X, y)
            balanced_datasets['ADASYN'] = (X_adasyn, y_adasyn)
            print("  ✅ ADASYN applied")
        except Exception as e:
            print(f"  ❌ ADASYN failed: {e}")
        
        # 3. Random Undersampling
        try:
            undersampler = RandomUnderSampler(random_state=42)
            X_under, y_under = undersampler.fit_resample(X, y)
            balanced_datasets['Undersampling'] = (X_under, y_under)
            print("  ✅ Random Undersampling applied")
        except Exception as e:
            print(f"  ❌ Undersampling failed: {e}")
        
        # 4. SMOTETomek (Combined over and under sampling)
        try:
            smote_tomek = SMOTETomek(random_state=42)
            X_smote_tomek, y_smote_tomek = smote_tomek.fit_resample(X, y)
            balanced_datasets['SMOTETomek'] = (X_smote_tomek, y_smote_tomek)
            print("  ✅ SMOTETomek applied")
        except Exception as e:
            print(f"  ❌ SMOTETomek failed: {e}")
        
        # Add original dataset for comparison
        balanced_datasets['Original'] = (X, y)
        
        # Show new distributions
        print(f"\n📊 Rebalanced Dataset Sizes:")
        for name, (X_bal, y_bal) in balanced_datasets.items():
            unique, counts = np.unique(y_bal, return_counts=True)
            print(f"  {name}: {dict(zip(unique, counts))} (Total: {len(y_bal)})")
        
        return balanced_datasets
    
    def advanced_model_training(self, balanced_datasets):
        """
        Step 4: Advanced Model Training with Hyperparameter Tuning
        """
        print(f"\n🤖 STEP 4: ADVANCED MODEL TRAINING")
        print("=" * 60)
        
        # Define models with hyperparameter grids
        model_configs = {
            'Random Forest': {
                'model': RandomForestClassifier(random_state=42),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2],
                    'class_weight': ['balanced', None]
                }
            },
            'Gradient Boosting': {
                'model': GradientBoostingClassifier(random_state=42),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 1.0]
                }
            },
            'Logistic Regression': {
                'model': LogisticRegression(random_state=42, max_iter=1000),
                'params': {
                    'C': [0.1, 1, 10],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear', 'saga'],
                    'class_weight': ['balanced', None]
                }
            },
            'SVM': {
                'model': SVC(random_state=42, probability=True),
                'params': {
                    'C': [0.1, 1, 10],
                    'kernel': ['rbf', 'linear'],
                    'gamma': ['scale', 'auto'],
                    'class_weight': ['balanced', None]
                }
            }
        }
        
        results = {}
        
        # Train models on each balanced dataset
        for balance_method, (X_data, y_data) in balanced_datasets.items():
            print(f"\n🔄 Training on {balance_method} dataset...")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_data, y_data, test_size=0.2, random_state=42, stratify=y_data
            )
            
            method_results = {}
            
            for model_name, config in model_configs.items():
                print(f"  🎯 Training {model_name}...")
                
                # Scale features for models that need it
                if model_name in ['Logistic Regression', 'SVM']:
                    scaler = RobustScaler()  # More robust to outliers than StandardScaler
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)
                    train_features, test_features = X_train_scaled, X_test_scaled
                else:
                    train_features, test_features = X_train, X_test
                    scaler = None
                
                # Hyperparameter tuning with GridSearchCV
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                
                grid_search = GridSearchCV(
                    config['model'], 
                    config['params'],
                    cv=cv,
                    scoring='f1',  # Use F1-score for imbalanced datasets
                    n_jobs=-1,
                    verbose=0
                )
                
                grid_search.fit(train_features, y_train)
                
                # Get best model and make predictions
                best_model = grid_search.best_estimator_
                y_pred = best_model.predict(test_features)
                y_proba = best_model.predict_proba(test_features)[:, 1]
                
                # Calculate comprehensive metrics
                metrics = {
                    'accuracy': accuracy_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred, average='weighted'),
                    'recall': recall_score(y_test, y_pred, average='weighted'),
                    'f1_score': f1_score(y_test, y_pred, average='weighted'),
                    'roc_auc': roc_auc_score(y_test, y_proba),
                    'best_params': grid_search.best_params_,
                    'model': best_model,
                    'scaler': scaler
                }
                
                method_results[model_name] = metrics
                
                print(f"    ✅ F1-Score: {metrics['f1_score']:.4f} | ROC-AUC: {metrics['roc_auc']:.4f}")
            
            results[balance_method] = method_results
        
        return results
    
    def evaluate_and_select_best_model(self, results):
        """
        Step 5: Comprehensive Model Evaluation and Selection
        """
        print(f"\n🏆 STEP 5: MODEL EVALUATION AND SELECTION")
        print("=" * 60)
        
        # Create comprehensive results table
        evaluation_data = []
        
        for balance_method, models in results.items():
            for model_name, metrics in models.items():
                evaluation_data.append({
                    'Balance Method': balance_method,
                    'Model': model_name,
                    'F1-Score': metrics['f1_score'],
                    'ROC-AUC': metrics['roc_auc'],
                    'Accuracy': metrics['accuracy'],
                    'Precision': metrics['precision'],
                    'Recall': metrics['recall']
                })
        
        eval_df = pd.DataFrame(evaluation_data)
        
        # Sort by F1-Score (most important for imbalanced datasets)
        eval_df_sorted = eval_df.sort_values('F1-Score', ascending=False)
        
        print("📊 COMPREHENSIVE MODEL COMPARISON:")
        print(eval_df_sorted.round(4))
        
        # Select best model
        best_result = eval_df_sorted.iloc[0]
        best_balance_method = best_result['Balance Method']
        best_model_name = best_result['Model']
        
        print(f"\n🥇 BEST MODEL SELECTED:")
        print(f"   Balance Method: {best_balance_method}")
        print(f"   Model: {best_model_name}")
        print(f"   F1-Score: {best_result['F1-Score']:.4f}")
        print(f"   ROC-AUC: {best_result['ROC-AUC']:.4f}")
        print(f"   Accuracy: {best_result['Accuracy']:.4f}")
        
        # Get the actual best model
        self.best_model = results[best_balance_method][best_model_name]['model']
        self.best_scaler = results[best_balance_method][best_model_name]['scaler']
        best_params = results[best_balance_method][best_model_name]['best_params']
        
        print(f"\n🔧 BEST HYPERPARAMETERS:")
        for param, value in best_params.items():
            print(f"   {param}: {value}")
        
        return eval_df_sorted, best_balance_method, best_model_name
    
    def save_best_model(self, best_balance_method, best_model_name):
        """
        Step 6: Save the Best Model
        """
        print(f"\n💾 STEP 6: SAVING BEST MODEL")
        print("=" * 60)
        
        model_data = {
            'model': self.best_model,
            'scaler': self.best_scaler,
            'encoders': self.encoders,
            'balance_method': best_balance_method,
            'model_name': best_model_name,
            'training_info': {
                'timestamp': pd.Timestamp.now(),
                'balance_method_used': best_balance_method,
                'model_algorithm': best_model_name
            }
        }
        
        filename = f'best_heart_disease_model_{best_model_name.lower().replace(" ", "_")}.pkl'
        joblib.dump(model_data, filename)
        
        print(f"✅ Best model saved as: {filename}")
        print(f"   Algorithm: {best_model_name}")
        print(f"   Balance Method: {best_balance_method}")
        
        return filename
    
    def complete_training_pipeline(self, data_path='heart_disease.csv'):
        """
        Execute the complete proper training pipeline
        """
        print("🚀 COMPLETE HEART DISEASE MODEL TRAINING PIPELINE")
        print("=" * 80)
        print("This demonstrates the PROPER way to train a heart disease prediction model")
        print("=" * 80)
        
        # Step 1: Data Exploration
        df, numeric_cols, categorical_cols = self.load_and_explore_data(data_path)
        
        # Step 2: Advanced Preprocessing
        X, y = self.advanced_preprocessing(df, numeric_cols, categorical_cols)
        
        # Step 3: Handle Class Imbalance
        balanced_datasets = self.handle_class_imbalance(X, y)
        
        # Step 4: Advanced Model Training
        results = self.advanced_model_training(balanced_datasets)
        
        # Step 5: Model Evaluation and Selection
        eval_results, best_balance, best_model = self.evaluate_and_select_best_model(results)
        
        # Step 6: Save Best Model
        model_filename = self.save_best_model(best_balance, best_model)
        
        print(f"\n🎉 TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"✅ Best Model: {best_model} with {best_balance} balancing")
        print(f"✅ Model saved as: {model_filename}")
        print(f"✅ Ready for production use!")
        
        return eval_results, model_filename


def main():
    """
    Main function to demonstrate proper heart disease dataset training
    """
    print("💡 HEART DISEASE DATASET - PROPER TRAINING GUIDE")
    print("=" * 80)
    print("This script shows you the CORRECT way to train a heart disease prediction model")
    print("Key improvements over basic training:")
    print("  1. 📊 Thorough data exploration and analysis")
    print("  2. 🔧 Advanced preprocessing techniques")
    print("  3. ⚖️ Multiple class imbalance handling methods")
    print("  4. 🎯 Hyperparameter tuning with GridSearchCV")
    print("  5. 📈 Comprehensive model evaluation")
    print("  6. 🏆 Automatic best model selection")
    print("=" * 80)
    
    # Initialize the training class
    trainer = ProperHeartDiseaseTraining()
    
    # Run complete training pipeline
    results, model_file = trainer.complete_training_pipeline()
    
    print(f"\n📋 TRAINING SUMMARY:")
    print("Top 5 Model Configurations:")
    print(results.head()[['Model', 'Balance Method', 'F1-Score', 'ROC-AUC', 'Accuracy']].round(4))


if __name__ == "__main__":
    main()