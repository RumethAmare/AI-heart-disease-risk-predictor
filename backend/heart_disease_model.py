#!/usr/bin/env python3
"""
Heart Disease Prediction Model Training
FDM Mini Project 2025 - Machine Learning Model Component
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import joblib
import warnings
warnings.filterwarnings('ignore')

class HeartDiseasePredictor:
    """
    Heart Disease Prediction Model using advanced ML techniques.
    """
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = None
        self.model_metrics = {}
        
    def preprocess_data(self, data, is_training=True):
        """
        Preprocess the heart disease data for ML model.
        
        Args:
            data (DataFrame): Raw heart disease data
            is_training (bool): Whether this is training data or new prediction data
            
        Returns:
            DataFrame: Preprocessed features ready for ML
        """
        df = data.copy()
        
        # Handle missing values
        numeric_imputer = SimpleImputer(strategy='median')
        categorical_imputer = SimpleImputer(strategy='most_frequent')
        
        # Identify numeric and categorical columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
        
        # Remove target column from features if present
        if 'Heart Disease Status' in numeric_columns:
            numeric_columns.remove('Heart Disease Status')
        if 'Heart Disease Status' in categorical_columns:
            categorical_columns.remove('Heart Disease Status')
        
        # Impute missing values
        if numeric_columns:
            df[numeric_columns] = numeric_imputer.fit_transform(df[numeric_columns])
        
        if categorical_columns:
            df[categorical_columns] = categorical_imputer.fit_transform(df[categorical_columns])
        
        # Encode categorical variables
        for col in categorical_columns:
            if is_training:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
            else:
                if col in self.label_encoders:
                    # Handle unseen categories
                    unique_values = set(df[col].astype(str))
                    known_values = set(self.label_encoders[col].classes_)
                    
                    # Replace unseen values with most common value
                    for val in unique_values - known_values:
                        most_common = self.label_encoders[col].classes_[0]
                        df.loc[df[col] == val, col] = most_common
                    
                    df[col] = self.label_encoders[col].transform(df[col].astype(str))
        
        # Store feature columns for consistency
        if is_training:
            feature_cols = [col for col in df.columns if col != 'Heart Disease Status']
            self.feature_columns = feature_cols
        
        # Select only the expected features
        if self.feature_columns is not None:
            available_cols = [col for col in self.feature_columns if col in df.columns]
            df = df[available_cols]
        
        return df
    
    def train_model(self, data_path):
        """
        Train the heart disease prediction model.
        
        Args:
            data_path (str): Path to the heart disease dataset
        """
        print("🔬 Training Heart Disease Prediction Model...")
        
        # Load data
        df = pd.read_csv(data_path)
        print(f"📊 Dataset loaded: {df.shape}")
        
        # Separate features and target
        X = df.drop('Heart Disease Status', axis=1)
        y = df['Heart Disease Status']
        
        # Encode target variable
        target_encoder = LabelEncoder()
        y_encoded = target_encoder.fit_transform(y)
        self.target_encoder = target_encoder
        
        print(f"📈 Target distribution: {np.bincount(y_encoded)}")
        
        # Preprocess features
        X_processed = self.preprocess_data(X, is_training=True)
        print(f"🔧 Features after preprocessing: {X_processed.shape}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train multiple models and select the best
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'SVM': SVC(probability=True, random_state=42)
        }
        
        best_score = 0
        best_model_name = None
        
        print("\n🎯 Training and evaluating models...")
        
        for name, model in models.items():
            # Train model
            if name == 'SVM':
                # Use scaled features for SVM
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                y_proba = model.predict_proba(X_test_scaled)[:, 1]
            else:
                # Use original features for tree-based models
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            
            print(f"\n{name} Results:")
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  F1-Score: {f1:.4f}")
            
            # Store metrics
            self.model_metrics[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'model': model
            }
            
            # Select best model based on F1-score
            if f1 > best_score:
                best_score = f1
                best_model_name = name
                self.model = model
        
        print(f"\n🏆 Best model: {best_model_name} (F1-Score: {best_score:.4f})")
        
        # Final evaluation with best model
        if best_model_name == 'SVM':
            final_pred = self.model.predict(X_test_scaled)
        else:
            final_pred = self.model.predict(X_test)
        
        print(f"\n📋 Final Model Performance:")
        print(f"Confusion Matrix:")
        print(confusion_matrix(y_test, final_pred))
        print(f"\nClassification Report:")
        print(classification_report(y_test, final_pred, 
                                  target_names=self.target_encoder.classes_))
        
        # Save model and preprocessors
        self.save_model()
        
        return self.model, self.model_metrics
    
    def predict(self, input_data):
        """
        Make predictions on new data.
        
        Args:
            input_data (dict or DataFrame): Input features
            
        Returns:
            dict: Prediction results with probability
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train_model() first.")
        
        # Convert input to DataFrame if it's a dict
        if isinstance(input_data, dict):
            df = pd.DataFrame([input_data])
        else:
            df = input_data.copy()
        
        # Preprocess the input
        X_processed = self.preprocess_data(df, is_training=False)
        
        # Scale if using SVM
        model_name = type(self.model).__name__
        if model_name == 'SVC':
            X_scaled = self.scaler.transform(X_processed)
            prediction = self.model.predict(X_scaled)[0]
            probability = self.model.predict_proba(X_scaled)[0]
        else:
            prediction = self.model.predict(X_processed)[0]
            probability = self.model.predict_proba(X_processed)[0]
        
        # Convert back to original labels
        predicted_class = self.target_encoder.inverse_transform([prediction])[0]
        
        # Calculate risk percentage
        risk_probability = probability[1] if len(probability) > 1 else probability[0]
        
        return {
            'prediction': predicted_class,
            'risk_probability': float(risk_probability),
            'risk_percentage': f"{risk_probability * 100:.1f}%",
            'confidence': f"{max(probability) * 100:.1f}%"
        }
    
    def save_model(self, filename='heart_disease_model.pkl'):
        """Save the trained model and preprocessors."""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'target_encoder': self.target_encoder,
            'feature_columns': self.feature_columns,
            'model_metrics': self.model_metrics
        }
        
        joblib.dump(model_data, filename)
        print(f"💾 Model saved as {filename}")
    
    def load_model(self, filename='heart_disease_model.pkl'):
        """Load a pre-trained model."""
        try:
            model_data = joblib.load(filename)
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.label_encoders = model_data['label_encoders']
            self.target_encoder = model_data['target_encoder']
            self.feature_columns = model_data['feature_columns']
            self.model_metrics = model_data.get('model_metrics', {})
            
            print(f"✅ Model loaded from {filename}")
            return True
        except FileNotFoundError:
            print(f"❌ Model file {filename} not found")
            return False
    
    def get_feature_importance(self):
        """Get feature importance from the trained model."""
        if self.model is None:
            return None
        
        # Only tree-based models have feature_importances_
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
            feature_importance = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            return feature_importance
        else:
            return None


def train_heart_disease_model():
    """Train and save the heart disease prediction model."""
    
    # Initialize predictor
    predictor = HeartDiseasePredictor()
    
    # Train model
    model, metrics = predictor.train_model('heart_disease.csv')
    
    # Test with sample prediction
    print("\n🧪 Testing model with sample prediction...")
    
    sample_input = {
        'Age': 55,
        'Gender': 'Male',
        'Blood Pressure': 140,
        'Cholesterol Level': 200,
        'Exercise Habits': 'Medium',
        'Smoking': 'No',
        'Family Heart Disease': 'Yes',
        'Diabetes': 'No',
        'BMI': 28.5,
        'High Blood Pressure': 'Yes',
        'Low HDL Cholesterol': 'No',
        'High LDL Cholesterol': 'Yes',
        'Alcohol Consumption': 'Low',
        'Stress Level': 'Medium',
        'Sleep Hours': 7.0,
        'Sugar Consumption': 'Medium',
        'Triglyceride Level': 150.0,
        'Fasting Blood Sugar': 100.0,
        'CRP Level': 2.5,
        'Homocysteine Level': 10.0
    }
    
    result = predictor.predict(sample_input)
    print(f"Sample Prediction Result: {result}")
    
    # Show feature importance
    feature_importance = predictor.get_feature_importance()
    if feature_importance is not None:
        print(f"\n📊 Top 10 Most Important Features:")
        print(feature_importance.head(10))
    
    return predictor


if __name__ == "__main__":
    predictor = train_heart_disease_model()