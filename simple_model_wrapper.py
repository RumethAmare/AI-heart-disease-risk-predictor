#!/usr/bin/env python3
"""
Intelligent Model Wrapper - Smart Risk Assessment
"""

import joblib
import pandas as pd
import numpy as np

class SimpleHeartDiseasePredictor:
    def __init__(self):
        self.model_data = None
        self.is_loaded = False
    
    def load_model(self, filename):
        try:
            self.model_data = joblib.load(filename)
            self.is_loaded = True
            return True
        except:
            return False
    
    def predict(self, input_data):
        if not self.is_loaded:
            raise ValueError("Model not loaded")
        
        # Convert input to DataFrame
        if isinstance(input_data, dict):
            df = pd.DataFrame([input_data])
        else:
            df = input_data.copy()
        
        # Get model components
        model = self.model_data['model']
        label_encoders = self.model_data.get('label_encoders', {})
        target_encoder = self.model_data.get('target_encoder')
        
        # Encode categorical variables
        for col, encoder in label_encoders.items():
            if col in df.columns:
                try:
                    val = str(df[col].iloc[0])
                    if val in encoder.classes_:
                        df[col] = encoder.transform([val])[0]
                    else:
                        df[col] = 0
                except:
                    df[col] = 0
        
        # Fill missing columns with defaults
        expected_features = self.model_data.get('feature_columns', df.columns)
        for col in expected_features:
            if col not in df.columns:
                df[col] = 0
        
        # Select only expected features in correct order
        df = df[expected_features]
        
        # Make prediction
        probabilities = model.predict_proba(df)[0]
        risk_probability = 1.3*probabilities[1]  # Probability of "Yes"
        
        # SMART THRESHOLDING: Dynamic based on risk level
        if risk_probability >= 0.65:
            prediction = 1  # High confidence "Yes"
            risk_level = "High"
            color = "red"
        elif risk_probability >= 0.45:
            prediction = 1  # Medium-high risk "Yes"  
            risk_level = "Medium-High"
            color = "orange"
        elif risk_probability >= 0.35:
            prediction = 0  # Medium risk "No" but warn
            risk_level = "Medium"
            color = "orange"
        else:
            prediction = 0  # Low risk "No"
            risk_level = "Low"
            color = "green"
        
        # Convert prediction back to text
        if target_encoder:
            predicted_class = target_encoder.inverse_transform([prediction])[0]
        else:
            predicted_class = "Yes" if prediction == 1 else "No"
        
        # confidence = max(1.3*probabilities)
        
        # Generate appropriate recommendation
        if risk_probability >= 0.65:
            recommendation = "High risk detected. Immediate medical consultation recommended."
            confidence = risk_probability     
        elif risk_probability >= 0.45:
            recommendation = "Moderate-high risk. Consider lifestyle changes and medical evaluation."
            confidence = 1 - risk_probability
        elif risk_probability >= 0.35:
            recommendation = "Moderate risk. Maintain healthy lifestyle and monitor regularly."
            confidence = 1 - risk_probability
        else:
            recommendation = "Low risk. Continue healthy lifestyle habits."
            confidence = 1 - risk_probability
        
        return {
            'prediction': predicted_class,
            'risk_probability': float(risk_probability),
            'risk_percentage': f"{risk_probability * 100:.1f}%",
            'confidence': f"{confidence * 100:.1f}%",
            'risk_level': risk_level,
            'risk_color': color,
            'recommendation': recommendation
        }
