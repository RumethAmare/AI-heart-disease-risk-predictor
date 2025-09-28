#!/usr/bin/env python3
"""
Simple Model Wrapper for Flask App
Compatible with the new properly trained model
"""

import joblib
import pandas as pd
import numpy as np

class SimpleHeartDiseasePredictor:
    """Simple wrapper for the properly trained model"""
    
    def __init__(self):
        self.model_data = None
        self.is_loaded = False
    
    def load_model(self, filename):
        """Load the properly trained model"""
        try:
            self.model_data = joblib.load(filename)
            self.is_loaded = True
            return True
        except:
            return False
    
    def predict(self, input_data):
        """Make prediction with proper format"""
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
                    # Handle the value
                    val = str(df[col].iloc[0])
                    if val in encoder.classes_:
                        df[col] = encoder.transform([val])[0]
                    else:
                        df[col] = 0  # Default for unknown values
                except:
                    df[col] = 0
        
        # Fill missing columns with defaults
        expected_features = self.model_data.get('feature_columns', df.columns)
        for col in expected_features:
            if col not in df.columns:
                df[col] = 0  # Default value
        
        # Select only expected features in correct order
        df = df[expected_features]
        
        # Make prediction
        prediction = model.predict(df)[0]
        probabilities = model.predict_proba(df)[0]
        
        # Convert prediction back to text
        if target_encoder:
            predicted_class = target_encoder.inverse_transform([prediction])[0]
        else:
            predicted_class = "Yes" if prediction == 1 else "No"
        
        # Calculate risk probability (probability of "Yes")
        if len(probabilities) > 1:
            risk_probability = probabilities[1] if predicted_class == "No" else probabilities[0]
            if predicted_class == "Yes":
                risk_probability = probabilities[1] if len(probabilities) > 1 else probabilities[0]
        else:
            risk_probability = probabilities[0]
        
        # Ensure risk_probability is for "Yes" class
        if predicted_class == "No":
            risk_probability = probabilities[1] if len(probabilities) > 1 else (1 - probabilities[0])
        
        confidence = max(probabilities)
        
        return {
            'prediction': predicted_class,
            'risk_probability': float(risk_probability),
            'risk_percentage': f"{risk_probability * 100:.1f}%",
            'confidence': f"{confidence * 100:.1f}%"
        }