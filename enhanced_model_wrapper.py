#!/usr/bin/env python3
"""
Enhanced Model Wrapper with Realistic Risk Assessment
"""

import joblib
import pandas as pd
import numpy as np

class EnhancedHeartDiseasePredictor:
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
    
    def calculate_clinical_risk_score(self, input_data):
        """Calculate clinical risk score based on established risk factors"""
        
        risk_score = 0
        
        # Age risk (major factor)
        age = input_data.get('Age', 50)
        if age >= 75:
            risk_score += 25
        elif age >= 65:
            risk_score += 20
        elif age >= 55:
            risk_score += 15
        elif age >= 45:
            risk_score += 10
        elif age >= 35:
            risk_score += 5
        
        # Gender risk
        if input_data.get('Gender', 'Male') == 'Male':
            risk_score += 8
        
        # Blood Pressure risk
        bp = input_data.get('Blood Pressure', 120)
        if bp >= 180:
            risk_score += 20
        elif bp >= 160:
            risk_score += 15
        elif bp >= 140:
            risk_score += 10
        elif bp >= 130:
            risk_score += 5
        
        # Cholesterol risk
        chol = input_data.get('Cholesterol Level', 200)
        if chol >= 300:
            risk_score += 15
        elif chol >= 250:
            risk_score += 10
        elif chol >= 220:
            risk_score += 5
        
        # BMI risk
        bmi = input_data.get('BMI', 25)
        if bmi >= 35:
            risk_score += 10
        elif bmi >= 30:
            risk_score += 7
        elif bmi >= 27:
            risk_score += 4
        
        # Smoking risk (major factor)
        smoking = input_data.get('Smoking', 'No')
        if smoking == 'Yes':
            risk_score += 15
        elif smoking == 'Former':
            risk_score += 5
        
        # Diabetes risk (major factor)
        if input_data.get('Diabetes', 'No') == 'Yes':
            risk_score += 15
        
        # Family history risk
        if input_data.get('Family Heart Disease', 'No') == 'Yes':
            risk_score += 10
        
        # Exercise habits (protective factor)
        exercise = input_data.get('Exercise Habits', 'Medium')
        if exercise == 'Low':
            risk_score += 8
        elif exercise == 'Medium':
            risk_score += 3
        
        # High blood pressure
        if input_data.get('High Blood Pressure', 'No') == 'Yes':
            risk_score += 10
        
        # Cholesterol levels
        if input_data.get('High LDL Cholesterol', 'No') == 'Yes':
            risk_score += 8
        if input_data.get('Low HDL Cholesterol', 'No') == 'Yes':
            risk_score += 6
        
        # Lifestyle factors
        if input_data.get('Stress Level', 'Medium') == 'High':
            risk_score += 5
        
        if input_data.get('Alcohol Consumption', 'None') == 'Heavy':
            risk_score += 6
        
        if input_data.get('Sugar Consumption', 'Medium') == 'High':
            risk_score += 4
        
        # Sleep (less than 6 or more than 9 hours increases risk)
        sleep_hours = input_data.get('Sleep Hours', 7)
        if sleep_hours < 6 or sleep_hours > 9:
            risk_score += 3
        
        # Advanced biomarkers
        crp = input_data.get('CRP Level', 1.0)
        if crp > 3.0:
            risk_score += 8
        elif crp > 1.0:
            risk_score += 4
        
        homocysteine = input_data.get('Homocysteine Level', 10.0)
        if homocysteine > 15:
            risk_score += 6
        elif homocysteine > 12:
            risk_score += 3
        
        triglycerides = input_data.get('Triglyceride Level', 150)
        if triglycerides > 200:
            risk_score += 5
        elif triglycerides > 150:
            risk_score += 2
        
        fasting_sugar = input_data.get('Fasting Blood Sugar', 90)
        if fasting_sugar > 125:
            risk_score += 8
        elif fasting_sugar > 100:
            risk_score += 4
        
        return min(risk_score, 100)  # Cap at 100
    
    def predict(self, input_data):
        if not self.is_loaded:
            raise ValueError("Model not loaded")
        
        # Calculate clinical risk score
        clinical_score = self.calculate_clinical_risk_score(input_data)
        
        # Try to get ML model prediction as well
        ml_probability = 0.3  # Default fallback
        
        try:
            # Convert input to DataFrame for ML model
            if isinstance(input_data, dict):
                df = pd.DataFrame([input_data])
            else:
                df = input_data.copy()
            
            # Get model components
            model = self.model_data['model']
            label_encoders = self.model_data.get('label_encoders', {})
            
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
            
            # Get ML prediction
            probabilities = model.predict_proba(df)[0]
            ml_probability = probabilities[1]  # Probability of "Yes"
            
        except Exception as e:
            print(f"ML model prediction failed, using clinical score only: {e}")
        
        # Combine clinical score and ML prediction
        # Clinical score: 0-100, convert to 0-1
        clinical_probability = clinical_score / 100.0
        
        # Weighted combination (70% clinical, 30% ML for reliability)
        combined_probability = 0.7 * clinical_probability + 0.3 * ml_probability
        
        # Apply some realistic calibration
        if combined_probability > 0.9:
            combined_probability = 0.9  # Cap at 90%
        elif combined_probability < 0.05:
            combined_probability = 0.05  # Minimum 5%
        
        risk_probability = combined_probability
        
        # Determine risk level and prediction
        if risk_probability >= 0.7:
            prediction = 1
            risk_level = "High"
            color = "red"
            recommendation = "High risk detected. Immediate medical consultation and comprehensive cardiac evaluation recommended. Consider lifestyle changes and possible medication."
        elif risk_probability >= 0.5:
            prediction = 1
            risk_level = "Medium-High"
            color = "orange"
            recommendation = "Moderate-high risk detected. Schedule medical evaluation within 2-4 weeks. Implement lifestyle changes including diet, exercise, and stress management."
        elif risk_probability >= 0.3:
            prediction = 0
            risk_level = "Medium"
            color = "orange"
            recommendation = "Moderate risk detected. Consider lifestyle modifications and regular monitoring. Schedule routine check-up within 3-6 months."
        else:
            prediction = 0
            risk_level = "Low"
            color = "green"
            recommendation = "Low risk detected. Continue healthy lifestyle habits. Routine annual check-ups recommended for prevention."
        
        # Calculate confidence based on consistency of risk factors
        confidence = min(0.95, 0.6 + abs(clinical_probability - ml_probability))
        
        # Convert prediction back to text
        predicted_class = "Yes" if prediction == 1 else "No"
        
        return {
            'prediction': predicted_class,
            'risk_probability': float(risk_probability),
            'risk_percentage': f"{risk_probability * 100:.1f}%",
            'confidence': f"{confidence * 100:.1f}%",
            'risk_level': risk_level,
            'risk_color': color,
            'recommendation': recommendation,
            'clinical_score': clinical_score,
            'ml_probability': float(ml_probability),
            'combined_approach': True
        }
    
    def create_basic_model(self):
        """
        Create a basic rule-based model for production deployment when no pre-trained model exists.
        This ensures the app works in environments like Render where model files might not be available.
        """
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        import pandas as pd
        
        try:
            # Create a simple synthetic dataset for basic functionality
            # This is just to ensure the app works in production
            np.random.seed(42)
            n_samples = 1000
            
            # Generate synthetic data that mimics heart disease patterns
            data = {
                'Age': np.random.normal(55, 15, n_samples).clip(18, 90),
                'Gender': np.random.choice(['Male', 'Female'], n_samples),
                'Blood Pressure': np.random.normal(130, 20, n_samples).clip(80, 200),
                'Cholesterol Level': np.random.normal(200, 50, n_samples).clip(100, 400),
                'BMI': np.random.normal(25, 5, n_samples).clip(15, 50),
                'Exercise Habits': np.random.choice(['Low', 'Medium', 'High'], n_samples),
                'Alcohol Consumption': np.random.choice(['None', 'Light', 'Moderate', 'Heavy'], n_samples),
                'Stress Level': np.random.choice(['Low', 'Medium', 'High'], n_samples),
                'Sleep Hours': np.random.normal(7, 1.5, n_samples).clip(3, 12),
                'Sugar Consumption': np.random.choice(['Low', 'Medium', 'High'], n_samples),
                'Triglyceride Level': np.random.normal(150, 40, n_samples).clip(50, 400),
                'Fasting Blood Sugar': np.random.normal(95, 20, n_samples).clip(60, 200),
                'CRP Level': np.random.exponential(2, n_samples).clip(0.1, 10),
                'Homocysteine Level': np.random.normal(10, 3, n_samples).clip(5, 25)
            }
            
            df = pd.DataFrame(data)
            
            # Create target based on realistic risk factors
            risk_scores = []
            for _, row in df.iterrows():
                score = 0
                # Age risk
                if row['Age'] > 65: score += 0.3
                elif row['Age'] > 55: score += 0.2
                elif row['Age'] > 45: score += 0.1
                
                # Gender risk (males higher risk)
                if row['Gender'] == 'Male': score += 0.1
                
                # Blood pressure risk
                if row['Blood Pressure'] > 140: score += 0.2
                elif row['Blood Pressure'] > 130: score += 0.1
                
                # Cholesterol risk
                if row['Cholesterol Level'] > 240: score += 0.15
                elif row['Cholesterol Level'] > 200: score += 0.1
                
                # BMI risk
                if row['BMI'] > 30: score += 0.1
                elif row['BMI'] > 25: score += 0.05
                
                # Exercise protection
                if row['Exercise Habits'] == 'High': score -= 0.1
                elif row['Exercise Habits'] == 'Low': score += 0.1
                
                risk_scores.append(score)
            
            # Convert to binary classification (threshold at median + noise)
            threshold = np.median(risk_scores) + np.random.normal(0, 0.1, len(risk_scores))
            y = (np.array(risk_scores) > threshold).astype(int)
            
            # Prepare features
            X = df.copy()
            
            # Encode categorical variables
            categorical_cols = ['Gender', 'Exercise Habits', 'Alcohol Consumption', 'Stress Level', 'Sugar Consumption']
            X_encoded = pd.get_dummies(X, columns=categorical_cols)
            
            # Train a basic model
            X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
            
            model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
            model.fit(X_train, y_train)
            
            # Store the model data
            self.model_data = {
                'model': model,
                'feature_names': list(X_encoded.columns),
                'feature_means': X_encoded.mean().to_dict(),
                'original_columns': list(df.columns)
            }
            
            self.is_loaded = True
            return True
            
        except Exception as e:
            print(f"Error creating basic model: {e}")
            return False