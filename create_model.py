import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
from datetime import datetime

print("Creating Render-optimized model...")

# Simple synthetic data generation
np.random.seed(42)
n = 1500

data = {
    'Age': np.random.randint(25, 80, n),
    'Gender': np.random.choice(['Male', 'Female'], n),
    'Blood Pressure': np.random.randint(90, 200, n), 
    'Cholesterol Level': np.random.randint(150, 350, n),
    'BMI': np.random.uniform(18, 40, n).round(1),
    'Exercise Habits': np.random.choice(['Low', 'Medium', 'High'], n),
    'Alcohol Consumption': np.random.choice(['None', 'Light', 'Moderate'], n),
    'Stress Level': np.random.choice(['Low', 'Medium', 'High'], n),
    'Sleep Hours': np.random.uniform(5, 10, n).round(1),
    'Sugar Consumption': np.random.choice(['Low', 'Medium', 'High'], n),
    'Triglyceride Level': np.random.randint(50, 300, n),
    'Fasting Blood Sugar': np.random.randint(70, 150, n),
    'CRP Level': np.random.uniform(0.5, 8, n).round(2),
    'Homocysteine Level': np.random.uniform(5, 20, n).round(1)
}

df = pd.DataFrame(data)

# Create realistic target based on medical risk factors
risk = (df['Age'] > 55).astype(int) * 0.3 + \
       (df['Gender'] == 'Male').astype(int) * 0.15 + \
       (df['Blood Pressure'] > 140).astype(int) * 0.25 + \
       (df['Cholesterol Level'] > 240).astype(int) * 0.2 + \
       (df['BMI'] > 30).astype(int) * 0.1

df['Heart Disease'] = (risk + np.random.normal(0, 0.1, n) > 0.4).astype(int)

print(f"Dataset: {len(df)} samples")
print(f"Target distribution: {df['Heart Disease'].value_counts().to_dict()}")

# Encode categorical variables
X = df.drop('Heart Disease', axis=1)
y = df['Heart Disease']

categorical_cols = ['Gender', 'Exercise Habits', 'Alcohol Consumption', 'Stress Level', 'Sugar Consumption']
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

# Train optimized RandomForest model
print("Training RandomForest model...")
model = RandomForestClassifier(
    n_estimators=100, 
    max_depth=12, 
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42, 
    n_jobs=-1
)

model.fit(X, y)
accuracy = model.score(X, y)
print(f"Model accuracy: {accuracy:.3f}")

# Create comprehensive model package for enhanced_model_wrapper compatibility
model_package = {
    'model': model,
    'label_encoders': label_encoders,
    'feature_columns': list(X.columns),
    'feature_names': list(X.columns),  # For compatibility
    'original_columns': list(X.columns),  # For compatibility
    'feature_means': X.mean().to_dict(),  # For compatibility
    'model_info': {
        'model_type': 'RandomForest',
        'training_date': datetime.now().isoformat(),
        'n_features': len(X.columns),
        'optimized_for': 'render_deployment',
        'accuracy': accuracy,
        'version': '2.0'
    }
}

# Save the optimized model
filename = 'heart_disease_render_optimized.pkl'
joblib.dump(model_package, filename, compress=3)

import os
file_size = os.path.getsize(filename) / (1024*1024)
print(f"Model saved: {filename} ({file_size:.1f} MB)")
print("Render-optimized model ready for deployment!")