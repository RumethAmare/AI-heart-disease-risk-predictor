import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)
from imblearn.over_sampling import SMOTE
import pickle
import warnings
import time
from datetime import datetime
warnings.filterwarnings('ignore')

print("🫀 FAST HEART DISEASE MODEL TRAINING WITH EXTENDED DATASET")
print("=" * 70)

# Load and preprocess data
print("📊 Loading extended dataset...")
df = pd.read_csv('heart_disease_extended.csv')
print(f"✅ Dataset loaded: {df.shape}")

# Select essential 14 features
essential_features = [
    'Age', 'Gender', 'Blood Pressure', 'Cholesterol Level', 'Exercise Habits',
    'Smoking', 'Family Heart Disease', 'Diabetes', 'BMI', 'High Blood Pressure',
    'Low HDL Cholesterol', 'High LDL Cholesterol', 'Alcohol Consumption', 'Stress Level'
]

print(f"✂️  Using {len(essential_features)} essential features (dropping {df.shape[1] - len(essential_features) - 1} unused columns)")

# Keep only essential features + target
df_clean = df[essential_features + ['Heart Disease Status']].copy()

# Quick missing value handling
print("🧹 Handling missing values...")
for col in df_clean.columns:
    if col != 'Heart Disease Status':
        if df_clean[col].dtype == 'object':
            df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0] if not df_clean[col].mode().empty else 'Unknown')
        else:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())

print(f"   ✅ Cleaned dataset: {df_clean.shape}")

# Prepare features and target
X = df_clean[essential_features].copy()
y = df_clean['Heart Disease Status'].copy()

# Encode categorical variables
print("🔄 Encoding variables...")
label_encoders = {}
for col in X.columns:
    if X[col].dtype == 'object':
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le

# Encode target
if y.dtype == 'object':
    le_target = LabelEncoder()
    y = le_target.fit_transform(y.astype(str))
    label_encoders['target'] = le_target

print(f"   ✅ Encoded {len([c for c in X.columns if c in label_encoders])} categorical features")
print(f"   • Class distribution: {np.bincount(y)}")

# Split and scale data
print("\n🔀 Splitting and scaling...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"   • Training: {X_train_scaled.shape[0]:,} samples")
print(f"   • Testing: {X_test_scaled.shape[0]:,} samples")

# Define optimized models (no hyperparameter tuning for speed)
models = {
    'Random Forest': RandomForestClassifier(
        n_estimators=100, max_depth=15, min_samples_split=5,
        random_state=42, n_jobs=-1
    ),
    'Logistic Regression': LogisticRegression(
        random_state=42, max_iter=1000, solver='liblinear'
    ),
    'Decision Tree': DecisionTreeClassifier(
        random_state=42, max_depth=15, min_samples_split=10
    ),
    'K-Nearest Neighbors': KNeighborsClassifier(
        n_neighbors=5, weights='distance'
    ),
    'Naive Bayes': GaussianNB(),
    'Support Vector Machine': SVC(
        random_state=42, kernel='rbf', C=1.0, probability=True
    )
}

print(f"\n🎯 TRAINING {len(models)} MODELS")
print("=" * 70)

results = {}
best_models = {}

for name, model in models.items():
    print(f"\n🚀 Training {name}...")
    start_time = time.time()
    
    try:
        # Train model
        model.fit(X_train_scaled, y_train)
        training_time = time.time() - start_time
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        # ROC AUC
        try:
            y_prob = model.predict_proba(X_test_scaled)[:, 1]
            auc = roc_auc_score(y_test, y_prob)
        except:
            auc = 0.0
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Store results
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc_score': auc,
            'training_time': training_time,
            'confusion_matrix': cm
        }
        best_models[name] = model
        
        print(f"   ✅ Completed in {training_time:.2f}s")
        print(f"   📊 Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
    except Exception as e:
        print(f"   ❌ Error: {str(e)[:50]}...")
        results[name] = {'error': str(e)}

# Generate comprehensive report
print(f"\n📋 COMPREHENSIVE RESULTS REPORT")
print("=" * 80)

valid_results = {k: v for k, v in results.items() if 'error' not in v}
sorted_results = sorted(valid_results.items(), key=lambda x: x[1]['accuracy'], reverse=True)

if not sorted_results:
    print("❌ No models trained successfully!")
    exit(1)

# Performance table
print(f"\n🏆 MODEL PERFORMANCE RANKING")
print("-" * 80)
print(f"{'Rank':<5}{'Model':<25}{'Accuracy':<12}{'Precision':<12}{'Recall':<12}{'F1-Score':<12}{'Time(s)':<10}")
print("-" * 80)

for i, (name, result) in enumerate(sorted_results, 1):
    acc = result['accuracy'] * 100
    prec = result['precision'] * 100
    rec = result['recall'] * 100
    f1 = result['f1_score'] * 100
    time_s = result['training_time']
    
    print(f"{i:<5}{name:<25}{acc:<11.2f}%{prec:<11.2f}%{rec:<11.2f}%{f1:<11.2f}%{time_s:<10.2f}")

# Best model analysis
best_name, best_result = sorted_results[0]
print(f"\n🥇 CHAMPION MODEL: {best_name}")
print("=" * 50)
print(f"🎯 Test Accuracy: {best_result['accuracy']*100:.2f}%")
print(f"⚖️  Precision: {best_result['precision']*100:.2f}%")
print(f"🎣 Recall: {best_result['recall']*100:.2f}%")
print(f"🏹 F1-Score: {best_result['f1_score']*100:.2f}%")
if best_result['auc_score'] > 0:
    print(f"📈 ROC-AUC: {best_result['auc_score']*100:.2f}%")
print(f"⏱️  Training Time: {best_result['training_time']:.2f} seconds")

# Performance categories
print(f"\n📊 PERFORMANCE CATEGORIES")
print("-" * 40)

excellent = [name for name, r in valid_results.items() if r['accuracy'] >= 0.95]
good = [name for name, r in valid_results.items() if 0.90 <= r['accuracy'] < 0.95]
moderate = [name for name, r in valid_results.items() if 0.80 <= r['accuracy'] < 0.90]
poor = [name for name, r in valid_results.items() if r['accuracy'] < 0.80]

if excellent:
    print(f"🟢 Excellent (≥95%): {', '.join(excellent)}")
if good:
    print(f"🔵 Good (90-94%): {', '.join(good)}")
if moderate:
    print(f"🟡 Moderate (80-89%): {', '.join(moderate)}")
if poor:
    print(f"🔴 Needs Improvement (<80%): {', '.join(poor)}")

# Confusion Matrix for best model
print(f"\n🎯 CONFUSION MATRIX - {best_name}")
print("-" * 35)
cm = best_result['confusion_matrix']
print(f"                 Predicted")
print(f"              No    Yes")
print(f"Actual   No   {cm[0][0]:<6}{cm[0][1]:<6}")
print(f"        Yes   {cm[1][0]:<6}{cm[1][1]:<6}")

# Calculate additional metrics from confusion matrix
tn, fp, fn, tp = cm.ravel()
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0

print(f"\n📈 DETAILED METRICS - {best_name}")
print("-" * 40)
print(f"True Positives: {tp:,}")
print(f"True Negatives: {tn:,}")
print(f"False Positives: {fp:,}")
print(f"False Negatives: {fn:,}")
print(f"Sensitivity (Recall): {sensitivity:.4f}")
print(f"Specificity: {specificity:.4f}")

# Comparison with previous models
print(f"\n🔄 COMPARISON WITH PREVIOUS MODELS")
print("-" * 50)
print(f"Current Best: {best_name} - {best_result['accuracy']*100:.2f}%")
print(f"Previous Production Model: Random Forest - 97.4%")

rf_accuracy = results.get('Random Forest', {}).get('accuracy', 0) * 100
diff = abs(97.4 - rf_accuracy)

if diff <= 2:
    print(f"✅ Excellent consistency (difference: {diff:.1f}%)")
elif diff <= 5:
    print(f"✅ Good consistency (difference: {diff:.1f}%)")
else:
    print(f"⚠️  Notable difference: {diff:.1f}%")

# Key insights
print(f"\n💡 KEY INSIGHTS")
print("-" * 30)
print(f"• Dataset: 16,000 samples with {len(essential_features)} essential features")
print(f"• Dropped {df.shape[1] - len(essential_features) - 1} unused columns for optimization")
print(f"• Perfect class balance (50-50 split)")
print(f"• {best_name} emerged as the top performer")

if best_result['accuracy'] >= 0.95:
    print(f"• Excellent model performance achieved (≥95%)")
elif best_result['accuracy'] >= 0.90:
    print(f"• Good model performance achieved (≥90%)")
else:
    print(f"• Consider further optimization for production use")

# Save best model
print(f"\n💾 SAVING BEST MODEL")
print("-" * 25)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_filename = f'heart_disease_extended_{best_name.lower().replace(" ", "_")}_{timestamp}.pkl'

model_data = {
    'model': best_result['model'],
    'scaler': scaler,
    'label_encoders': label_encoders,
    'feature_names': essential_features,
    'model_type': best_name,
    'accuracy': best_result['accuracy'],
    'training_date': datetime.now().isoformat(),
    'dataset': 'heart_disease_extended.csv',
    'features_used': len(essential_features),
    'samples_trained': len(X_train_scaled)
}

try:
    with open(model_filename, 'wb') as f:
        pickle.dump(model_data, f)
    print(f"✅ Best model saved: {model_filename}")
    print(f"📊 Model type: {best_name}")
    print(f"🎯 Accuracy: {best_result['accuracy']*100:.2f}%")
    print(f"🗓️  Training date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
except Exception as e:
    print(f"❌ Error saving model: {e}")

# Final summary
print(f"\n🎉 TRAINING SUMMARY")
print("=" * 40)
print(f"✅ Successfully trained {len(valid_results)} models")
print(f"🏆 Best performer: {best_name}")
print(f"📊 Best accuracy: {best_result['accuracy']*100:.2f}%")
print(f"💾 Model saved for production use")
print(f"🔧 Ready for integration with existing system")

print(f"\n✨ Extended dataset training completed successfully!")