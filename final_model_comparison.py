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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_data():
    """Load and preprocess the heart disease dataset"""
    print("Loading data...")
    
    # Try different CSV files
    try:
        df = pd.read_csv('heart_disease.csv')
    except:
        try:
            df = pd.read_csv('heart_disease_extended.csv')
        except:
            raise FileNotFoundError("Could not find heart disease CSV file")
    
    print(f"Dataset shape: {df.shape}")
    print(f"Missing values per column:")
    print(df.isnull().sum())
    
    # Handle missing values
    # For numerical columns, use median imputation
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'Heart Disease Status' in numerical_cols:
        numerical_cols.remove('Heart Disease Status')
    
    # For categorical columns, use most frequent imputation
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    print(f"Numerical columns: {numerical_cols}")
    print(f"Categorical columns: {categorical_cols}")
    
    # Impute missing values
    if numerical_cols:
        num_imputer = SimpleImputer(strategy='median')
        df[numerical_cols] = num_imputer.fit_transform(df[numerical_cols])
    
    if categorical_cols:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])
    
    # Select the 14 features used in the trained model
    selected_features = [
        'Age', 'Gender', 'Blood Pressure', 'Cholesterol Level', 'Exercise Habits',
        'Smoking', 'Family Heart Disease', 'Diabetes', 'BMI', 'High Blood Pressure',
        'Low HDL Cholesterol', 'High LDL Cholesterol', 'Alcohol Consumption', 'Stress Level'
    ]
    
    # Check which features exist
    available_features = [col for col in selected_features if col in df.columns]
    print(f"Available features: {available_features}")
    
    if len(available_features) < 10:
        print("Warning: Less than 10 features available. Using all available features.")
        # Use all features except target
        available_features = [col for col in df.columns if col != 'Heart Disease Status']
    
    # Prepare features and target
    X = df[available_features].copy()
    y = df['Heart Disease Status'].copy()
    
    # Encode categorical variables
    label_encoders = {}
    for col in X.columns:
        if X[col].dtype == 'object':
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le
    
    # Encode target variable
    if y.dtype == 'object':
        target_encoder = LabelEncoder()
        y = target_encoder.fit_transform(y.astype(str))
    
    print(f"Final dataset shape: {X.shape}")
    print(f"Class distribution: {pd.Series(y).value_counts().to_dict()}")
    
    return X, y, available_features

def run_model_comparison():
    """Run comparison across multiple models"""
    
    # Load and preprocess data
    X, y, feature_names = load_and_preprocess_data()
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Apply SMOTE for balancing (if needed)
    try:
        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
        print(f"After SMOTE - Training set shape: {X_train_balanced.shape}")
        print(f"After SMOTE - Class distribution: {pd.Series(y_train_balanced).value_counts().to_dict()}")
    except Exception as e:
        print(f"SMOTE failed: {e}")
        print("Using original training data...")
        X_train_balanced, y_train_balanced = X_train_scaled, y_train
    
    # Define models to compare
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
        'Naive Bayes': GaussianNB(),
        'Support Vector Machine': SVC(random_state=42)
    }
    
    results = {}
    
    print("\n" + "="*70)
    print("MODEL COMPARISON RESULTS")
    print("="*70)
    
    for name, model in models.items():
        try:
            print(f"\nTraining {name}...")
            
            # Train the model
            model.fit(X_train_balanced, y_train_balanced)
            
            # Make predictions
            y_pred = model.predict(X_test_scaled)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            
            # Get detailed classification report
            report = classification_report(y_test, y_pred, output_dict=True)
            
            results[name] = {
                'accuracy': accuracy,
                'precision': report['weighted avg']['precision'],
                'recall': report['weighted avg']['recall'],
                'f1_score': report['weighted avg']['f1-score']
            }
            
            print(f"{name} Accuracy: {accuracy:.4f}")
            
        except Exception as e:
            print(f"Error training {name}: {e}")
            results[name] = {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'error': str(e)
            }
    
    return results, feature_names

def display_results(results, feature_names):
    """Display comparison results in a nice format"""
    
    print("\n" + "="*80)
    print("FINAL COMPARISON SUMMARY")
    print("="*80)
    
    # Create a summary table
    print(f"{'Model':<25} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
    print("-" * 75)
    
    # Sort by accuracy
    sorted_results = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
    
    for name, metrics in sorted_results:
        if 'error' in metrics:
            print(f"{name:<25} {'ERROR':<10} {'ERROR':<10} {'ERROR':<10} {'ERROR':<10}")
        else:
            print(f"{name:<25} {metrics['accuracy']:<10.4f} {metrics['precision']:<10.4f} {metrics['recall']:<10.4f} {metrics['f1_score']:<10.4f}")
    
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    
    # Find best model
    valid_results = {k: v for k, v in results.items() if 'error' not in v}
    if valid_results:
        best_model = max(valid_results.items(), key=lambda x: x[1]['accuracy'])
        print(f"Best performing model: {best_model[0]}")
        print(f"Best accuracy: {best_model[1]['accuracy']:.4f}")
        
        # Performance categories
        excellent = [k for k, v in valid_results.items() if v['accuracy'] >= 0.95]
        good = [k for k, v in valid_results.items() if 0.90 <= v['accuracy'] < 0.95]
        moderate = [k for k, v in valid_results.items() if 0.80 <= v['accuracy'] < 0.90]
        
        if excellent:
            print(f"\nExcellent performance (>=95%): {', '.join(excellent)}")
        if good:
            print(f"Good performance (90-94%): {', '.join(good)}")
        if moderate:
            print(f"Moderate performance (80-89%): {', '.join(moderate)}")
    
    print(f"\nDataset used {len(feature_names)} features:")
    for i, feature in enumerate(feature_names, 1):
        print(f"{i:2d}. {feature}")
    
    print("\nNote: Current production model uses Random Forest with enhanced clinical scoring.")

if __name__ == "__main__":
    try:
        print("HEART DISEASE MODEL COMPARISON")
        print("="*50)
        
        results, features = run_model_comparison()
        display_results(results, features)
        
        print("\nComparison completed successfully!")
        
    except Exception as e:
        print(f"Error running comparison: {e}")
        import traceback
        traceback.print_exc()