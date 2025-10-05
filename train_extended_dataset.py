import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
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

class HeartDiseaseTrainer:
    """Comprehensive Heart Disease Model Training and Comparison"""
    
    def __init__(self):
        # The 14 essential features used in production
        self.selected_features = [
            'Age', 'Gender', 'Blood Pressure', 'Cholesterol Level', 'Exercise Habits',
            'Smoking', 'Family Heart Disease', 'Diabetes', 'BMI', 'High Blood Pressure',
            'Low HDL Cholesterol', 'High LDL Cholesterol', 'Alcohol Consumption', 'Stress Level'
        ]
        
        # Model configurations
        self.models = {
            'Random Forest': {
                'model': RandomForestClassifier(random_state=42),
                'params': {
                    'n_estimators': [100, 200],
                    'max_depth': [10, 15, None],
                    'min_samples_split': [2, 5, 10]
                }
            },
            'Logistic Regression': {
                'model': LogisticRegression(random_state=42, max_iter=2000),
                'params': {
                    'C': [0.1, 1, 10],
                    'solver': ['liblinear', 'lbfgs']
                }
            },
            'Decision Tree': {
                'model': DecisionTreeClassifier(random_state=42),
                'params': {
                    'max_depth': [5, 10, 15, 20],
                    'min_samples_split': [2, 5, 10, 20]
                }
            },
            'K-Nearest Neighbors': {
                'model': KNeighborsClassifier(),
                'params': {
                    'n_neighbors': [3, 5, 7, 9],
                    'weights': ['uniform', 'distance']
                }
            },
            'Naive Bayes': {
                'model': GaussianNB(),
                'params': {}  # No hyperparameters to tune
            },
            'Support Vector Machine': {
                'model': SVC(random_state=42, probability=True),
                'params': {
                    'C': [0.1, 1, 10],
                    'kernel': ['rbf', 'linear']
                }
            }
        }
        
        self.results = {}
        self.best_models = {}
        self.X_train_scaled = None
        self.X_test_scaled = None
        self.y_train = None
        self.y_test = None
        self.scaler = None
        self.label_encoders = {}
        
    def load_and_preprocess_data(self):
        """Load and preprocess the extended heart disease dataset"""
        print("🫀 HEART DISEASE MODEL TRAINING WITH EXTENDED DATASET")
        print("=" * 60)
        print("📊 Loading and preprocessing data...")
        
        # Load the extended dataset
        df = pd.read_csv('heart_disease_extended.csv')
        print(f"✅ Loaded dataset shape: {df.shape}")
        
        # Display missing values info
        missing_values = df.isnull().sum()
        print(f"\n📋 Missing values summary:")
        for col, missing in missing_values[missing_values > 0].items():
            print(f"   • {col}: {missing} ({missing/len(df)*100:.1f}%)")
        
        # Keep only the essential 14 features + target
        available_features = [col for col in self.selected_features if col in df.columns]
        missing_features = [col for col in self.selected_features if col not in df.columns]
        
        if missing_features:
            print(f"\n⚠️  Missing expected features: {missing_features}")
        
        print(f"\n✂️  Using {len(available_features)} essential features:")
        for i, feature in enumerate(available_features, 1):
            print(f"   {i:2d}. {feature}")
        
        # Select features and target
        feature_cols = available_features + ['Heart Disease Status']
        df_clean = df[feature_cols].copy()
        
        print(f"\n🧹 Cleaning data...")
        
        # Handle missing values
        # Numerical columns: median imputation
        numerical_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
        if 'Heart Disease Status' in numerical_cols:
            numerical_cols.remove('Heart Disease Status')
        
        # Categorical columns: most frequent imputation
        categorical_cols = df_clean.select_dtypes(include=['object']).columns.tolist()
        if 'Heart Disease Status' in categorical_cols:
            categorical_cols.remove('Heart Disease Status')
        
        # Apply imputation
        if numerical_cols:
            num_imputer = SimpleImputer(strategy='median')
            df_clean[numerical_cols] = num_imputer.fit_transform(df_clean[numerical_cols])
        
        if categorical_cols:
            cat_imputer = SimpleImputer(strategy='most_frequent')
            df_clean[categorical_cols] = cat_imputer.fit_transform(df_clean[categorical_cols])
        
        print(f"   ✅ Imputed missing values")
        print(f"   • Numerical columns: {len(numerical_cols)}")
        print(f"   • Categorical columns: {len(categorical_cols)}")
        
        # Prepare features and target
        X = df_clean[available_features].copy()
        y = df_clean['Heart Disease Status'].copy()
        
        # Encode categorical variables
        print(f"\n🔄 Encoding categorical variables...")
        for col in X.columns:
            if X[col].dtype == 'object':
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                self.label_encoders[col] = le
                print(f"   • Encoded {col}")
        
        # Encode target variable
        if y.dtype == 'object':
            le_target = LabelEncoder()
            y = le_target.fit_transform(y.astype(str))
            self.label_encoders['target'] = le_target
            print(f"   • Encoded target variable")
        
        print(f"\n📈 Final dataset info:")
        print(f"   • Features shape: {X.shape}")
        print(f"   • Target distribution: {np.bincount(y)}")
        print(f"   • Class balance: {np.bincount(y)[1] / len(y) * 100:.1f}% positive cases")
        
        return X, y, available_features
    
    def split_and_scale_data(self, X, y):
        """Split data and apply feature scaling"""
        print(f"\n🔀 Splitting and scaling data...")
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Apply SMOTE for balancing (optional)
        try:
            smote = SMOTE(random_state=42)
            X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
            print(f"   ✅ Applied SMOTE balancing")
            print(f"   • Original training: {X_train_scaled.shape[0]} samples")
            print(f"   • Balanced training: {X_train_balanced.shape[0]} samples")
            X_train_scaled = X_train_balanced
            y_train = y_train_balanced
        except Exception as e:
            print(f"   ⚠️  SMOTE failed: {e}")
            print(f"   • Using original training data")
        
        print(f"   • Training samples: {X_train_scaled.shape[0]:,}")
        print(f"   • Testing samples: {X_test_scaled.shape[0]:,}")
        print(f"   • Features: {X_train_scaled.shape[1]}")
        
        # Store for later use
        self.X_train_scaled = X_train_scaled
        self.X_test_scaled = X_test_scaled
        self.y_train = y_train
        self.y_test = y_test
        self.scaler = scaler
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_model(self, name, model_config, X_train, y_train):
        """Train a single model with hyperparameter optimization"""
        print(f"\n🚀 Training {name}...")
        start_time = time.time()
        
        model = model_config['model']
        params = model_config['params']
        
        if params:  # If there are parameters to tune
            print(f"   🔧 Optimizing hyperparameters...")
            grid_search = GridSearchCV(
                model, params, cv=3, scoring='accuracy', 
                n_jobs=-1, verbose=0
            )
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            cv_score = grid_search.best_score_
            
            print(f"   ✅ Best CV score: {cv_score:.4f}")
            print(f"   🎯 Best parameters: {best_params}")
        else:
            # No hyperparameters to tune (e.g., Naive Bayes)
            best_model = model
            best_model.fit(X_train, y_train)
            cv_scores = cross_val_score(best_model, X_train, y_train, cv=3)
            cv_score = cv_scores.mean()
            best_params = "Default parameters"
            
            print(f"   ✅ CV score: {cv_score:.4f}")
        
        training_time = time.time() - start_time
        print(f"   ⏱️  Training time: {training_time:.2f} seconds")
        
        return best_model, best_params, cv_score, training_time
    
    def evaluate_model(self, name, model, X_test, y_test):
        """Evaluate a trained model on test data"""
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        # ROC AUC (if model supports probability prediction)
        try:
            y_prob = model.predict_proba(X_test)[:, 1]
            auc_score = roc_auc_score(y_test, y_prob)
        except:
            auc_score = 0.0
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc_score': auc_score,
            'confusion_matrix': cm,
            'predictions': y_pred
        }
    
    def train_all_models(self):
        """Train and evaluate all models"""
        print(f"\n🎯 TRAINING ALL MODELS")
        print("=" * 60)
        
        for name, model_config in self.models.items():
            try:
                # Train model
                best_model, best_params, cv_score, training_time = self.train_model(
                    name, model_config, self.X_train_scaled, self.y_train
                )
                
                # Evaluate model
                test_results = self.evaluate_model(
                    name, best_model, self.X_test_scaled, self.y_test
                )
                
                # Store results
                self.results[name] = {
                    'model': best_model,
                    'best_params': best_params,
                    'cv_score': cv_score,
                    'training_time': training_time,
                    **test_results
                }
                self.best_models[name] = best_model
                
                print(f"   📊 Test Accuracy: {test_results['accuracy']:.4f}")
                
            except Exception as e:
                print(f"   ❌ Error training {name}: {str(e)}")
                self.results[name] = {'error': str(e)}
    
    def generate_report(self):
        """Generate comprehensive comparison report"""
        print(f"\n📋 COMPREHENSIVE MODEL COMPARISON REPORT")
        print("=" * 80)
        
        # Filter successful results
        valid_results = {k: v for k, v in self.results.items() if 'error' not in v}
        
        if not valid_results:
            print("❌ No models trained successfully!")
            return
        
        # Sort by accuracy
        sorted_results = sorted(valid_results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
        
        # Performance table
        print(f"\n🏆 PERFORMANCE COMPARISON")
        print("-" * 80)
        print(f"{'Rank':<4} {'Model':<25} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'AUC':<10}")
        print("-" * 80)
        
        for i, (name, results) in enumerate(sorted_results, 1):
            acc = results['accuracy'] * 100
            prec = results['precision'] * 100
            rec = results['recall'] * 100
            f1 = results['f1_score'] * 100
            auc = results['auc_score'] * 100 if results['auc_score'] > 0 else 0
            
            print(f"{i:<4} {name:<25} {acc:<9.2f}% {prec:<9.2f}% {rec:<9.2f}% {f1:<9.2f}% {auc:<9.2f}%")
        
        # Best model details
        best_name, best_results = sorted_results[0]
        print(f"\n🥇 CHAMPION MODEL: {best_name}")
        print("-" * 50)
        print(f"🎯 Accuracy: {best_results['accuracy']*100:.2f}%")
        print(f"⚖️  Precision: {best_results['precision']*100:.2f}%")
        print(f"🎣 Recall: {best_results['recall']*100:.2f}%")
        print(f"🏹 F1-Score: {best_results['f1_score']*100:.2f}%")
        if best_results['auc_score'] > 0:
            print(f"📈 AUC Score: {best_results['auc_score']*100:.2f}%")
        print(f"⏱️  Training Time: {best_results['training_time']:.2f} seconds")
        print(f"🔧 Best Parameters: {best_results['best_params']}")
        
        # Performance categories
        print(f"\n📊 PERFORMANCE CATEGORIES")
        print("-" * 50)
        
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
        
        # Detailed analysis
        print(f"\n💡 KEY INSIGHTS")
        print("-" * 50)
        
        # Training time analysis
        fastest_model = min(valid_results.items(), key=lambda x: x[1]['training_time'])
        print(f"⚡ Fastest Training: {fastest_model[0]} ({fastest_model[1]['training_time']:.2f}s)")
        
        # Accuracy spread
        accuracies = [r['accuracy'] for r in valid_results.values()]
        acc_range = (max(accuracies) - min(accuracies)) * 100
        print(f"📏 Performance Range: {acc_range:.1f}% (Max: {max(accuracies)*100:.1f}%, Min: {min(accuracies)*100:.1f}%)")
        
        # Confusion matrix for best model
        print(f"\n🎯 CONFUSION MATRIX - {best_name}")
        print("-" * 30)
        cm = best_results['confusion_matrix']
        print(f"True Negative: {cm[0][0]:,}  |  False Positive: {cm[0][1]:,}")
        print(f"False Negative: {cm[1][0]:,} |  True Positive: {cm[1][1]:,}")
        
        # Recommendations
        print(f"\n🚀 RECOMMENDATIONS")
        print("-" * 50)
        
        if best_results['accuracy'] >= 0.95:
            print(f"✅ {best_name} achieves excellent performance - recommended for production")
        elif best_results['accuracy'] >= 0.90:
            print(f"✅ {best_name} shows good performance - suitable for production with monitoring")
        else:
            print(f"⚠️  Consider further feature engineering or data collection")
        
        print(f"💾 Best model ready for deployment")
        
        return best_name, best_results
    
    def save_best_model(self, model_name, model_results):
        """Save the best performing model"""
        print(f"\n💾 SAVING BEST MODEL")
        print("-" * 30)
        
        best_model = model_results['model']
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save model
        model_filename = f'heart_disease_model_extended_{model_name.lower().replace(" ", "_")}_{timestamp}.pkl'
        
        model_data = {
            'model': best_model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_names': self.selected_features,
            'model_type': model_name,
            'accuracy': model_results['accuracy'],
            'training_date': datetime.now().isoformat(),
            'dataset': 'heart_disease_extended.csv'
        }
        
        with open(model_filename, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"✅ Model saved as: {model_filename}")
        print(f"📊 Model accuracy: {model_results['accuracy']*100:.2f}%")
        print(f"🗓️  Training date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return model_filename

def main():
    """Main training pipeline"""
    trainer = HeartDiseaseTrainer()
    
    # Load and preprocess data
    X, y, features = trainer.load_and_preprocess_data()
    
    # Split and scale data
    trainer.split_and_scale_data(X, y)
    
    # Train all models
    trainer.train_all_models()
    
    # Generate comprehensive report
    best_model_name, best_results = trainer.generate_report()
    
    # Save best model
    if best_model_name and best_results:
        model_file = trainer.save_best_model(best_model_name, best_results)
        print(f"\n🎉 Training completed successfully!")
        print(f"🏆 Best model: {best_model_name}")
        print(f"💾 Saved as: {model_file}")
    else:
        print(f"\n❌ Training failed - no successful models")

if __name__ == "__main__":
    main()