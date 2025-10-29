#!/usr/bin/env python3
"""
FDM Mini Project 2025 - Specific Requirements Implementation
Data preprocessing tailored for the project requirements:
- Datasets with 10,000+ rows
- Recent data preprocessing
- Support for Data Mining/ML techniques
- Real-world problem solving
"""

from data_preprocessor import DataPreprocessor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

class FDMProjectPreprocessor(DataPreprocessor):
    """
    Extended preprocessor class specifically for FDM Mini Project requirements.
    """
    
    def __init__(self):
        super().__init__()
        self.project_requirements_met = {
            'min_rows': False,
            'recent_data': False,
            'preprocessing_applied': False,
            'ml_ready': False
        }
    
    def validate_project_requirements(self):
        """
        Validate that the dataset meets FDM project requirements.
        """
        print("\n" + "="*60)
        print("FDM PROJECT REQUIREMENTS VALIDATION")
        print("="*60)
        
        # Check minimum rows requirement (10,000+)
        if len(self.data) >= 10000:
            self.project_requirements_met['min_rows'] = True
            print(f"✓ Minimum rows requirement: {len(self.data):,} rows (≥10,000)")
        else:
            print(f"❌ Minimum rows requirement: {len(self.data):,} rows (<10,000)")
            print("⚠️  Consider finding additional data or using data augmentation techniques")
        
        # Check for recent data (look for date columns)
        date_columns = self._identify_date_columns()
        if date_columns:
            latest_dates = {}
            for col in date_columns:
                try:
                    latest_date = pd.to_datetime(self.data[col]).max()
                    latest_dates[col] = latest_date
                    
                    # Check if data is from last 3 years (considered "recent")
                    if latest_date >= datetime.now() - timedelta(days=3*365):
                        self.project_requirements_met['recent_data'] = True
                        print(f"✓ Recent data found: {col} (latest: {latest_date.strftime('%Y-%m-%d')})")
                    else:
                        print(f"⚠️  Data might be outdated: {col} (latest: {latest_date.strftime('%Y-%m-%d')})")
                except:
                    print(f"⚠️  Could not parse dates in column: {col}")
        else:
            print("⚠️  No date columns identified. Cannot verify data recency.")
        
        # Check data preprocessing potential
        preprocessing_opportunities = self._identify_preprocessing_opportunities()
        if preprocessing_opportunities['total_opportunities'] > 0:
            self.project_requirements_met['preprocessing_applied'] = True
            print(f"✓ Preprocessing opportunities identified: {preprocessing_opportunities['total_opportunities']}")
        else:
            print("❌ Limited preprocessing opportunities found")
        
        # Check ML readiness
        ml_readiness = self._assess_ml_readiness()
        if ml_readiness['score'] >= 0.7:
            self.project_requirements_met['ml_ready'] = True
            print(f"✓ Dataset is ML-ready (score: {ml_readiness['score']:.2f})")
        else:
            print(f"⚠️  Dataset needs more preprocessing for ML (score: {ml_readiness['score']:.2f})")
        
        return self.project_requirements_met
    
    def _identify_date_columns(self):
        """Identify potential date columns in the dataset."""
        date_columns = []
        
        for col in self.data.columns:
            # Check by column name
            if any(keyword in col.lower() for keyword in ['date', 'time', 'year', 'month', 'day']):
                date_columns.append(col)
            # Check by data type
            elif self.data[col].dtype == 'datetime64[ns]':
                date_columns.append(col)
            # Check by sample values (try to parse as dates)
            else:
                try:
                    sample_values = self.data[col].dropna().head(10)
                    if len(sample_values) > 0:
                        pd.to_datetime(sample_values, errors='raise')
                        date_columns.append(col)
                except:
                    continue
        
        return list(set(date_columns))
    
    def _identify_preprocessing_opportunities(self):
        """Identify preprocessing opportunities in the dataset."""
        opportunities = {
            'missing_values': 0,
            'categorical_encoding': 0,
            'numeric_scaling': 0,
            'outliers': 0,
            'feature_engineering': 0,
            'total_opportunities': 0
        }
        
        # Missing values
        missing_count = self.data.isnull().sum().sum()
        if missing_count > 0:
            opportunities['missing_values'] = missing_count
        
        # Categorical columns needing encoding
        categorical_count = len(self.categorical_columns)
        if categorical_count > 0:
            opportunities['categorical_encoding'] = categorical_count
        
        # Numeric columns needing scaling
        numeric_count = len(self.numeric_columns)
        if numeric_count > 0:
            # Check if scaling is needed (different ranges)
            ranges = {}
            for col in self.numeric_columns:
                col_range = self.data[col].max() - self.data[col].min()
                ranges[col] = col_range
            
            if len(ranges) > 1 and max(ranges.values()) / min(ranges.values()) > 10:
                opportunities['numeric_scaling'] = numeric_count
        
        # Potential outliers
        outlier_columns = 0
        for col in self.numeric_columns:
            Q1 = self.data[col].quantile(0.25)
            Q3 = self.data[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = self.data[(self.data[col] < Q1 - 1.5*IQR) | (self.data[col] > Q3 + 1.5*IQR)]
            if len(outliers) > 0:
                outlier_columns += 1
        opportunities['outliers'] = outlier_columns
        
        # Feature engineering opportunities
        date_columns = self._identify_date_columns()
        text_columns = [col for col in self.categorical_columns 
                       if self.data[col].astype(str).str.len().mean() > 20]
        opportunities['feature_engineering'] = len(date_columns) + len(text_columns)
        
        opportunities['total_opportunities'] = sum([
            1 if opportunities['missing_values'] > 0 else 0,
            1 if opportunities['categorical_encoding'] > 0 else 0,
            1 if opportunities['numeric_scaling'] > 0 else 0,
            1 if opportunities['outliers'] > 0 else 0,
            1 if opportunities['feature_engineering'] > 0 else 0
        ])
        
        return opportunities
    
    def _assess_ml_readiness(self):
        """Assess how ready the dataset is for machine learning."""
        score = 0.0
        issues = []
        
        # Check for missing values (penalty)
        missing_ratio = self.data.isnull().sum().sum() / (self.data.shape[0] * self.data.shape[1])
        if missing_ratio == 0:
            score += 0.2
        elif missing_ratio < 0.05:
            score += 0.1
        else:
            issues.append(f"High missing values: {missing_ratio:.1%}")
        
        # Check for non-numeric data that needs encoding
        if len(self.categorical_columns) == 0:
            score += 0.2
        else:
            # Check if categorical data has reasonable cardinality
            high_cardinality = [col for col in self.categorical_columns 
                              if self.data[col].nunique() > 50]
            if not high_cardinality:
                score += 0.2
            else:
                issues.append(f"High cardinality categorical columns: {len(high_cardinality)}")
        
        # Check data size
        if len(self.data) >= 10000:
            score += 0.2
        elif len(self.data) >= 1000:
            score += 0.1
        else:
            issues.append(f"Small dataset: {len(self.data)} rows")
        
        # Check feature diversity
        if len(self.numeric_columns) >= 3 and len(self.categorical_columns) >= 1:
            score += 0.2
        elif len(self.numeric_columns) + len(self.categorical_columns) >= 3:
            score += 0.1
        else:
            issues.append("Limited feature diversity")
        
        # Check for potential target variable
        if self.target_column and self.target_column in self.data.columns:
            score += 0.2
        else:
            issues.append("No target variable specified")
        
        return {'score': score, 'issues': issues}
    
    def create_feature_engineering_suggestions(self):
        """Generate feature engineering suggestions for the dataset."""
        suggestions = []
        
        # Date feature engineering
        date_columns = self._identify_date_columns()
        for col in date_columns:
            suggestions.extend([
                f"Extract year, month, day of week from '{col}'",
                f"Create time-based features from '{col}' (e.g., is_weekend, season)",
                f"Calculate time differences using '{col}' as reference"
            ])
        
        # Text feature engineering
        text_columns = [col for col in self.categorical_columns 
                       if self.data[col].astype(str).str.len().mean() > 20]
        for col in text_columns:
            suggestions.extend([
                f"Extract text length from '{col}'",
                f"Create word count features from '{col}'",
                f"Apply TF-IDF or other text vectorization to '{col}'"
            ])
        
        # Numeric feature engineering
        if len(self.numeric_columns) > 1:
            suggestions.extend([
                "Create polynomial features from numeric columns",
                "Calculate ratios between related numeric features",
                "Create binned/discretized versions of continuous features"
            ])
        
        # Interaction features
        if len(self.data.columns) >= 5:
            suggestions.append("Create interaction features between key variables")
        
        return suggestions
    
    def generate_fdm_project_report(self):
        """Generate a comprehensive report for FDM project submission."""
        print("\n" + "="*80)
        print("FDM MINI PROJECT 2025 - DATA PREPROCESSING REPORT")
        print("="*80)
        
        # Basic dataset information
        print("1. DATASET OVERVIEW")
        print("-" * 40)
        print(f"Dataset Name: {getattr(self, 'dataset_name', 'Not specified')}")
        print(f"Total Rows: {len(self.data):,}")
        print(f"Total Columns: {len(self.data.columns)}")
        print(f"Memory Usage: {self.data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Project requirements validation
        print("\n2. PROJECT REQUIREMENTS VALIDATION")
        print("-" * 40)
        requirements = self.validate_project_requirements()
        
        requirements_score = sum(requirements.values()) / len(requirements) * 100
        print(f"\nOverall Requirements Compliance: {requirements_score:.1f}%")
        
        # Data quality assessment
        print("\n3. DATA QUALITY ASSESSMENT")
        print("-" * 40)
        quality_score = self._calculate_quality_score()
        print(f"Data Quality Score: {quality_score:.2f}/10.0")
        
        missing_summary = self.data.isnull().sum()
        print(f"Missing Values: {missing_summary.sum():,} ({missing_summary.sum()/(len(self.data)*len(self.data.columns))*100:.1f}%)")
        print(f"Duplicate Rows: {self.data.duplicated().sum():,}")
        
        # Feature analysis
        print("\n4. FEATURE ANALYSIS")
        print("-" * 40)
        print(f"Numeric Features: {len(self.numeric_columns)}")
        print(f"Categorical Features: {len(self.categorical_columns)}")
        
        if self.numeric_columns:
            print(f"\nNumeric Feature Statistics:")
            print(self.data[self.numeric_columns].describe().round(2))
        
        # Preprocessing recommendations
        print("\n5. PREPROCESSING RECOMMENDATIONS")
        print("-" * 40)
        
        opportunities = self._identify_preprocessing_opportunities()
        if opportunities['missing_values'] > 0:
            print(f"• Handle {opportunities['missing_values']:,} missing values")
        
        if opportunities['categorical_encoding'] > 0:
            print(f"• Encode {opportunities['categorical_encoding']} categorical features")
        
        if opportunities['numeric_scaling'] > 0:
            print(f"• Scale {opportunities['numeric_scaling']} numeric features")
        
        if opportunities['outliers'] > 0:
            print(f"• Address outliers in {opportunities['outliers']} columns")
        
        # Feature engineering suggestions
        print("\n6. FEATURE ENGINEERING SUGGESTIONS")
        print("-" * 40)
        suggestions = self.create_feature_engineering_suggestions()
        for i, suggestion in enumerate(suggestions[:10], 1):  # Show top 10
            print(f"{i}. {suggestion}")
        
        # ML readiness
        print("\n7. MACHINE LEARNING READINESS")
        print("-" * 40)
        ml_readiness = self._assess_ml_readiness()
        print(f"ML Readiness Score: {ml_readiness['score']:.2f}/1.0")
        
        if ml_readiness['issues']:
            print("Issues to address:")
            for issue in ml_readiness['issues']:
                print(f"• {issue}")
        
        # Preprocessing actions log
        if self.preprocessing_log:
            print("\n8. PREPROCESSING ACTIONS PERFORMED")
            print("-" * 40)
            for i, action in enumerate(self.preprocessing_log, 1):
                print(f"{i}. {action}")
        
        print("\n" + "="*80)
        print("REPORT COMPLETE - READY FOR FDM PROJECT SUBMISSION")
        print("="*80)
        
        return {
            'requirements_compliance': requirements_score,
            'data_quality_score': quality_score,
            'ml_readiness_score': ml_readiness['score'],
            'preprocessing_opportunities': opportunities,
            'feature_engineering_suggestions': suggestions
        }


def preprocess_for_fdm_project(file_path, dataset_name=None, target_column=None):
    """
    Complete preprocessing pipeline specifically for FDM Mini Project 2025.
    
    Args:
        file_path (str): Path to the dataset
        dataset_name (str): Name of the dataset for reporting
        target_column (str): Target variable for supervised learning
    
    Returns:
        FDMProjectPreprocessor: Configured preprocessor with processed data
    """
    print("FDM MINI PROJECT 2025 - DATA PREPROCESSING PIPELINE")
    print("="*60)
    
    # Initialize preprocessor
    preprocessor = FDMProjectPreprocessor()
    if dataset_name:
        preprocessor.dataset_name = dataset_name
    
    # Load and validate data
    print("\n🔍 Loading and validating dataset...")
    data = preprocessor.load_data(file_path, target_column)
    if data is None:
        print("❌ Failed to load dataset")
        return None
    
    # Validate project requirements
    print("\n📋 Validating FDM project requirements...")
    preprocessor.validate_project_requirements()
    
    # Exploratory Data Analysis
    print("\n📊 Performing Exploratory Data Analysis...")
    preprocessor.exploratory_data_analysis(save_plots=True)
    
    # Data preprocessing pipeline
    print("\n🔧 Starting data preprocessing pipeline...")
    
    # 1. Handle missing values
    print("\n1️⃣ Handling missing values...")
    preprocessor.handle_missing_values(strategy='auto')
    
    # 2. Detect and handle outliers
    print("\n2️⃣ Detecting and handling outliers...")
    outliers = preprocessor.detect_outliers(method='iqr')
    total_outliers = sum(info['count'] for info in outliers.values())
    if total_outliers > 0:
        print(f"Found {total_outliers:,} outliers across {len(outliers)} columns")
        preprocessor.handle_outliers(method='cap', outliers_info=outliers)
    
    # 3. Encode categorical features
    print("\n3️⃣ Encoding categorical features...")
    if preprocessor.categorical_columns:
        preprocessor.encode_categorical_features(encoding_type='auto')
    
    # 4. Scale numeric features
    print("\n4️⃣ Scaling numeric features...")
    if preprocessor.numeric_columns:
        preprocessor.scale_features(scaling_type='standard')
    
    # 5. Feature selection (if target is available)
    if target_column and target_column in preprocessor.data.columns:
        print("\n5️⃣ Performing feature selection...")
        try:
            selected_features = preprocessor.select_features(method='f_test', k=min(20, len(preprocessor.data.columns)-1))
            if selected_features:
                print(f"Selected {len(selected_features)} most important features")
        except Exception as e:
            print(f"⚠️  Feature selection failed: {str(e)}")
    
    # 6. Split data for ML (if target is available)
    if target_column and target_column in preprocessor.data.columns:
        print("\n6️⃣ Splitting data for machine learning...")
        split_results = preprocessor.split_data(test_size=0.2, validation_size=0.1)
        if split_results:
            X_train, X_val, X_test, y_train, y_val, y_test = split_results
            print(f"Data split completed: Train({len(X_train)}), Val({len(X_val)}), Test({len(X_test)})")
    
    # Generate comprehensive report
    print("\n📋 Generating FDM project report...")
    report = preprocessor.generate_fdm_project_report()
    
    # Save processed data
    output_file = file_path.replace('.csv', '_processed.csv').replace('.xlsx', '_processed.csv')
    preprocessor.save_processed_data(output_file)
    
    print(f"\n✅ Preprocessing complete! Processed data saved to: {output_file}")
    
    return preprocessor


if __name__ == "__main__":
    print("FDM Mini Project 2025 - Specific Requirements Implementation")
    print("This module extends the base preprocessing framework with:")
    print("• Project requirement validation (10,000+ rows, recent data)")
    print("• ML readiness assessment")
    print("• Feature engineering suggestions")
    print("• Comprehensive project reporting")
    print("\nUse preprocess_for_fdm_project() for complete FDM preprocessing pipeline")

# One-line complete preprocessing for your FDM project
from fdm_preprocessor import preprocess_for_fdm_project

preprocessor = preprocess_for_fdm_project(
    file_path='heart_disease.csv',           # Your 10,000+ row dataset
    dataset_name='Hearth Disease',     # Name for reports
    target_column='Heart Disease Status'     # What you're predicting
)