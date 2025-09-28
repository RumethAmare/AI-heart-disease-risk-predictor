#!/usr/bin/env python3
"""
FDM Mini Project 2025 - Example Usage and Documentation
Complete examples showing how to use the data preprocessing framework
"""

import pandas as pd
import numpy as np
from fdm_preprocessor import preprocess_for_fdm_project, FDMProjectPreprocessor
from data_preprocessor import DataPreprocessor, quick_preprocess

# Example 1: Create a sample dataset for demonstration
def create_sample_dataset():
    """Create a sample dataset that meets FDM project requirements (10,000+ rows)."""
    
    print("Creating sample dataset for demonstration...")
    
    np.random.seed(42)
    n_rows = 12000  # Meets the 10,000+ requirement
    
    # Generate sample data
    data = {
        'customer_id': range(1, n_rows + 1),
        'age': np.random.normal(35, 12, n_rows).astype(int),
        'income': np.random.exponential(50000, n_rows),
        'education_level': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_rows, p=[0.4, 0.4, 0.15, 0.05]),
        'employment_status': np.random.choice(['Employed', 'Unemployed', 'Self-Employed', 'Retired'], n_rows, p=[0.7, 0.1, 0.15, 0.05]),
        'credit_score': np.random.normal(650, 100, n_rows).astype(int),
        'loan_amount': np.random.exponential(25000, n_rows),
        'loan_term': np.random.choice([12, 24, 36, 48, 60], n_rows),
        'interest_rate': np.random.normal(8.5, 2.5, n_rows),
        'monthly_payment': np.random.normal(800, 300, n_rows),
        'city': np.random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 'Philadelphia'], n_rows),
        'loan_purpose': np.random.choice(['Personal', 'Business', 'Education', 'Home', 'Car'], n_rows),
        'application_date': pd.date_range('2022-01-01', '2024-09-01', periods=n_rows),
    }
    
    df = pd.DataFrame(data)
    
    # Add some realistic relationships
    df['monthly_payment'] = df['loan_amount'] * (df['interest_rate'] / 100) / 12 * (1 + np.random.normal(0, 0.1, n_rows))
    df['debt_to_income'] = df['monthly_payment'] * 12 / df['income']
    
    # Create target variable (loan approval - binary classification)
    # Higher income, better credit score, lower debt-to-income ratio = higher approval chance
    approval_probability = (
        0.3 +
        0.3 * (df['credit_score'] - 300) / (850 - 300) +  # Credit score factor
        0.2 * (df['income'] - df['income'].min()) / (df['income'].max() - df['income'].min()) +  # Income factor
        0.2 * (1 - np.clip(df['debt_to_income'], 0, 1))  # Debt-to-income factor (inverted)
    )
    
    df['loan_approved'] = np.random.binomial(1, approval_probability, n_rows)
    
    # Introduce some missing values (realistic scenario)
    missing_indices = np.random.choice(df.index, size=int(0.05 * len(df)), replace=False)
    df.loc[missing_indices, 'credit_score'] = np.nan
    
    missing_indices = np.random.choice(df.index, size=int(0.02 * len(df)), replace=False)
    df.loc[missing_indices, 'income'] = np.nan
    
    missing_indices = np.random.choice(df.index, size=int(0.01 * len(df)), replace=False)
    df.loc[missing_indices, 'employment_status'] = np.nan
    
    # Add some outliers
    outlier_indices = np.random.choice(df.index, size=int(0.01 * len(df)), replace=False)
    df.loc[outlier_indices, 'income'] *= np.random.uniform(5, 10, len(outlier_indices))
    
    # Save sample dataset
    sample_file = 'sample_loan_dataset.csv'
    df.to_csv(sample_file, index=False)
    print(f"✅ Sample dataset created: {sample_file}")
    print(f"Dataset shape: {df.shape}")
    print(f"Target variable: loan_approved")
    
    return sample_file

# Example 2: Basic preprocessing workflow
def example_basic_preprocessing():
    """Demonstrate basic preprocessing workflow."""
    
    print("\n" + "="*80)
    print("EXAMPLE 1: BASIC PREPROCESSING WORKFLOW")
    print("="*80)
    
    # Create sample data
    sample_file = create_sample_dataset()
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Step 1: Load data
    print("\n1. Loading data...")
    data = preprocessor.load_data(sample_file, target_column='loan_approved')
    
    # Step 2: Exploratory Data Analysis
    print("\n2. Performing EDA...")
    missing_df = preprocessor.exploratory_data_analysis(save_plots=False)  # Skip plots for example
    
    # Step 3: Handle missing values
    print("\n3. Handling missing values...")
    preprocessor.handle_missing_values(strategy='auto')
    
    # Step 4: Detect and handle outliers
    print("\n4. Handling outliers...")
    outliers = preprocessor.detect_outliers(method='iqr')
    preprocessor.handle_outliers(method='cap', outliers_info=outliers)
    
    # Step 5: Encode categorical features
    print("\n5. Encoding categorical features...")
    preprocessor.encode_categorical_features(encoding_type='auto')
    
    # Step 6: Scale features
    print("\n6. Scaling features...")
    preprocessor.scale_features(scaling_type='standard')
    
    # Step 7: Split data
    print("\n7. Splitting data...")
    split_results = preprocessor.split_data(test_size=0.2, validation_size=0.1)
    
    # Step 8: Generate report
    print("\n8. Generating report...")
    quality_score = preprocessor.generate_preprocessing_report()
    
    return preprocessor, split_results

# Example 3: FDM Project specific preprocessing
def example_fdm_preprocessing():
    """Demonstrate FDM project specific preprocessing."""
    
    print("\n" + "="*80)
    print("EXAMPLE 2: FDM PROJECT SPECIFIC PREPROCESSING")
    print("="*80)
    
    # Create sample data
    sample_file = create_sample_dataset()
    
    # Use FDM-specific preprocessing pipeline
    preprocessor = preprocess_for_fdm_project(
        file_path=sample_file,
        dataset_name="Loan Approval Dataset",
        target_column='loan_approved'
    )
    
    return preprocessor

# Example 4: Custom preprocessing workflow
def example_custom_preprocessing():
    """Demonstrate custom preprocessing workflow for specific needs."""
    
    print("\n" + "="*80)
    print("EXAMPLE 3: CUSTOM PREPROCESSING WORKFLOW")
    print("="*80)
    
    # Create sample data
    sample_file = create_sample_dataset()
    
    # Initialize FDM preprocessor
    preprocessor = FDMProjectPreprocessor()
    
    print("\n🔧 Custom preprocessing workflow for loan approval prediction...")
    
    # Load data
    data = preprocessor.load_data(sample_file, target_column='loan_approved')
    
    # Custom validation
    print("\n📋 Validating FDM requirements...")
    preprocessor.validate_project_requirements()
    
    # Custom EDA focusing on business insights
    print("\n📊 Business-focused EDA...")
    print("\nLoan Approval Rate by Education Level:")
    approval_by_education = data.groupby('education_level')['loan_approved'].mean().sort_values(ascending=False)
    print(approval_by_education)
    
    print("\nAverage Income by Employment Status:")
    income_by_employment = data.groupby('employment_status')['income'].mean().sort_values(ascending=False)
    print(income_by_employment)
    
    print("\nCorrelation with Loan Approval:")
    numeric_cols = preprocessor.numeric_columns + ['loan_approved']
    correlations = data[numeric_cols].corr()['loan_approved'].sort_values(ascending=False)
    print(correlations[:-1])  # Exclude self-correlation\n    \n    # Custom missing value handling for business rules\n    print(\"\\n🔧 Custom missing value handling...\")\n    # For credit scores, use median by employment status\n    for status in data['employment_status'].unique():\n        if pd.notna(status):\n            mask = (data['employment_status'] == status) & (data['credit_score'].isna())\n            median_score = data[data['employment_status'] == status]['credit_score'].median()\n            data.loc[mask, 'credit_score'] = median_score\n            if mask.sum() > 0:\n                print(f\"Filled {mask.sum()} missing credit scores for {status} with median: {median_score:.0f}\")\n    \n    # Update preprocessor data\n    preprocessor.data = data\n    preprocessor._identify_column_types()\n    \n    # Feature engineering for loan domain\n    print(\"\\n⚙️ Domain-specific feature engineering...\")\n    \n    # Create age groups\n    data['age_group'] = pd.cut(data['age'], bins=[0, 25, 35, 45, 55, 100], \n                              labels=['Young', 'Early_Career', 'Mid_Career', 'Senior', 'Retirement'])\n    \n    # Create income brackets\n    data['income_bracket'] = pd.qcut(data['income'], q=5, labels=['Low', 'Lower_Mid', 'Mid', 'Upper_Mid', 'High'])\n    \n    # Create credit score categories\n    data['credit_category'] = pd.cut(data['credit_score'], \n                                   bins=[0, 580, 670, 740, 800, 850],\n                                   labels=['Poor', 'Fair', 'Good', 'Very_Good', 'Excellent'])\n    \n    # Create loan-to-income ratio\n    data['loan_to_income_ratio'] = data['loan_amount'] / data['income']\n    \n    # Create seasonal features from application date\n    data['application_month'] = data['application_date'].dt.month\n    data['application_quarter'] = data['application_date'].dt.quarter\n    data['days_since_application'] = (data['application_date'].max() - data['application_date']).dt.days\n    \n    print(\"Created new features: age_group, income_bracket, credit_category, loan_to_income_ratio, seasonal features\")\n    \n    # Update preprocessor with engineered features\n    preprocessor.data = data\n    preprocessor._identify_column_types()\n    \n    # Continue with standard preprocessing\n    print(\"\\n🔄 Continuing with standard preprocessing...\")\n    \n    # Handle remaining missing values\n    preprocessor.handle_missing_values(strategy='auto')\n    \n    # Encode categorical features\n    preprocessor.encode_categorical_features(encoding_type='auto')\n    \n    # Scale features\n    preprocessor.scale_features(scaling_type='robust')  # Robust scaling for financial data\n    \n    # Feature selection\n    selected_features = preprocessor.select_features(method='mutual_info', k=15)\n    print(f\"\\n🎯 Selected top 15 features using mutual information\")\n    \n    # Split data\n    split_results = preprocessor.split_data(test_size=0.2, validation_size=0.1, stratify='loan_approved')\n    \n    # Generate comprehensive report\n    report = preprocessor.generate_fdm_project_report()\n    \n    return preprocessor, split_results, report\n\n# Example 5: Quick preprocessing for rapid prototyping\ndef example_quick_preprocessing():\n    \"\"\"Demonstrate quick preprocessing for rapid prototyping.\"\"\"\n    \n    print(\"\\n\" + \"=\"*80)\n    print(\"EXAMPLE 4: QUICK PREPROCESSING FOR RAPID PROTOTYPING\")\n    print(\"=\"*80)\n    \n    # Create sample data\n    sample_file = create_sample_dataset()\n    \n    # Use quick preprocessing function\n    preprocessor, split_results = quick_preprocess(\n        file_path=sample_file,\n        target_column='loan_approved',\n        test_size=0.2\n    )\n    \n    if split_results:\n        X_train, X_val, X_test, y_train, y_val, y_test = split_results\n        print(f\"\\n✅ Quick preprocessing complete!\")\n        print(f\"Ready for ML: Train({len(X_train)}), Val({len(X_val)}), Test({len(X_test)})\")\n    \n    return preprocessor, split_results\n\n# Documentation and usage guide\ndef print_usage_guide():\n    \"\"\"Print comprehensive usage guide.\"\"\"\n    \n    print(\"\\n\" + \"=\"*100)\n    print(\"FDM MINI PROJECT 2025 - DATA PREPROCESSING FRAMEWORK USAGE GUIDE\")\n    print(\"=\"*100)\n    \n    print(\"\"\"\n🎯 PURPOSE:\nThis framework provides comprehensive data preprocessing capabilities specifically designed\nfor the FDM Mini Project 2025, supporting datasets with 10,000+ rows and various\nData Mining/Machine Learning techniques.\n\n📚 AVAILABLE CLASSES:\n1. DataPreprocessor - Base preprocessing class with core functionality\n2. FDMProjectPreprocessor - Extended class with FDM-specific requirements\n\n🔧 KEY FUNCTIONS:\n1. preprocess_for_fdm_project() - Complete FDM preprocessing pipeline\n2. quick_preprocess() - Rapid preprocessing for prototyping\n\n📋 PREPROCESSING CAPABILITIES:\n✓ Data loading (CSV, Excel, JSON)\n✓ Exploratory Data Analysis (EDA)\n✓ Missing value handling (multiple strategies)\n✓ Outlier detection and treatment\n✓ Categorical feature encoding\n✓ Feature scaling and normalization\n✓ Feature selection\n✓ Data splitting for ML\n✓ Comprehensive reporting\n✓ FDM project requirement validation\n\n📊 USAGE EXAMPLES:\n\n1. BASIC USAGE:\n```python\nfrom fdm_preprocessor import preprocess_for_fdm_project\n\n# Complete FDM preprocessing pipeline\npreprocessor = preprocess_for_fdm_project(\n    file_path='your_dataset.csv',\n    dataset_name='Your Dataset Name',\n    target_column='target_variable'\n)\n```\n\n2. STEP-BY-STEP PROCESSING:\n```python\nfrom data_preprocessor import DataPreprocessor\n\npreprocessor = DataPreprocessor()\ndata = preprocessor.load_data('dataset.csv', 'target_column')\npreprocessor.exploratory_data_analysis()\npreprocessor.handle_missing_values(strategy='auto')\npreprocessor.encode_categorical_features(encoding_type='auto')\npreprocessor.scale_features(scaling_type='standard')\nsplit_results = preprocessor.split_data(test_size=0.2)\n```\n\n3. QUICK PROTOTYPING:\n```python\nfrom data_preprocessor import quick_preprocess\n\npreprocessor, splits = quick_preprocess(\n    file_path='dataset.csv',\n    target_column='target',\n    test_size=0.2\n)\n```\n\n🔍 FDM PROJECT REQUIREMENTS:\n✓ Minimum 10,000 rows\n✓ Recent data (within last 3 years)\n✓ Preprocessing opportunities identified\n✓ ML-ready format\n✓ Comprehensive documentation\n\n📁 OUTPUT FILES:\n• processed_dataset.csv - Cleaned and processed data\n• missing_values_heatmap.png - Missing values visualization\n• correlation_matrix.png - Feature correlation heatmap\n• numeric_distributions.png - Distribution plots\n• project_instructions.txt - Extracted PDF content\n\n⚙️ CUSTOMIZATION OPTIONS:\n• Missing value strategies: 'auto', 'drop', 'mean', 'median', 'mode', 'knn'\n• Outlier handling: 'drop', 'cap', 'transform'\n• Encoding methods: 'auto', 'label', 'onehot', 'ordinal'\n• Scaling methods: 'standard', 'minmax', 'robust'\n• Feature selection: 'correlation', 'chi2', 'f_test', 'mutual_info'\n\n📊 REPORTING FEATURES:\n• Data quality assessment\n• FDM requirement validation\n• ML readiness evaluation\n• Feature engineering suggestions\n• Preprocessing action log\n• Comprehensive project reports\n\n🚀 GETTING STARTED:\n1. Run example_basic_preprocessing() for a simple workflow\n2. Run example_fdm_preprocessing() for FDM-specific processing\n3. Run example_custom_preprocessing() for advanced customization\n4. Run example_quick_preprocessing() for rapid prototyping\n\n💡 TIPS:\n• Always validate your dataset meets FDM requirements (10,000+ rows)\n• Use EDA to understand your data before preprocessing\n• Consider domain-specific feature engineering\n• Save processed data for reproducibility\n• Review the generated reports for insights\n\n🆘 SUPPORT:\n• Check the preprocessing log for action history\n• Use validate_project_requirements() to check FDM compliance\n• Review feature engineering suggestions for domain improvements\n• Consult the ML readiness assessment for preparation status\n\"\"\")\n\ndef run_all_examples():\n    \"\"\"Run all examples in sequence.\"\"\"\n    \n    print(\"\\n🚀 RUNNING ALL PREPROCESSING EXAMPLES...\")\n    \n    try:\n        # Print usage guide\n        print_usage_guide()\n        \n        # Run examples\n        print(\"\\n\" + \"⏭️ \"*20)\n        example1_results = example_basic_preprocessing()\n        \n        print(\"\\n\" + \"⏭️ \"*20)\n        example2_results = example_fdm_preprocessing()\n        \n        print(\"\\n\" + \"⏭️ \"*20)\n        example3_results = example_custom_preprocessing()\n        \n        print(\"\\n\" + \"⏭️ \"*20)\n        example4_results = example_quick_preprocessing()\n        \n        print(\"\\n\" + \"🎉\"*20)\n        print(\"ALL EXAMPLES COMPLETED SUCCESSFULLY!\")\n        print(\"🎉\"*20)\n        \n    except Exception as e:\n        print(f\"❌ Error running examples: {str(e)}\")\n        import traceback\n        traceback.print_exc()\n\nif __name__ == \"__main__\":\n    # Run all examples when script is executed directly\n    run_all_examples()