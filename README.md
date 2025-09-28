# FDM Mini Project 2025 - Data Preprocessing Framework

## 📋 Project Overview

This comprehensive data preprocessing framework is specifically designed for the **Fundamentals of Data Mining (FDM) Mini Project 2025** at Sri Lanka Institute of Information Technology (SLIIT). The framework supports datasets with **10,000+ rows** and provides advanced preprocessing capabilities for Data Mining and Machine Learning applications.

## 🎯 Project Requirements Compliance

### ✅ FDM Project Requirements Met:
- ✓ Supports datasets with **10,000+ rows**
- ✓ Handles **recent data** preprocessing
- ✓ **Real-world problem** focus (loan approval prediction example)
- ✓ **Data Mining/ML techniques** ready
- ✓ Comprehensive **preprocessing pipeline**
- ✓ **Jupyter notebook** compatible
- ✓ Complete **documentation** and examples

## 📁 File Structure

```
FDM Project/
├── 📄 FDM - IT3051- Mini Project - 2025 - finalized instructions.pdf
├── 🐍 pdf_reader.py                    # PDF content extraction
├── 🐍 data_preprocessor.py             # Base preprocessing framework
├── 🐍 fdm_preprocessor.py              # FDM-specific preprocessing
├── 🐍 examples_and_documentation.py    # Usage examples
├── 📄 project_instructions.txt         # Extracted PDF content
├── 📄 README.md                        # This documentation
├── 📄 sample_loan_dataset.csv          # Generated sample data
└── 📊 Generated visualizations (PNG files)
```

## 🚀 Quick Start

### 1. Complete FDM Preprocessing Pipeline
```python
from fdm_preprocessor import preprocess_for_fdm_project

# One-line complete preprocessing
preprocessor = preprocess_for_fdm_project(
    file_path='your_dataset.csv',
    dataset_name='Your Dataset Name',
    target_column='target_variable'
)
```

### 2. Step-by-Step Processing
```python
from data_preprocessor import DataPreprocessor

preprocessor = DataPreprocessor()
data = preprocessor.load_data('dataset.csv', 'target_column')
preprocessor.exploratory_data_analysis()
preprocessor.handle_missing_values(strategy='auto')
preprocessor.encode_categorical_features(encoding_type='auto')
preprocessor.scale_features(scaling_type='standard')
splits = preprocessor.split_data(test_size=0.2)
```

### 3. Quick Prototyping
```python
from data_preprocessor import quick_preprocess

preprocessor, splits = quick_preprocess(
    file_path='dataset.csv',
    target_column='target',
    test_size=0.2
)
```

## 🔧 Key Features

### 📊 Data Loading & Validation
- **Multiple formats**: CSV, Excel, JSON support
- **FDM requirements validation**: Automatic check for 10,000+ rows
- **Data recency verification**: Identifies recent data (within 3 years)
- **Memory usage optimization**: Efficient handling of large datasets

### 🔍 Exploratory Data Analysis (EDA)
- **Comprehensive statistics**: Detailed data summary
- **Missing value analysis**: Visual heatmaps and statistics
- **Correlation analysis**: Feature relationship insights
- **Distribution visualization**: Automatic plot generation
- **Data quality scoring**: 0-10 quality assessment

### 🧹 Data Cleaning & Preprocessing
- **Missing value handling**: Multiple strategies (auto, mean, median, mode, KNN)
- **Outlier detection**: IQR, Z-score, Isolation Forest methods
- **Outlier treatment**: Drop, cap, or transform options
- **Duplicate removal**: Automatic duplicate detection
- **Data type optimization**: Automatic type inference and conversion

### 🏷️ Feature Engineering
- **Categorical encoding**: Label, One-hot, Ordinal encoding
- **Feature scaling**: Standard, MinMax, Robust scaling
- **Feature selection**: Correlation, Chi-square, F-test, Mutual information
- **Date feature extraction**: Year, month, season, day of week
- **Custom feature creation**: Domain-specific feature engineering

### 🎯 Machine Learning Preparation
- **Data splitting**: Train/validation/test splits with stratification
- **ML readiness assessment**: Automated evaluation (0-1 score)
- **Feature importance ranking**: Multiple selection methods
- **Preprocessing pipeline saving**: Reproducible transformations

### 📋 Comprehensive Reporting
- **FDM compliance report**: Requirements validation
- **Data quality assessment**: Multi-metric evaluation
- **Preprocessing action log**: Complete audit trail
- **Feature engineering suggestions**: Domain-specific recommendations
- **ML readiness evaluation**: Preparation status assessment

## 📚 Available Classes

### 1. `DataPreprocessor`
**Base preprocessing class with core functionality**

```python
preprocessor = DataPreprocessor()
preprocessor.load_data(file_path, target_column)
preprocessor.exploratory_data_analysis()
preprocessor.handle_missing_values(strategy='auto')
preprocessor.encode_categorical_features(encoding_type='auto')
preprocessor.scale_features(scaling_type='standard')
```

### 2. `FDMProjectPreprocessor`
**Extended class with FDM-specific requirements**

```python
from fdm_preprocessor import FDMProjectPreprocessor

preprocessor = FDMProjectPreprocessor()
preprocessor.validate_project_requirements()
preprocessor.generate_fdm_project_report()
```

## 🛠️ Preprocessing Options

### Missing Value Strategies
- `'auto'` - Intelligent strategy selection based on data type and missing percentage
- `'drop'` - Remove rows/columns with missing values
- `'mean'` - Fill with mean (numeric columns)
- `'median'` - Fill with median (numeric columns)
- `'mode'` - Fill with mode (categorical columns)
- `'knn'` - KNN imputation for complex patterns
- `'forward_fill'` - Forward fill for time series
- `'backward_fill'` - Backward fill for time series

### Outlier Handling Methods
- `'iqr'` - Interquartile Range method
- `'z_score'` - Z-score based detection (>3 standard deviations)
- `'isolation_forest'` - Machine learning based detection

### Outlier Treatment Options
- `'drop'` - Remove outlier observations
- `'cap'` - Cap at reasonable bounds (1.5 * IQR)
- `'transform'` - Log transformation to reduce impact

### Feature Encoding Methods
- `'auto'` - Smart encoding based on cardinality
- `'label'` - Label encoding (ordinal mapping)
- `'onehot'` - One-hot encoding (binary features)
- `'ordinal'` - Ordinal encoding with specified order

### Feature Scaling Methods
- `'standard'` - StandardScaler (mean=0, std=1)
- `'minmax'` - MinMaxScaler (range 0-1)
- `'robust'` - RobustScaler (median-based, outlier-resistant)

### Feature Selection Methods
- `'correlation'` - Correlation with target variable
- `'chi2'` - Chi-square test for independence
- `'f_test'` - F-statistic for classification
- `'mutual_info'` - Mutual information score

## 📊 Example Datasets

### Included Sample Dataset: Loan Approval Prediction
**Features:**
- **Target Variable**: `loan_approved` (binary classification)
- **Numeric Features**: age, income, credit_score, loan_amount, interest_rate, monthly_payment, debt_to_income
- **Categorical Features**: education_level, employment_status, city, loan_purpose
- **Date Features**: application_date
- **Dataset Size**: 12,000+ rows (meets FDM requirements)
- **Preprocessing Challenges**: Missing values, outliers, categorical encoding, feature engineering

## 🔍 Usage Examples

### Example 1: Complete FDM Pipeline
```python
# Run complete FDM preprocessing pipeline
from examples_and_documentation import example_fdm_preprocessing

preprocessor = example_fdm_preprocessing()
```

### Example 2: Custom Business Logic
```python
# Run custom preprocessing with domain knowledge
from examples_and_documentation import example_custom_preprocessing

preprocessor, splits, report = example_custom_preprocessing()
```

### Example 3: Quick Prototyping
```python
# Rapid preprocessing for quick experimentation
from examples_and_documentation import example_quick_preprocessing

preprocessor, splits = example_quick_preprocessing()
```

### Example 4: Run All Examples
```python
# Execute all examples in sequence
from examples_and_documentation import run_all_examples

run_all_examples()
```

## 📈 Output Files

### Processed Data
- `*_processed.csv` - Cleaned and processed dataset
- Maintains original structure with preprocessing applied
- Ready for machine learning algorithms

### Visualizations
- `missing_values_heatmap.png` - Missing data patterns
- `correlation_matrix.png` - Feature correlations
- `numeric_distributions.png` - Feature distributions
- High-resolution (300 DPI) for report inclusion

### Reports
- `project_instructions.txt` - Extracted PDF requirements
- Comprehensive console reports with metrics and recommendations
- FDM project compliance validation

## 🎯 FDM Project Integration

### For Your FDM Project:

1. **Replace Sample Data**: Use your chosen dataset (ensure 10,000+ rows)
2. **Specify Target Variable**: Define your prediction target
3. **Custom Feature Engineering**: Add domain-specific features
4. **Model Integration**: Use processed data with your ML algorithms
5. **Report Generation**: Include preprocessing reports in your submission

### Project Deliverables Support:
- ✅ **SOW Documentation**: Preprocessing approach description
- ✅ **Final Report**: Comprehensive data preparation section
- ✅ **Software Solution**: Production-ready preprocessing code
- ✅ **Video Presentation**: Visual aids and process demonstration

## 🚨 Important Notes

### Data Requirements:
- **Minimum 10,000 rows** as per project requirements
- **Recent data** preferred (within last 3 years)
- **Publicly available datasets** recommended
- **Real-world problems** focus required

### Best Practices:
- Always run EDA before preprocessing
- Validate FDM requirements early
- Save original data before modifications
- Document all preprocessing steps
- Test with train/validation/test splits
- Review generated reports for insights

## 🆘 Troubleshooting

### Common Issues:

**"Dataset too small" warning:**
- Ensure your dataset has 10,000+ rows
- Consider data augmentation techniques
- Look for additional data sources

**Missing value handling errors:**
- Check for completely empty columns
- Verify data types are correct
- Try different imputation strategies

**Encoding failures:**
- Check for special characters in categorical data
- Verify string encoding (UTF-8 recommended)
- Handle high cardinality categories appropriately

**Memory issues with large datasets:**
- Process data in chunks
- Use data type optimization
- Consider sampling for initial exploration

## 👨‍💻 Development Team

**FDM Mini Project 2025**
- Framework designed for SLIIT FDM course requirements
- Supports real-world data mining problems
- Production-ready preprocessing pipeline

## 📞 Support

For FDM project specific questions:
- Review the comprehensive examples
- Check preprocessing logs for debugging
- Validate requirements using built-in functions
- Refer to generated reports for insights

---

**Ready to preprocess your FDM project data? Start with the examples and adapt for your specific dataset!** 🚀