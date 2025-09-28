#!/usr/bin/env python3
"""
Comprehensive Data Preprocessing Framework for FDM Mini Project 2025
Author: Data Science Team
Purpose: Data preprocessing for datasets with 10,000+ rows for Data Mining/ML applications
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder, 
    OneHotEncoder, OrdinalEncoder
)
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import SelectKBest, f_classif, chi2, mutual_info_classif
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    """
    Comprehensive data preprocessing class for Data Mining projects.
    
    Key Features:
    - Data loading and initial analysis
    - Missing value handling
    - Outlier detection and treatment
    - Data type conversions
    - Feature encoding (categorical variables)
    - Feature scaling and normalization
    - Feature selection
    - Data splitting for ML models
    - Visualization and reporting
    """
    
    def __init__(self):
        self.data = None
        self.target_column = None
        self.numeric_columns = []
        self.categorical_columns = []
        self.preprocessing_log = []
        self.scalers = {}
        self.encoders = {}
        
    def log_action(self, action):
        """Log preprocessing actions for reporting."""
        self.preprocessing_log.append(action)
        print(f"✓ {action}")
    
    def load_data(self, file_path, target_column=None):
        """
        Load dataset from various file formats.
        
        Args:
            file_path (str): Path to the dataset file
            target_column (str): Name of the target column (for supervised learning)
        """
        try:
            if file_path.endswith('.csv'):
                self.data = pd.read_csv(file_path)
            elif file_path.endswith(('.xlsx', '.xls')):
                self.data = pd.read_excel(file_path)
            elif file_path.endswith('.json'):
                self.data = pd.read_json(file_path)
            else:
                raise ValueError("Unsupported file format. Use CSV, Excel, or JSON.")
            
            self.target_column = target_column
            self._identify_column_types()
            self.log_action(f"Loaded dataset with shape: {self.data.shape}")
            
            # Validate minimum row requirement
            if len(self.data) < 10000:
                print(f"⚠️  Warning: Dataset has only {len(self.data)} rows. Project requires 10,000+ rows.")
            
            return self.data
            
        except Exception as e:
            print(f"❌ Error loading data: {str(e)}")
            return None
    
    def _identify_column_types(self):
        """Identify numeric and categorical columns."""
        self.numeric_columns = self.data.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_columns = self.data.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Remove target column from feature lists if specified
        if self.target_column:
            if self.target_column in self.numeric_columns:
                self.numeric_columns.remove(self.target_column)
            if self.target_column in self.categorical_columns:
                self.categorical_columns.remove(self.target_column)
    
    def exploratory_data_analysis(self, save_plots=True):
        """
        Perform comprehensive EDA on the dataset.
        
        Args:
            save_plots (bool): Whether to save plots to files
        """
        print("\n" + "="*60)
        print("EXPLORATORY DATA ANALYSIS")
        print("="*60)
        
        # Basic info
        print(f"Dataset Shape: {self.data.shape}")
        print(f"Memory Usage: {self.data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Data types
        print("\nData Types:")
        print(self.data.dtypes.value_counts())
        
        # Missing values
        print("\nMissing Values:")
        missing_data = self.data.isnull().sum()
        missing_percent = (missing_data / len(self.data)) * 100
        missing_df = pd.DataFrame({
            'Column': missing_data.index,
            'Missing Count': missing_data.values,
            'Missing Percentage': missing_percent.values
        }).sort_values('Missing Percentage', ascending=False)
        print(missing_df[missing_df['Missing Count'] > 0])
        
        # Statistical summary
        print("\nStatistical Summary (Numeric Columns):")
        print(self.data[self.numeric_columns].describe())
        
        # Visualizations
        if save_plots:
            self._create_eda_plots()
        
        self.log_action("Completed Exploratory Data Analysis")
        return missing_df
    
    def _create_eda_plots(self):
        """Create and save EDA visualization plots."""
        
        # Missing values heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(self.data.isnull(), cbar=True, yticklabels=False, cmap='viridis')
        plt.title('Missing Values Heatmap')
        plt.tight_layout()
        plt.savefig('missing_values_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Correlation matrix for numeric columns
        if len(self.numeric_columns) > 1:
            plt.figure(figsize=(12, 10))
            correlation_matrix = self.data[self.numeric_columns].corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
            plt.title('Correlation Matrix of Numeric Features')
            plt.tight_layout()
            plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        # Distribution plots for numeric columns
        if len(self.numeric_columns) > 0:
            n_cols = min(4, len(self.numeric_columns))
            n_rows = (len(self.numeric_columns) + n_cols - 1) // n_cols
            
            plt.figure(figsize=(15, 4 * n_rows))
            for i, col in enumerate(self.numeric_columns[:12]):  # Limit to first 12 columns
                plt.subplot(n_rows, n_cols, i + 1)
                self.data[col].hist(bins=30, edgecolor='black')
                plt.title(f'Distribution of {col}')
                plt.xlabel(col)
                plt.ylabel('Frequency')
            
            plt.tight_layout()
            plt.savefig('numeric_distributions.png', dpi=300, bbox_inches='tight')
            plt.show()
    
    def handle_missing_values(self, strategy='auto', columns=None):
        """
        Handle missing values using various strategies.
        
        Args:
            strategy (str): 'auto', 'drop', 'mean', 'median', 'mode', 'knn', 'forward_fill', 'backward_fill'
            columns (list): Specific columns to process (None for all)
        """
        if columns is None:
            columns = self.data.columns.tolist()
        
        initial_shape = self.data.shape
        
        if strategy == 'auto':
            # Auto strategy: use different methods for different column types
            for col in columns:
                if col in self.data.columns:
                    missing_pct = (self.data[col].isnull().sum() / len(self.data)) * 100
                    
                    if missing_pct > 50:
                        # Drop columns with >50% missing values
                        self.data = self.data.drop(columns=[col])
                        self.log_action(f"Dropped column '{col}' (>{missing_pct:.1f}% missing)")
                    elif missing_pct > 0:
                        if col in self.numeric_columns:
                            # Use median for numeric columns
                            self.data[col].fillna(self.data[col].median(), inplace=True)
                        else:
                            # Use mode for categorical columns
                            mode_value = self.data[col].mode()
                            if len(mode_value) > 0:
                                self.data[col].fillna(mode_value[0], inplace=True)
                            else:
                                self.data[col].fillna('Unknown', inplace=True)
        
        elif strategy == 'drop':
            self.data = self.data.dropna(subset=columns)
        
        elif strategy in ['mean', 'median']:
            imputer = SimpleImputer(strategy=strategy)
            numeric_cols = [col for col in columns if col in self.numeric_columns]
            if numeric_cols:
                self.data[numeric_cols] = imputer.fit_transform(self.data[numeric_cols])
        
        elif strategy == 'mode':
            for col in columns:
                if col in self.data.columns:
                    mode_value = self.data[col].mode()
                    if len(mode_value) > 0:
                        self.data[col].fillna(mode_value[0], inplace=True)
        
        elif strategy == 'knn':
            numeric_cols = [col for col in columns if col in self.numeric_columns]
            if numeric_cols:
                imputer = KNNImputer(n_neighbors=5)
                self.data[numeric_cols] = imputer.fit_transform(self.data[numeric_cols])
        
        elif strategy == 'forward_fill':
            self.data[columns] = self.data[columns].fillna(method='ffill')
        
        elif strategy == 'backward_fill':
            self.data[columns] = self.data[columns].fillna(method='bfill')
        
        final_shape = self.data.shape
        self.log_action(f"Handled missing values using '{strategy}' strategy. Shape: {initial_shape} → {final_shape}")
        
        # Update column types after handling missing values
        self._identify_column_types()
    
    def detect_outliers(self, method='iqr', columns=None):
        """
        Detect outliers using various methods.
        
        Args:
            method (str): 'iqr', 'z_score', 'isolation_forest'
            columns (list): Columns to check for outliers
        
        Returns:
            dict: Dictionary with outlier information for each column
        """
        if columns is None:
            columns = self.numeric_columns
        
        outliers_info = {}
        
        for col in columns:
            if col in self.data.columns:
                outliers = []
                
                if method == 'iqr':
                    Q1 = self.data[col].quantile(0.25)
                    Q3 = self.data[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    outliers = self.data[(self.data[col] < lower_bound) | (self.data[col] > upper_bound)].index.tolist()
                
                elif method == 'z_score':
                    z_scores = np.abs(stats.zscore(self.data[col].dropna()))
                    outliers = self.data[col].dropna().index[z_scores > 3].tolist()
                
                outliers_info[col] = {
                    'count': len(outliers),
                    'percentage': (len(outliers) / len(self.data)) * 100,
                    'indices': outliers
                }
        
        self.log_action(f"Detected outliers using '{method}' method")
        return outliers_info
    
    def handle_outliers(self, method='cap', outliers_info=None, columns=None):
        """
        Handle outliers using various methods.
        
        Args:
            method (str): 'drop', 'cap', 'transform'
            outliers_info (dict): Output from detect_outliers method
            columns (list): Columns to process
        """
        if outliers_info is None:
            outliers_info = self.detect_outliers(columns=columns)
        
        initial_shape = self.data.shape
        
        for col, info in outliers_info.items():
            if col in self.data.columns and info['count'] > 0:
                
                if method == 'drop':
                    self.data = self.data.drop(info['indices'])
                
                elif method == 'cap':
                    Q1 = self.data[col].quantile(0.25)
                    Q3 = self.data[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    self.data[col] = np.where(self.data[col] < lower_bound, lower_bound, self.data[col])
                    self.data[col] = np.where(self.data[col] > upper_bound, upper_bound, self.data[col])
                
                elif method == 'transform':
                    # Log transformation (adding 1 to handle zeros)
                    if self.data[col].min() > 0:
                        self.data[col] = np.log1p(self.data[col])
        
        final_shape = self.data.shape
        self.log_action(f"Handled outliers using '{method}' method. Shape: {initial_shape} → {final_shape}")
    
    def encode_categorical_features(self, encoding_type='auto', columns=None):
        """
        Encode categorical features.
        
        Args:
            encoding_type (str): 'auto', 'label', 'onehot', 'ordinal'
            columns (list): Columns to encode
        """
        if columns is None:
            columns = self.categorical_columns
        
        for col in columns:
            if col in self.data.columns:
                unique_values = self.data[col].nunique()
                
                if encoding_type == 'auto':
                    # Auto strategy based on number of unique values
                    if unique_values <= 5:
                        # One-hot encoding for low cardinality
                        encoded_df = pd.get_dummies(self.data[col], prefix=col, drop_first=True)
                        self.data = self.data.drop(columns=[col])
                        self.data = pd.concat([self.data, encoded_df], axis=1)
                        self.log_action(f"One-hot encoded '{col}' ({unique_values} unique values)")
                    else:
                        # Label encoding for high cardinality
                        encoder = LabelEncoder()
                        self.data[col] = encoder.fit_transform(self.data[col].astype(str))
                        self.encoders[col] = encoder
                        self.log_action(f"Label encoded '{col}' ({unique_values} unique values)")
                
                elif encoding_type == 'label':
                    encoder = LabelEncoder()
                    self.data[col] = encoder.fit_transform(self.data[col].astype(str))
                    self.encoders[col] = encoder
                
                elif encoding_type == 'onehot':
                    encoded_df = pd.get_dummies(self.data[col], prefix=col, drop_first=True)
                    self.data = self.data.drop(columns=[col])
                    self.data = pd.concat([self.data, encoded_df], axis=1)
        
        # Update column types after encoding
        self._identify_column_types()
    
    def scale_features(self, scaling_type='standard', columns=None):
        """
        Scale/normalize numeric features.
        
        Args:
            scaling_type (str): 'standard', 'minmax', 'robust'
            columns (list): Columns to scale
        """
        if columns is None:
            columns = self.numeric_columns
        
        if scaling_type == 'standard':
            scaler = StandardScaler()
        elif scaling_type == 'minmax':
            scaler = MinMaxScaler()
        elif scaling_type == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError("Invalid scaling type. Use 'standard', 'minmax', or 'robust'.")
        
        scaled_columns = [col for col in columns if col in self.data.columns]
        if scaled_columns:
            self.data[scaled_columns] = scaler.fit_transform(self.data[scaled_columns])
            self.scalers[scaling_type] = scaler
            self.log_action(f"Applied {scaling_type} scaling to {len(scaled_columns)} columns")
    
    def select_features(self, method='correlation', k=10, target_column=None):
        """
        Select best features using various methods.
        
        Args:
            method (str): 'correlation', 'chi2', 'f_test', 'mutual_info'
            k (int): Number of features to select
            target_column (str): Target column for supervised feature selection
        """
        if target_column is None:
            target_column = self.target_column
        
        if target_column is None or target_column not in self.data.columns:
            print("❌ Target column not specified or not found. Cannot perform supervised feature selection.")
            return None
        
        feature_columns = [col for col in self.data.columns if col != target_column]
        X = self.data[feature_columns]
        y = self.data[target_column]
        
        if method == 'correlation':
            # Select features with highest correlation to target
            correlations = X.corrwith(y).abs().sort_values(ascending=False)
            selected_features = correlations.head(k).index.tolist()
        
        elif method in ['chi2', 'f_test', 'mutual_info']:
            if method == 'chi2':
                selector = SelectKBest(chi2, k=k)
            elif method == 'f_test':
                selector = SelectKBest(f_classif, k=k)
            elif method == 'mutual_info':
                selector = SelectKBest(mutual_info_classif, k=k)
            
            # Ensure all features are non-negative for chi2 test
            if method == 'chi2' and X.min().min() < 0:
                print("⚠️  Chi2 test requires non-negative features. Converting to non-negative values.")
                X = X - X.min()
            
            selector.fit(X, y)
            selected_features = X.columns[selector.get_support()].tolist()
        
        self.log_action(f"Selected top {len(selected_features)} features using '{method}' method")
        return selected_features
    
    def split_data(self, test_size=0.2, validation_size=0.1, random_state=42, stratify=None):
        """
        Split data into train, validation, and test sets.
        
        Args:
            test_size (float): Proportion of test set
            validation_size (float): Proportion of validation set
            random_state (int): Random state for reproducibility
            stratify (str): Column name for stratified splitting
        
        Returns:
            tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        if self.target_column is None or self.target_column not in self.data.columns:
            print("❌ Target column not specified or not found.")
            return None
        
        feature_columns = [col for col in self.data.columns if col != self.target_column]
        X = self.data[feature_columns]
        y = self.data[self.target_column]
        
        stratify_param = y if stratify == self.target_column else None
        
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=stratify_param
        )
        
        # Second split: separate train and validation from remaining data
        val_size_adjusted = validation_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state,
            stratify=y_temp if stratify == self.target_column else None
        )
        
        self.log_action(f"Split data: Train({len(X_train)}), Val({len(X_val)}), Test({len(X_test)})")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def generate_preprocessing_report(self):
        """Generate a comprehensive preprocessing report."""
        print("\n" + "="*80)
        print("DATA PREPROCESSING REPORT")
        print("="*80)
        
        print("Final Dataset Information:")
        print(f"Shape: {self.data.shape}")
        print(f"Numeric Columns: {len(self.numeric_columns)}")
        print(f"Categorical Columns: {len(self.categorical_columns)}")
        print(f"Memory Usage: {self.data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        print(f"\nMissing Values: {self.data.isnull().sum().sum()}")
        print(f"Duplicate Rows: {self.data.duplicated().sum()}")
        
        print("\nPreprocessing Actions Performed:")
        for i, action in enumerate(self.preprocessing_log, 1):
            print(f"{i}. {action}")
        
        print("\nData Quality Score:")
        quality_score = self._calculate_quality_score()
        print(f"Overall Quality: {quality_score:.2f}/10.0")
        
        return quality_score
    
    def _calculate_quality_score(self):
        """Calculate a data quality score based on various metrics."""
        score = 10.0
        
        # Penalize for missing values
        missing_ratio = self.data.isnull().sum().sum() / (self.data.shape[0] * self.data.shape[1])
        score -= missing_ratio * 3
        
        # Penalize for duplicate rows
        duplicate_ratio = self.data.duplicated().sum() / len(self.data)
        score -= duplicate_ratio * 2
        
        # Reward for sufficient data size
        if len(self.data) >= 10000:
            score += 0.5
        
        return max(0, min(10, score))
    
    def save_processed_data(self, file_path):
        """Save the processed dataset to a file."""
        try:
            if file_path.endswith('.csv'):
                self.data.to_csv(file_path, index=False)
            elif file_path.endswith(('.xlsx', '.xls')):
                self.data.to_excel(file_path, index=False)
            else:
                # Default to CSV
                file_path += '.csv'
                self.data.to_csv(file_path, index=False)
            
            self.log_action(f"Saved processed data to: {file_path}")
            return file_path
        
        except Exception as e:
            print(f"❌ Error saving data: {str(e)}")
            return None


# Convenience function for quick preprocessing
def quick_preprocess(file_path, target_column=None, test_size=0.2):
    """
    Perform quick preprocessing with default settings.
    
    Args:
        file_path (str): Path to dataset
        target_column (str): Target column name
        test_size (float): Test set proportion
    
    Returns:
        tuple: (preprocessor, train_test_split_results)
    """
    preprocessor = DataPreprocessor()
    
    # Load data
    data = preprocessor.load_data(file_path, target_column)
    if data is None:
        return None, None
    
    # EDA
    preprocessor.exploratory_data_analysis()
    
    # Handle missing values
    preprocessor.handle_missing_values(strategy='auto')
    
    # Handle outliers
    outliers = preprocessor.detect_outliers(method='iqr')
    preprocessor.handle_outliers(method='cap', outliers_info=outliers)
    
    # Encode categorical features
    preprocessor.encode_categorical_features(encoding_type='auto')
    
    # Scale features
    preprocessor.scale_features(scaling_type='standard')
    
    # Split data (if target column is provided)
    split_results = None
    if target_column:
        split_results = preprocessor.split_data(test_size=test_size)
    
    # Generate report
    preprocessor.generate_preprocessing_report()
    
    return preprocessor, split_results


if __name__ == "__main__":
    print("Data Preprocessing Framework for FDM Mini Project 2025")
    print("="*60)
    print("This framework provides comprehensive data preprocessing capabilities:")
    print("- Load data from CSV, Excel, or JSON formats")
    print("- Exploratory Data Analysis (EDA)")
    print("- Missing value handling")
    print("- Outlier detection and treatment")
    print("- Categorical feature encoding")
    print("- Feature scaling and normalization")
    print("- Feature selection")
    print("- Data splitting for ML models")
    print("- Comprehensive reporting")
    print("\nUse quick_preprocess() for automated preprocessing or")
    print("DataPreprocessor class for custom preprocessing workflows.")