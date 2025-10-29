#!/usr/bin/env python3
"""
Dataset Statistics Generator
Analyzes the heart disease dataset and generates comprehensive statistics for visualization
"""

import pandas as pd
import numpy as np
import json
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def generate_comprehensive_statistics():
    """Generate comprehensive statistics for all attributes in the dataset"""
    
    print("🔄 Loading heart disease dataset...")
    df = pd.read_csv('heart_disease_extended.csv')
    
    print(f"📊 Dataset loaded: {df.shape[0]} records, {df.shape[1]} attributes")
    
    # Initialize statistics dictionary
    statistics = {
        'overview': {},
        'numerical_attributes': {},
        'categorical_attributes': {},
        'correlation_matrix': {},
        'heart_disease_analysis': {},
        'missing_values': {}
    }
    
    # Generate overview statistics
    statistics['overview'] = generate_overview_stats(df)
    
    # Identify numerical and categorical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Remove target variable from analysis
    if 'Heart Disease Status' in numerical_cols:
        numerical_cols.remove('Heart Disease Status')
    if 'Heart Disease Status' in categorical_cols:
        categorical_cols.remove('Heart Disease Status')
    
    print(f"📈 Analyzing {len(numerical_cols)} numerical attributes...")
    # Generate numerical attributes statistics
    for col in numerical_cols:
        if col != 'Heart Disease Status':
            statistics['numerical_attributes'][col] = analyze_numerical_attribute(df, col)
    
    print(f"📋 Analyzing {len(categorical_cols)} categorical attributes...")
    # Generate categorical attributes statistics
    for col in categorical_cols:
        if col != 'Heart Disease Status':
            statistics['categorical_attributes'][col] = analyze_categorical_attribute(df, col)
    
    # Generate correlation matrix (numerical features only)
    print("🔗 Computing correlation matrix...")
    statistics['correlation_matrix'] = generate_correlation_matrix(df, numerical_cols)
    
    # Heart disease specific analysis
    print("🩺 Analyzing heart disease patterns...")
    statistics['heart_disease_analysis'] = analyze_heart_disease_patterns(df)
    
    # Missing values analysis
    print("❓ Analyzing missing values...")
    statistics['missing_values'] = analyze_missing_values(df)
    
    # Save statistics to JSON file
    output_file = 'dataset_statistics.json'
    with open(output_file, 'w') as f:
        json.dump(statistics, f, indent=2, default=str)
    
    print(f"✅ Statistics generated and saved to {output_file}")
    print("\n" + "="*50)
    print("STATISTICS SUMMARY")
    print("="*50)
    
    # Print summary
    print_statistics_summary(statistics)
    
    return statistics

def generate_overview_stats(df):
    """Generate general overview statistics"""
    
    total_records = len(df)
    heart_disease_count = len(df[df['Heart Disease Status'] == 'Yes']) if 'Heart Disease Status' in df.columns else 0
    heart_disease_rate = round((heart_disease_count / total_records) * 100, 1)
    
    # Calculate average age
    avg_age = round(df['Age'].mean(), 1) if 'Age' in df.columns else 0
    
    return {
        'total_records': total_records,
        'total_attributes': len(df.columns),
        'heart_disease_count': heart_disease_count,
        'heart_disease_rate': heart_disease_rate,
        'no_heart_disease_count': total_records - heart_disease_count,
        'average_age': avg_age,
        'age_range': f"{df['Age'].min():.0f} - {df['Age'].max():.0f}" if 'Age' in df.columns else "N/A"
    }

def analyze_numerical_attribute(df, column):
    """Analyze a numerical attribute and return comprehensive statistics"""
    
    data = df[column].dropna()
    
    # Basic statistics
    statistics = {
        'count': len(data),
        'mean': float(data.mean()),
        'median': float(data.median()),
        'std': float(data.std()),
        'min': float(data.min()),
        'max': float(data.max()),
        'q1': float(data.quantile(0.25)),
        'q3': float(data.quantile(0.75)),
        'iqr': float(data.quantile(0.75) - data.quantile(0.25)),
        'skewness': float(stats.skew(data)),
        'kurtosis': float(stats.kurtosis(data))
    }
    
    # Create histogram data
    hist, bin_edges = np.histogram(data, bins=20)
    
    # Outlier detection using IQR method
    q1, q3 = statistics['q1'], statistics['q3']
    iqr = statistics['iqr']
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = data[(data < lower_bound) | (data > upper_bound)]
    
    # Heart disease correlation
    heart_disease_correlation = {}
    if 'Heart Disease Status' in df.columns:
        yes_group = df[df['Heart Disease Status'] == 'Yes'][column].dropna()
        no_group = df[df['Heart Disease Status'] == 'No'][column].dropna()
        
        if len(yes_group) > 0 and len(no_group) > 0:
            # Perform t-test
            t_stat, p_value = stats.ttest_ind(yes_group, no_group)
            
            heart_disease_correlation = {
                'yes_mean': float(yes_group.mean()),
                'no_mean': float(no_group.mean()),
                'yes_std': float(yes_group.std()),
                'no_std': float(no_group.std()),
                't_statistic': float(t_stat),
                'p_value': float(p_value),
                'significant': p_value < 0.05
            }
    
    return {
        'statistics': statistics,
        'histogram': hist.tolist(),
        'bins': bin_edges.tolist(),
        'outliers_count': len(outliers),
        'outliers_percentage': round((len(outliers) / len(data)) * 100, 2),
        'heart_disease_correlation': heart_disease_correlation
    }

def analyze_categorical_attribute(df, column):
    """Analyze a categorical attribute and return comprehensive statistics"""
    
    # Value counts
    value_counts = df[column].value_counts().to_dict()
    
    # Percentages
    total_count = df[column].count()
    value_percentages = {k: round((v / total_count) * 100, 1) for k, v in value_counts.items()}
    
    # Unique values
    unique_values = df[column].nunique()
    
    # Mode
    mode_value = df[column].mode().iloc[0] if len(df[column].mode()) > 0 else None
    
    # Heart disease breakdown
    heart_disease_breakdown = {}
    if 'Heart Disease Status' in df.columns:
        crosstab = pd.crosstab(df[column], df['Heart Disease Status'])
        
        for category in value_counts.keys():
            if category in crosstab.index:
                heart_disease_breakdown[category] = {
                    'Yes': int(crosstab.loc[category, 'Yes']) if 'Yes' in crosstab.columns else 0,
                    'No': int(crosstab.loc[category, 'No']) if 'No' in crosstab.columns else 0
                }
                
                # Calculate percentages within each category
                total_in_category = sum(heart_disease_breakdown[category].values())
                if total_in_category > 0:
                    heart_disease_breakdown[category]['yes_percentage'] = round(
                        (heart_disease_breakdown[category]['Yes'] / total_in_category) * 100, 1
                    )
    
    return {
        'unique_values': unique_values,
        'mode': mode_value,
        'value_counts': value_counts,
        'value_percentages': value_percentages,
        'heart_disease_breakdown': heart_disease_breakdown,
        'missing_count': df[column].isnull().sum(),
        'missing_percentage': round((df[column].isnull().sum() / len(df)) * 100, 2)
    }

def generate_correlation_matrix(df, numerical_cols):
    """Generate correlation matrix for numerical features"""
    
    # Select only numerical columns that exist in the dataframe
    available_cols = [col for col in numerical_cols if col in df.columns]
    
    if len(available_cols) < 2:
        return {}
    
    # Calculate correlation matrix
    corr_matrix = df[available_cols].corr()
    
    # Convert to nested dictionary format
    correlation_dict = {}
    for col1 in available_cols:
        correlation_dict[col1] = {}
        for col2 in available_cols:
            correlation_dict[col1][col2] = float(corr_matrix.loc[col1, col2])
    
    return correlation_dict

def analyze_heart_disease_patterns(df):
    """Analyze patterns specific to heart disease"""
    
    if 'Heart Disease Status' not in df.columns:
        return {}
    
    # Age group analysis
    age_groups = pd.cut(df['Age'], bins=[0, 30, 40, 50, 60, 70, 80, 100], 
                       labels=['<30', '30-39', '40-49', '50-59', '60-69', '70-79', '80+'])
    
    age_heart_disease = pd.crosstab(age_groups, df['Heart Disease Status'], normalize='index') * 100
    
    # Risk factors analysis
    risk_factors = {}
    
    # Analyze key risk factors
    risk_columns = ['Age', 'Blood Pressure', 'Cholesterol Level', 'BMI']
    
    for col in risk_columns:
        if col in df.columns:
            # Create risk level bins
            if col == 'Age':
                bins = [0, 40, 55, 65, 100]
                labels = ['Low', 'Moderate', 'High', 'Very High']
            elif col == 'Blood Pressure':
                bins = [0, 120, 130, 140, 300]
                labels = ['Normal', 'Elevated', 'High', 'Very High']
            elif col == 'Cholesterol Level':
                bins = [0, 200, 240, 280, 500]
                labels = ['Desirable', 'Borderline', 'High', 'Very High']
            elif col == 'BMI':
                bins = [0, 18.5, 25, 30, 50]
                labels = ['Underweight', 'Normal', 'Overweight', 'Obese']
            else:
                continue
            
            risk_levels = pd.cut(df[col], bins=bins, labels=labels, include_lowest=True)
            risk_crosstab = pd.crosstab(risk_levels, df['Heart Disease Status'], normalize='index') * 100
            
            risk_factors[col] = {
                'risk_levels': risk_crosstab.to_dict() if not risk_crosstab.empty else {},
                'high_risk_threshold': bins[-2] if len(bins) > 2 else None
            }
    
    return {
        'age_group_analysis': age_heart_disease.to_dict() if not age_heart_disease.empty else {},
        'risk_factors': risk_factors,
        'total_high_risk_patients': len(df[(df['Age'] > 60) & (df['Blood Pressure'] > 140)]) if 'Age' in df.columns and 'Blood Pressure' in df.columns else 0
    }

def analyze_missing_values(df):
    """Analyze missing values across all attributes"""
    
    missing_stats = {}
    
    for column in df.columns:
        missing_count = df[column].isnull().sum()
        missing_percentage = (missing_count / len(df)) * 100
        
        missing_stats[column] = {
            'missing_count': int(missing_count),
            'missing_percentage': round(missing_percentage, 2),
            'data_type': str(df[column].dtype),
            'unique_values': int(df[column].nunique()) if missing_count < len(df) else 0
        }
    
    # Overall missing data summary
    total_missing = sum([stats['missing_count'] for stats in missing_stats.values()])
    total_cells = len(df) * len(df.columns)
    overall_missing_percentage = (total_missing / total_cells) * 100
    
    return {
        'by_attribute': missing_stats,
        'overall_summary': {
            'total_missing_values': total_missing,
            'total_cells': total_cells,
            'overall_missing_percentage': round(overall_missing_percentage, 2),
            'attributes_with_missing': len([col for col, stats in missing_stats.items() if stats['missing_count'] > 0])
        }
    }

def print_statistics_summary(statistics):
    """Print a formatted summary of the statistics"""
    
    overview = statistics['overview']
    print(f"📋 Total Records: {overview['total_records']:,}")
    print(f"🏥 Heart Disease Rate: {overview['heart_disease_rate']}%")
    print(f"📊 Total Attributes: {overview['total_attributes']}")
    print(f"🎯 Average Age: {overview['average_age']} years")
    
    print(f"\n📈 Numerical Attributes: {len(statistics['numerical_attributes'])}")
    for attr in list(statistics['numerical_attributes'].keys())[:5]:
        stats_data = statistics['numerical_attributes'][attr]['statistics']
        print(f"   • {attr}: Mean={stats_data['mean']:.1f}, Std={stats_data['std']:.1f}")
    
    print(f"\n📋 Categorical Attributes: {len(statistics['categorical_attributes'])}")
    for attr in list(statistics['categorical_attributes'].keys())[:5]:
        unique_vals = statistics['categorical_attributes'][attr]['unique_values']
        print(f"   • {attr}: {unique_vals} unique values")
    
    missing_summary = statistics['missing_values']['overall_summary']
    print(f"\n❓ Missing Data: {missing_summary['overall_missing_percentage']:.1f}% overall")
    
    print(f"\n✅ Complete statistics available in 'dataset_statistics.json'")

if __name__ == "__main__":
    try:
        stats = generate_comprehensive_statistics()
        print("\n🎉 Statistics generation completed successfully!")
    except Exception as e:
        print(f"❌ Error generating statistics: {str(e)}")
        import traceback
        traceback.print_exc()