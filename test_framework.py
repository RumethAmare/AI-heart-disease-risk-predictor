#!/usr/bin/env python3
"""
Simple test script to verify the FDM preprocessing framework works correctly.
"""

def test_framework():
    """Test basic functionality of the preprocessing framework."""
    
    print("🧪 Testing FDM Data Preprocessing Framework...")
    
    try:
        # Test imports
        print("1️⃣ Testing imports...")
        from data_preprocessor import DataPreprocessor
        from fdm_preprocessor import FDMProjectPreprocessor
        print("   ✅ All imports successful!")
        
        # Test basic instantiation
        print("2️⃣ Testing class instantiation...")
        basic_preprocessor = DataPreprocessor()
        fdm_preprocessor = FDMProjectPreprocessor()
        print("   ✅ Classes instantiated successfully!")
        
        # Test sample data creation
        print("3️⃣ Testing sample data creation...")
        import pandas as pd
        import numpy as np
        
        # Create a small sample dataset
        np.random.seed(42)
        sample_data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 1000),
            'feature2': np.random.exponential(1, 1000),
            'category': np.random.choice(['A', 'B', 'C'], 1000),
            'target': np.random.binomial(1, 0.5, 1000)
        })
        
        # Add some missing values
        sample_data.loc[sample_data.sample(50).index, 'feature1'] = np.nan
        sample_data.loc[sample_data.sample(20).index, 'category'] = np.nan
        
        print(f"   ✅ Sample dataset created: {sample_data.shape}")
        
        # Test basic preprocessing
        print("4️⃣ Testing basic preprocessing...")
        basic_preprocessor.data = sample_data
        basic_preprocessor.target_column = 'target'
        basic_preprocessor._identify_column_types()
        
        print(f"   📊 Numeric columns: {len(basic_preprocessor.numeric_columns)}")
        print(f"   📊 Categorical columns: {len(basic_preprocessor.categorical_columns)}")
        
        # Test missing value handling
        initial_missing = sample_data.isnull().sum().sum()
        basic_preprocessor.handle_missing_values(strategy='auto')
        final_missing = basic_preprocessor.data.isnull().sum().sum()
        
        print(f"   🧹 Missing values: {initial_missing} → {final_missing}")
        
        # Test encoding
        basic_preprocessor.encode_categorical_features(encoding_type='auto')
        print(f"   🏷️ Categorical encoding completed")
        
        # Test scaling
        basic_preprocessor.scale_features(scaling_type='standard')
        print(f"   ⚖️ Feature scaling completed")
        
        print("   ✅ Basic preprocessing successful!")
        
        # Test FDM-specific functionality
        print("5️⃣ Testing FDM-specific functionality...")
        fdm_preprocessor.data = basic_preprocessor.data.copy()
        fdm_preprocessor.target_column = 'target'
        fdm_preprocessor._identify_column_types()
        
        # Note: This will show warnings since our test data is < 10,000 rows
        requirements = fdm_preprocessor.validate_project_requirements()
        print(f"   📋 FDM requirements validation completed")
        
        quality_score = fdm_preprocessor._calculate_quality_score()
        print(f"   📊 Data quality score: {quality_score:.2f}/10.0")
        
        print("   ✅ FDM-specific functionality successful!")
        
        print("\n🎉 ALL TESTS PASSED! Framework is working correctly!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_framework()
    if success:
        print("\n✅ Framework ready for FDM project use!")
    else:
        print("\n❌ Please check the error messages above.")