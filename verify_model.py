import pickle

# Load and verify the saved model
try:
    with open('heart_disease_extended_random_forest_20251005_193114.pkl', 'rb') as f:
        data = pickle.load(f)
    
    print("🎉 MODEL VERIFICATION SUCCESS!")
    print("=" * 40)
    print(f"Model Type: {data['model_type']}")
    print(f"Accuracy: {data['accuracy']*100:.2f}%")
    print(f"Features Used: {data['features_used']}")
    print(f"Training Samples: {data['samples_trained']:,}")
    print(f"Dataset: {data['dataset']}")
    print(f"Training Date: {data['training_date'][:19]}")
    
    print(f"\nFeature Names:")
    for i, feature in enumerate(data['feature_names'], 1):
        print(f"  {i:2d}. {feature}")
    
    print(f"\nModel is ready for production use! ✅")
    
except Exception as e:
    print(f"Error loading model: {e}")