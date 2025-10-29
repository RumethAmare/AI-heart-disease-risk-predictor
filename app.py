#!/usr/bin/env python3
"""
Heart Disease Prediction API - Flask Backend
FDM Mini Project 2025 - Web Application Backend
"""

from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import os
import json
from enhanced_model_wrapper import EnhancedHeartDiseasePredictor
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app with template and static folders
app = Flask(__name__, template_folder='frontend', static_folder='frontend')
CORS(app)

# Global predictor instance
predictor = None

# Initialize model immediately for deployment compatibility
def init_model_on_startup():
    """Initialize model when module is imported (for deployment)."""
    global predictor
    try:
        initialize_model()
        logger.info("Model initialized during module import")
    except Exception as e:
        logger.error(f"Failed to initialize model on startup: {e}")
        # Create a minimal predictor as absolute fallback
        logger.info("Creating emergency fallback predictor...")
        predictor = EnhancedHeartDiseasePredictor()
        if not predictor.create_basic_model():
            logger.error("Emergency fallback creation failed")
            # Create the most basic predictor possible
            predictor.model_data = {'is_rule_based': True, 'model': 'rule_based'}
            predictor.is_loaded = True
        logger.info("Emergency fallback predictor created")

def initialize_model():
    """Initialize or load the heart disease prediction model with production fallback."""
    global predictor
    
    predictor = EnhancedHeartDiseasePredictor()
    
    # Check if any model files exist (prioritize Render-optimized model)
    model_files = [
        'heart_disease_render_optimized.pkl',  # New optimized model for Render
        'heart_disease_model_with_gender.pkl',
        'heart_disease_model_reduced.pkl', 
        'heart_disease_model_FIXED.pkl',
        'heart_disease_model.pkl'
    ]
    
    available_models = [f for f in model_files if os.path.exists(f)]
    
    if not available_models:
        logger.warning("No pre-trained model files found in production environment!")
        logger.info("Initializing with basic fallback model for Render deployment...")
        # Create a basic fallback for production deployment
        try:
            predictor.create_basic_model()
            logger.info("Basic fallback model created successfully for production!")
            return
        except Exception as e:
            logger.error(f"Failed to create fallback model: {e}")
            raise Exception("No trained model available and fallback creation failed")
    
    # Try to load the Render-optimized model first
    if not predictor.load_model('heart_disease_render_optimized.pkl'):
        logger.info("Render-optimized model not found. Trying model with gender...")
        if not predictor.load_model('heart_disease_model_with_gender.pkl'):
            logger.info("Model with gender not found. Trying reduced model...")
            if not predictor.load_model('heart_disease_model_reduced.pkl'):
                logger.info("Reduced model not found. Trying FIXED model...")
                if not predictor.load_model('heart_disease_model_FIXED.pkl'):
                    logger.info("FIXED model not found. Using original model...")
                    if not predictor.load_model('heart_disease_model.pkl'):
                        logger.error("No model could be loaded!")
                        raise Exception("No trained model available")
                    else:
                        logger.info("Original model loaded successfully!")
                else:
                    logger.info("FIXED model loaded successfully!")
            else:
                logger.info("Reduced model loaded successfully! (35% fewer features, 97.1% accuracy)")
        else:
            logger.info("Model with gender loaded successfully! (14 features, optimized performance)")
    else:
        logger.info("🚀 Render-optimized model loaded successfully! (Production-grade ML predictions)")
        
        # Validate model signature if available
        if hasattr(predictor, 'model_data') and predictor.model_data and 'model_signature' in predictor.model_data:
            signature = predictor.model_data['model_signature']
            logger.info(f"📋 Model Details: Version {signature.get('model_version', 'Unknown')}, "
                       f"Accuracy {signature.get('accuracy', 'Unknown'):.4f}, "
                       f"Features {signature.get('n_features', 'Unknown')}")

@app.route('/')
@app.route('/index.html')
def index():
    """Serve the main frontend page."""
    try:
        logger.info("Serving index.html page")
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Error serving index.html: {str(e)}")
        return f"Error loading index.html: {str(e)}", 500

@app.route('/statistics')
@app.route('/statistics.html')
def statistics_page():
    """Serve the statistics dashboard page."""
    try:
        logger.info("Serving statistics.html page")
        return render_template('statistics.html')
    except Exception as e:
        logger.error(f"Error serving statistics.html: {str(e)}")
        return f"Error loading statistics.html: {str(e)}", 500

@app.route('/assess')
@app.route('/assess.html')
@app.route('/predict')
@app.route('/predict.html')
def assessment_page():
    """Serve the heart disease risk assessment page."""
    try:
        logger.info("Serving assess.html page")
        return render_template('assess.html')
    except Exception as e:
        logger.error(f"Error serving assess.html: {str(e)}")
        return f"Error loading assess.html: {str(e)}", 500

@app.route('/debug/routes')
def debug_routes():
    """Debug endpoint to show all available routes."""
    routes = []
    for rule in app.url_map.iter_rules():
        routes.append({
            'endpoint': rule.endpoint,
            'methods': list(rule.methods),
            'rule': rule.rule
        })
    return jsonify({'routes': routes})

@app.route('/static/<path:filename>')
def static_files(filename):
    """Serve static files."""
    return send_from_directory('.', filename)

@app.route('/api/predict', methods=['POST'])
def predict_heart_disease():
    """
    API endpoint for heart disease prediction (Reduced Model - 14 features with Gender).
    
    Expected JSON input:
    {
        "age": 55,
        "gender": "Male",
        "blood_pressure": 140,
        "cholesterol_level": 200,
        "bmi": 28.5,
        "exercise_habits": "Medium",
        "alcohol_consumption": "None",
        "stress_level": "Medium",
        "sleep_hours": 7.0,
        "sugar_consumption": "Medium",
        "triglyceride_level": 150.0,
        "fasting_blood_sugar": 90.0,
        "crp_level": 1.0,
        "homocysteine_level": 10.0
    }
    
    Note: Dropped columns (not needed): Smoking, Diabetes, 
    High Blood Pressure, Family Heart Disease, Low HDL Cholesterol, 
    High LDL Cholesterol (6 columns dropped, Gender retained)
    """
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if not data:
            return jsonify({
                'error': 'No data provided',
                'success': False
            }), 400
        
        # Validate required fields (updated for reduced model with Gender)
        required_fields = ['age', 'gender', 'blood_pressure', 'cholesterol_level', 'bmi', 'exercise_habits']
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            return jsonify({
                'error': f'Missing required fields: {missing_fields}',
                'success': False
            }), 400
        
        # Convert input data to model format (reduced model - 14 features with Gender)
        model_input = {
            'Age': float(data.get('age')),
            'Gender': str(data.get('gender')),
            'Blood Pressure': float(data.get('blood_pressure')),
            'Cholesterol Level': float(data.get('cholesterol_level')),
            'Exercise Habits': str(data.get('exercise_habits', 'Medium')),
            'BMI': float(data.get('bmi')),
            'Alcohol Consumption': str(data.get('alcohol_consumption', 'None')),
            'Stress Level': str(data.get('stress_level', 'Medium')),
            'Sleep Hours': float(data.get('sleep_hours', 7.0)),
            'Sugar Consumption': str(data.get('sugar_consumption', 'Medium')),
            'Triglyceride Level': float(data.get('triglyceride_level', 150.0)),
            'Fasting Blood Sugar': float(data.get('fasting_blood_sugar', 90.0)),
            'CRP Level': float(data.get('crp_level', 1.0)),
            'Homocysteine Level': float(data.get('homocysteine_level', 10.0))
        }
        
        # Ensure predictor is loaded
        if predictor is None:
            logger.error("Predictor not initialized, attempting to initialize now")
            try:
                initialize_model()
                if predictor is None:
                    raise Exception("Model initialization failed")
            except Exception as e:
                return jsonify({
                    'error': f'Model not available: {str(e)}',
                    'success': False
                }), 500
        
        # Check if predictor is loaded
        if not getattr(predictor, 'is_loaded', False):
            logger.error("Predictor exists but not loaded, forcing initialization")
            try:
                if not predictor.create_basic_model():
                    # Force basic rule-based model
                    predictor.model_data = {'is_rule_based': True, 'model': 'rule_based'}
                    predictor.is_loaded = True
                logger.info("Forced predictor initialization successful")
            except Exception as e:
                logger.error(f"Forced initialization failed: {e}")
                return jsonify({
                    'error': 'Model could not be initialized',
                    'success': False
                }), 500
        
        # Double-check predictor is available before prediction
        if predictor is None or not hasattr(predictor, 'predict') or not predictor.is_loaded:
            return jsonify({
                'error': 'Model predictor is not properly initialized',
                'success': False
            }), 500
        
        # Make prediction with error handling
        try:
            logger.info(f"About to make prediction with predictor: {type(predictor)}, is_loaded: {getattr(predictor, 'is_loaded', 'Unknown')}")
            result = predictor.predict(model_input)
            if result is None:
                raise Exception("Prediction returned None")
            logger.info(f"Prediction successful: {result.get('prediction', 'Unknown')}")
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            return jsonify({
                'error': f'Prediction failed: {str(e)}',
                'success': False
            }), 500
        
        # Add risk interpretation
        risk_prob = result['risk_probability']
        if risk_prob < 0.3:
            risk_level = 'Low'
            risk_color = 'green'
            recommendation = 'Continue maintaining a healthy lifestyle.'
        elif risk_prob < 0.6:
            risk_level = 'Medium'
            risk_color = 'orange'
            recommendation = 'Consider lifestyle modifications and regular check-ups.'
        else:
            risk_level = 'High'
            risk_color = 'red'
            recommendation = 'Consult with a healthcare professional immediately.'
        
        # Return prediction results
        response = {
            'success': True,
            'prediction': result['prediction'],
            'risk_probability': result['risk_probability'],
            'risk_percentage': result['risk_percentage'],
            'confidence': result['confidence'],
            'risk_level': risk_level,
            'risk_color': risk_color,
            'recommendation': recommendation,
            'input_data': model_input
        }
        
        logger.info(f"Prediction made: {result['prediction']} ({result['risk_percentage']} risk)")
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        logger.info("Attempting emergency rule-based prediction...")
        
        # Ultimate fallback - create a basic prediction manually
        try:
            data = request.get_json() or {}
            age = float(data.get('age', 50))
            gender = str(data.get('gender', 'Male'))
            bp = float(data.get('blood_pressure', 120))
            chol = float(data.get('cholesterol_level', 200))
            bmi = float(data.get('bmi', 25))
            
            # Simple rule-based calculation
            risk_score = 0
            if age > 65: risk_score += 25
            elif age > 55: risk_score += 15
            elif age > 45: risk_score += 10
            
            if gender == 'Male': risk_score += 10
            if bp > 140: risk_score += 15
            if chol > 240: risk_score += 10
            if bmi > 30: risk_score += 10
            
            risk_prob = min(risk_score / 100.0, 0.95)
            prediction = "Yes" if risk_prob > 0.5 else "No"
            
            return jsonify({
                'success': True,
                'prediction': prediction,
                'risk_probability': risk_prob,
                'risk_percentage': f"{risk_prob * 100:.1f}%",
                'confidence': "75.0%",
                'risk_level': 'High' if risk_prob > 0.6 else ('Medium' if risk_prob > 0.3 else 'Low'),
                'risk_color': '#dc3545' if risk_prob > 0.6 else ('#ffc107' if risk_prob > 0.3 else '#28a745'),
                'recommendation': 'Emergency prediction - please consult healthcare professional',
                'model_type': 'emergency_fallback'
            })
            
        except Exception as fallback_error:
            logger.error(f"Emergency fallback also failed: {str(fallback_error)}")
            return jsonify({
                'error': f'All prediction methods failed: {str(e)}',
                'success': False
            }), 500

@app.route('/api/model-info', methods=['GET'])
def get_model_info():
    """Get detailed information about the trained model."""
    try:
        if predictor is None:
            return jsonify({
                'error': 'Predictor not initialized',
                'success': False
            }), 500
        
        # Basic model info
        model_info = {
            'success': True,
            'is_loaded': predictor.is_loaded,
            'has_model_data': predictor.model_data is not None
        }
        
        if predictor.model_data:
            # Get model signature if available
            if 'model_signature' in predictor.model_data:
                signature = predictor.model_data['model_signature']
                model_info.update({
                    'model_signature': signature,
                    'model_version': signature.get('model_version', 'Unknown'),
                    'accuracy': signature.get('accuracy', 'Unknown'),
                    'n_features': signature.get('n_features', 'Unknown'),
                    'training_date': signature.get('training_date', 'Unknown'),
                    'feature_order': signature.get('feature_order', [])
                })
            
            # Get model info
            if 'model_info' in predictor.model_data:
                info = predictor.model_data['model_info']
                model_info.update({
                    'model_type': info.get('model_type', 'Unknown'),
                    'optimized_for': info.get('optimized_for', 'Unknown'),
                    'version': info.get('version', 'Unknown')
                })
            
            # Get feature information
            if 'feature_columns' in predictor.model_data:
                model_info['feature_columns'] = predictor.model_data['feature_columns']
                model_info['feature_count'] = len(predictor.model_data['feature_columns'])
            
            # Check if rule-based fallback
            if predictor.model_data.get('is_rule_based', False):
                model_info['prediction_method'] = 'rule_based_fallback'
            else:
                model_info['prediction_method'] = 'machine_learning'
        
        return jsonify(model_info)
        
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        return jsonify({
            'error': f'Failed to get model info: {str(e)}',
            'success': False
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': predictor is not None and predictor.is_loaded,
        'success': True
    })

@app.route('/api/statistics', methods=['GET'])
def get_statistics():
    """Return comprehensive dataset statistics for visualization"""
    try:
        import json
        import os
        import subprocess
        
        # Load statistics from JSON file
        stats_file = 'dataset_statistics.json'
        
        if not os.path.exists(stats_file):
            logger.info("Statistics file not found, generating new statistics...")
            # Generate statistics if file doesn't exist
            result = subprocess.run(['./.venv/Scripts/python.exe', 'generate_dataset_statistics.py'], 
                                  capture_output=True, text=True, shell=True, cwd='.')
            
            if result.returncode != 0:
                logger.error(f"Failed to generate statistics: {result.stderr}")
                return jsonify({
                    'error': 'Failed to generate statistics',
                    'details': result.stderr,
                    'success': False
                }), 500
        
        # Load and return statistics
        with open(stats_file, 'r') as f:
            statistics = json.load(f)
        
        logger.info("Statistics loaded successfully")
        return jsonify({
            'success': True,
            'data': statistics
        })
        
    except Exception as e:
        logger.error(f"Error loading statistics: {str(e)}")
        return jsonify({
            'error': 'Failed to load statistics',
            'details': str(e),
            'success': False
        }), 500

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({
        'error': 'Endpoint not found',
        'success': False
    }), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    logger.error(f"Internal server error: {str(error)}")
    return jsonify({
        'error': 'Internal server error',
        'success': False
    }), 500

def create_app():
    """Create and configure the Flask application."""
    
    # Create templates and static directories if they don't exist
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static/css', exist_ok=True)
    os.makedirs('static/js', exist_ok=True)
    
    # Initialize the model
    try:
        initialize_model()
        logger.info("Flask app initialized successfully!")
    except Exception as e:
        logger.error(f"Failed to initialize model: {str(e)}")
        raise e
    
    return app

# Initialize model for deployment compatibility
init_model_on_startup()

if __name__ == '__main__':
    # Create the application
    app = create_app()
    
    # Run the Flask development server
    print("🚀 Starting Heart Disease Prediction API...")
    print("📱 Frontend will be available at: http://localhost:5000")
    print("🔗 API endpoints:")
    print("   - POST /api/predict - Make heart disease predictions")
    print("   - GET /api/model-info - Get model information")
    print("   - GET /api/health - Health check")
    
    app.run(debug=True, host='0.0.0.0', port=5000)