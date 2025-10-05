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

def initialize_model():
    """Initialize or load the heart disease prediction model."""
    global predictor
    
    predictor = EnhancedHeartDiseasePredictor()
    
    # Try to load the reduced model first (best performance)
    if not predictor.load_model('heart_disease_model_reduced.pkl'):
        logger.info("Reduced model not found. Trying FIXED model...")
        if not predictor.load_model('heart_disease_model_FIXED.pkl'):
            logger.info("FIXED model not found. Using original model...")
            if not predictor.load_model('heart_disease_model.pkl'):
                logger.error("No model found! Please train a model first.")
                raise Exception("No trained model available")
            else:
                logger.info("Original model loaded successfully!")
        else:
            logger.info("FIXED model loaded successfully!")
    else:
        logger.info("Reduced model loaded successfully! (35% fewer features, 97.1% accuracy)")

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
    API endpoint for heart disease prediction.
    
    Expected JSON input:
    {
        "age": 55,
        "gender": "Male",
        "blood_pressure": 140,
        "cholesterol_level": 200,
        "smoking": "No",
        "bmi": 28.5,
        "exercise_habits": "Medium",
        "family_heart_disease": "Yes",
        "diabetes": "No",
        "high_blood_pressure": "Yes",
        "stress_level": "Medium",
        "sleep_hours": 7.0,
        ...
    }
    """
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if not data:
            return jsonify({
                'error': 'No data provided',
                'success': False
            }), 400
        
        # Validate required fields
        required_fields = ['age', 'gender', 'blood_pressure', 'cholesterol_level', 'smoking', 'bmi']
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            return jsonify({
                'error': f'Missing required fields: {missing_fields}',
                'success': False
            }), 400
        
        # Convert input data to model format
        model_input = {
            'Age': float(data.get('age')),
            'Gender': str(data.get('gender')),
            'Blood Pressure': float(data.get('blood_pressure')),
            'Cholesterol Level': float(data.get('cholesterol_level')),
            'Exercise Habits': str(data.get('exercise_habits', 'Medium')),
            'Smoking': str(data.get('smoking')),
            'Family Heart Disease': str(data.get('family_heart_disease', 'No')),
            'Diabetes': str(data.get('diabetes', 'No')),
            'BMI': float(data.get('bmi')),
            'High Blood Pressure': str(data.get('high_blood_pressure', 'No')),
            'Low HDL Cholesterol': str(data.get('low_hdl_cholesterol', 'No')),
            'High LDL Cholesterol': str(data.get('high_ldl_cholesterol', 'No')),
            'Alcohol Consumption': str(data.get('alcohol_consumption', 'Low')),
            'Stress Level': str(data.get('stress_level', 'Medium')),
            'Sleep Hours': float(data.get('sleep_hours', 7.0)),
            'Sugar Consumption': str(data.get('sugar_consumption', 'Medium')),
            'Triglyceride Level': float(data.get('triglyceride_level', 150.0)),
            'Fasting Blood Sugar': float(data.get('fasting_blood_sugar', 100.0)),
            'CRP Level': float(data.get('crp_level', 2.0)),
            'Homocysteine Level': float(data.get('homocysteine_level', 10.0))
        }
        
        # Make prediction
        result = predictor.predict(model_input)
        
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
        return jsonify({
            'error': f'Prediction failed: {str(e)}',
            'success': False
        }), 500

@app.route('/api/model-info', methods=['GET'])
def get_model_info():
    """Get information about the trained model."""
    try:
        if predictor is None or predictor.model is None:
            return jsonify({
                'error': 'Model not loaded',
                'success': False
            }), 500
        
        # Get feature importance if available
        feature_importance = predictor.get_feature_importance()
        importance_data = None
        
        if feature_importance is not None:
            importance_data = feature_importance.head(10).to_dict('records')
        
        model_info = {
            'success': True,
            'model_type': type(predictor.model).__name__,
            'feature_count': len(predictor.feature_columns) if predictor.feature_columns else 0,
            'model_metrics': predictor.model_metrics,
            'feature_importance': importance_data
        }
        
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