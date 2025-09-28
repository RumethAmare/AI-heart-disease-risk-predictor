#!/usr/bin/env python3
"""
Heart Disease Prediction System Startup Script
FDM Mini Project 2025 - Complete System Launcher
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def check_requirements():
    """Check if all required files and packages are available."""
    
    print("🔍 Checking system requirements...")
    
    # Check required files
    required_files = [
        'heart_disease.csv',
        'heart_disease_model.py',
        'app.py',
        'templates/index.html',
        'static/css/style.css',
        'static/js/app.js'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    
    # Check if model file exists
    model_exists = os.path.exists('heart_disease_model.pkl')
    
    print("✅ All required files found")
    print(f"📊 Dataset: heart_disease.csv ({os.path.getsize('heart_disease.csv')} bytes)")
    print(f"🤖 Model: {'Found' if model_exists else 'Will be trained'}")
    
    return True

def train_model_if_needed():
    """Train the model if it doesn't exist."""
    
    if os.path.exists('heart_disease_model.pkl'):
        print("✅ Model already exists, skipping training")
        return True
    
    print("🔬 Training heart disease prediction model...")
    print("This may take a few minutes...")
    
    try:
        # Import and run model training
        from heart_disease_model import train_heart_disease_model
        
        print("📚 Loading dataset and preprocessing...")
        predictor = train_heart_disease_model()
        
        print("✅ Model training completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Model training failed: {str(e)}")
        return False

def start_web_application():
    """Start the Flask web application."""
    
    print("\n🚀 Starting Heart Disease Prediction Web Application...")
    print("=" * 60)
    
    try:
        # Import and start Flask app
        from app import create_app
        
        app = create_app()
        
        print("✅ Application initialized successfully!")
        print("\n🌐 Web Application Details:")
        print("   URL: http://localhost:5000")
        print("   Frontend: Modern responsive web interface")
        print("   Backend: Flask API with ML prediction")
        print("   Model: Trained on 10,000+ heart disease records")
        print("\n🎯 Features:")
        print("   • Real-time heart disease risk prediction")
        print("   • Interactive form with validation")
        print("   • Risk assessment with recommendations")
        print("   • Mobile-responsive design")
        print("   • Advanced medical parameters")
        
        print("\n" + "=" * 60)
        print("🏥 HEART DISEASE PREDICTION SYSTEM - READY!")
        print("=" * 60)
        print("\n⚡ Starting server on http://localhost:5000...")
        print("📱 Open your web browser and navigate to the URL above")
        print("🔄 Press Ctrl+C to stop the server\n")
        
        # Start the Flask development server
        app.run(debug=True, host='0.0.0.0', port=5000)
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Server stopped by user")
        print("Thank you for using the Heart Disease Prediction System!")
        
    except Exception as e:
        print(f"\n❌ Failed to start application: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def display_banner():
    """Display application banner."""
    
    banner = """
╔══════════════════════════════════════════════════════════════════╗
║               HEART DISEASE PREDICTION SYSTEM                    ║
║                   FDM Mini Project 2025                         ║
║                                                                  ║
║  🤖 AI-Powered Risk Assessment                                   ║
║  📊 10,000+ Training Data Points                                 ║
║  🎯 Advanced Machine Learning Models                             ║
║  🌐 Modern Web Interface                                         ║
║                                                                  ║
║  Built with: Python, Flask, Scikit-learn, HTML5, CSS3, JS      ║
╚══════════════════════════════════════════════════════════════════╝
"""
    
    print(banner)

def main():
    """Main startup function."""
    
    # Display banner
    display_banner()
    
    # Check requirements
    if not check_requirements():
        print("❌ System requirements not met. Please check the files.")
        sys.exit(1)
    
    # Train model if needed
    if not train_model_if_needed():
        print("❌ Failed to initialize the prediction model.")
        sys.exit(1)
    
    # Start web application
    start_web_application()

if __name__ == "__main__":
    main()