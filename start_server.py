#!/usr/bin/env python3
"""
Heart Disease Prediction API - Startup Script
Run this to start the backend server
"""

import os
import sys
import subprocess

def main():
    """Start the backend server."""
    
    # Change to backend directory
    backend_dir = os.path.join(os.path.dirname(__file__), 'backend')
    os.chdir(backend_dir)
    
    print("🚀 Starting Heart Disease Prediction API...")
    print("📁 Backend directory:", backend_dir)
    print("📱 Frontend will be available at: http://localhost:5000")
    print("🔗 API endpoints:")
    print("   - POST /api/predict - Make heart disease predictions")
    print("   - GET /api/model-info - Get model information")
    print("   - GET /api/health - Health check")
    print()
    
    # Run the backend
    try:
        subprocess.run([sys.executable, "app.py"], check=True)
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")

if __name__ == "__main__":
    main()