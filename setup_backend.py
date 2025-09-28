#!/usr/bin/env python3
"""
Backend Setup and Verification Script
Automates the setup process for the Heart Disease Prediction backend
"""

import subprocess
import sys
import os
import importlib
from pathlib import Path

def run_command(command, description):
    """Run a command and return the result"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"   ✅ Success: {description}")
            return True
        else:
            print(f"   ❌ Failed: {description}")
            print(f"   Error: {result.stderr}")
            return False
    except Exception as e:
        print(f"   ❌ Exception: {str(e)}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"   ✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
        return True
    else:
        print(f"   ❌ Python {version.major}.{version.minor}.{version.micro} is too old. Need Python 3.8+")
        return False

def check_virtual_environment():
    """Check if virtual environment exists and is activated"""
    print("📦 Checking virtual environment...")
    
    # Check if .venv directory exists
    venv_path = Path(".venv")
    if not venv_path.exists():
        print("   ⚠️  Virtual environment not found. Creating...")
        return run_command("python -m venv .venv", "Creating virtual environment")
    else:
        print("   ✅ Virtual environment directory found")
    
    # Check if we're in a virtual environment
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("   ✅ Virtual environment is activated")
        return True
    else:
        print("   ⚠️  Virtual environment exists but not activated")
        print("   💡 Please run: .venv\\Scripts\\Activate.ps1")
        return False

def check_required_packages():
    """Check if required packages are installed"""
    print("📚 Checking required packages...")
    
    required_packages = [
        'flask', 'flask_cors', 'sklearn', 'pandas', 'numpy', 
        'joblib', 'matplotlib', 'seaborn'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            # Handle special package names
            if package == 'sklearn':
                importlib.import_module('sklearn')
            elif package == 'flask_cors':
                importlib.import_module('flask_cors')
            else:
                importlib.import_module(package)
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package} - MISSING")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"   ⚠️  Missing packages: {', '.join(missing_packages)}")
        print("   💡 Run: pip install -r requirements.txt")
        return False
    else:
        print("   ✅ All required packages are installed")
        return True

def check_project_files():
    """Check if required project files exist"""
    print("📁 Checking project files...")
    
    required_files = [
        'app.py',
        'simple_model_wrapper.py',
        'templates/index.html',
        'static/css/style.css',
        'static/js/app.js',
        'heart_disease_extended.csv'
    ]
    
    missing_files = []
    
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"   ✅ {file_path}")
        else:
            print(f"   ❌ {file_path} - MISSING")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"   ⚠️  Missing files: {', '.join(missing_files)}")
        return False
    else:
        print("   ✅ All required project files found")
        return True

def check_model_file():
    """Check if model file exists"""
    print("🤖 Checking model file...")
    
    model_files = ['heart_disease_model.pkl']
    
    for model_file in model_files:
        if Path(model_file).exists():
            print(f"   ✅ {model_file} found")
            return True
    
    print("   ⚠️  No model file found")
    print("   💡 Run: python train_heart_disease_model.py")
    return False

def test_flask_app():
    """Test if Flask app can be imported and initialized"""
    print("🌐 Testing Flask app...")
    
    try:
        # Try to import the app
        sys.path.insert(0, '.')
        from app import app
        print("   ✅ Flask app imports successfully")
        
        # Test app configuration
        with app.app_context():
            print("   ✅ Flask app context works")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Flask app test failed: {str(e)}")
        return False

def generate_setup_report(checks):
    """Generate a setup report"""
    print("\n📊 SETUP REPORT")
    print("=" * 50)
    
    total_checks = len(checks)
    passed_checks = sum(1 for result in checks.values() if result)
    
    for check_name, result in checks.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {check_name}")
    
    print(f"\nScore: {passed_checks}/{total_checks} checks passed")
    
    if passed_checks == total_checks:
        print("🎉 SETUP COMPLETE - Backend is ready!")
        print("\n🚀 Next steps:")
        print("1. Run: python app.py")
        print("2. Open: http://localhost:5000")
    else:
        print("⚠️  SETUP INCOMPLETE - Please fix the failed checks")
        print("\n💡 Quick fixes:")
        if not checks.get("Virtual Environment", True):
            print("- Activate virtual environment: .venv\\Scripts\\Activate.ps1")
        if not checks.get("Required Packages", True):
            print("- Install packages: pip install -r requirements.txt")
        if not checks.get("Model File", True):
            print("- Train model: python train_heart_disease_model.py")

def main():
    """Main setup verification function"""
    print("🔧 BACKEND ENVIRONMENT SETUP CHECKER")
    print("=" * 50)
    print("Verifying Heart Disease Prediction backend environment...\n")
    
    # Run all checks
    checks = {
        "Python Version": check_python_version(),
        "Virtual Environment": check_virtual_environment(),
        "Required Packages": check_required_packages(),
        "Project Files": check_project_files(),
        "Model File": check_model_file(),
        "Flask App": test_flask_app()
    }
    
    # Generate report
    generate_setup_report(checks)
    
    return all(checks.values())

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
    else:
        print("\n✨ Backend environment is properly configured!")