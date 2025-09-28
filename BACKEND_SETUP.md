# Backend Environment Setup Guide

## 🚀 Heart Disease Prediction Backend Setup

This guide will help you properly set up the backend environment for the Heart Disease Prediction system.

## 📋 Prerequisites

- Python 3.8 or higher
- Windows PowerShell (or Command Prompt)
- Git (optional, for version control)

## 🔧 Step-by-Step Setup

### 1. Clone or Navigate to Project Directory
```powershell
# If cloning from repository
git clone https://github.com/RumethAmare/AI-heart-disease-risk-predictor.git
cd AI-heart-disease-risk-predictor

# Or navigate to existing project
cd "C:\Users\Dineth\Desktop\FDM - reading\FDM\FDM project"
```

### 2. Create Virtual Environment
```powershell
# Create virtual environment
python -m venv .venv

# Activate virtual environment (Windows)
.\.venv\Scripts\Activate.ps1

# If you get execution policy error, run:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### 3. Install Dependencies
```powershell
# Upgrade pip
python -m pip install --upgrade pip

# Install all required packages
pip install -r requirements.txt

# Verify installation
pip list
```

### 4. Verify Project Structure
```
FDM project/
├── .venv/                          # Virtual environment
├── static/                         # Frontend assets
│   ├── css/style.css
│   └── js/app.js
├── templates/                      # HTML templates
│   └── index.html
├── app.py                         # Flask web application
├── simple_model_wrapper.py       # ML model wrapper
├── heart_disease_model.pkl       # Trained model
├── heart_disease_extended.csv    # Dataset
├── requirements.txt              # Dependencies
└── README.md                     # Documentation
```

### 5. Train/Verify Model (if needed)
```powershell
# Train new model with dataset
python train_heart_disease_model.py

# Verify model exists
dir heart_disease_model.pkl
```

### 6. Start the Backend Server
```powershell
# Method 1: Direct execution
python app.py

# Method 2: Flask development server
set FLASK_APP=app.py
set FLASK_ENV=development
flask run

# Method 3: Production-like server (install gunicorn first)
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

## 🌐 Accessing the Application

Once the server starts successfully, you'll see:
```
🚀 Starting Heart Disease Prediction API...
📱 Frontend will be available at: http://localhost:5000
🔗 API endpoints:
   - POST /api/predict - Make heart disease predictions
   - GET /api/model-info - Get model information
   - GET /api/health - Health check
 * Running on http://127.0.0.1:5000
```

Open your browser and navigate to: **http://localhost:5000**

## 🧪 Testing the Setup

### Test API Endpoints
```powershell
# Health check
curl http://localhost:5000/api/health

# Model information
curl http://localhost:5000/api/model-info

# Test prediction (using PowerShell)
$body = @{
    age = 55
    gender = "Male"
    blood_pressure = 140
    cholesterol_level = 200
    smoking = "No"
    bmi = 28.5
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:5000/api/predict" -Method POST -Body $body -ContentType "application/json"
```

## 🐛 Troubleshooting

### Common Issues and Solutions

#### 1. **Virtual Environment Activation Error**
```powershell
# Error: execution of scripts is disabled
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Then try again
.\.venv\Scripts\Activate.ps1
```

#### 2. **Module Not Found Errors**
```powershell
# Ensure virtual environment is activated
.\.venv\Scripts\Activate.ps1

# Reinstall requirements
pip install -r requirements.txt

# Check if packages are installed
pip list | findstr flask
pip list | findstr scikit-learn
```

#### 3. **Model Loading Issues**
```powershell
# Retrain the model
python train_heart_disease_model.py

# Check model file exists
dir *.pkl
```

#### 4. **Port Already in Use**
```powershell
# Find process using port 5000
netstat -ano | findstr :5000

# Kill the process (replace PID with actual process ID)
taskkill /PID <PID> /F

# Or use a different port
set FLASK_RUN_PORT=5001
python app.py
```

#### 5. **CORS Issues (Frontend/Backend on different ports)**
The Flask app includes CORS configuration, but if you encounter issues:
- Ensure `flask-cors` is installed
- Check browser console for CORS errors
- Verify API endpoints are accessible

## 📊 Environment Variables (Optional)

Create a `.env` file for configuration:
```
FLASK_APP=app.py
FLASK_ENV=development
FLASK_DEBUG=True
MODEL_PATH=heart_disease_model.pkl
PORT=5000
```

Load with:
```powershell
pip install python-dotenv
```

## 🔒 Production Deployment

For production deployment:

1. **Use a production WSGI server**:
```powershell
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

2. **Set environment variables**:
```powershell
set FLASK_ENV=production
set FLASK_DEBUG=False
```

3. **Use a reverse proxy** (nginx, Apache)
4. **Enable HTTPS**
5. **Set up monitoring and logging**

## ✅ Success Indicators

Your backend is properly set up when:
- ✅ Virtual environment activates without errors
- ✅ All packages install successfully
- ✅ Flask server starts without errors
- ✅ Health check endpoint returns 200 OK
- ✅ Frontend loads at http://localhost:5000
- ✅ Predictions work through the web interface

## 📞 Support

If you encounter issues:
1. Check the terminal output for error messages
2. Verify all dependencies are installed
3. Ensure the model file exists and is not corrupted
4. Check Windows firewall and antivirus settings
5. Try running with administrator privileges if needed

## 📚 Additional Resources

- [Flask Documentation](https://flask.palletsprojects.com/)
- [scikit-learn Documentation](https://scikit-learn.org/)
- [Python Virtual Environments](https://docs.python.org/3/tutorial/venv.html)