// Heart Disease Prediction App JavaScript - FDM Mini Project 2025

class HeartDiseasePredictor {
    constructor() {
        this.form = document.getElementById('predictionForm');
        this.resultsContainer = document.getElementById('resultsContainer');
        this.resultSummary = document.getElementById('resultSummary');
        this.resultDetails = document.getElementById('resultDetails');
        this.predictBtn = document.getElementById('predictBtn');
        this.newPredictionBtn = document.getElementById('newPredictionBtn');
        this.shareResultsBtn = document.getElementById('shareResultsBtn');
        this.toggleAdvancedBtn = document.getElementById('toggleAdvanced');
        this.advancedSection = document.getElementById('advancedSection');
        
        this.initializeEventListeners();
        this.initializeValidation();
        this.checkAPIHealth();
    }
    
    initializeEventListeners() {
        // Form submission
        this.form.addEventListener('submit', (e) => this.handleFormSubmit(e));
        
        // Advanced options toggle
        this.toggleAdvancedBtn.addEventListener('click', () => this.toggleAdvancedOptions());
        
        // Result actions
        this.newPredictionBtn.addEventListener('click', () => this.resetForm());
        this.shareResultsBtn.addEventListener('click', () => this.shareResults());
        
        // Real-time validation
        const inputs = this.form.querySelectorAll('input, select');
        inputs.forEach(input => {
            input.addEventListener('input', () => this.validateField(input));
            input.addEventListener('blur', () => this.validateField(input));
        });
        
        // BMI calculator helper
        const ageInput = document.getElementById('age');
        const weightInput = document.getElementById('weight');
        const heightInput = document.getElementById('height');
        
        // Auto-calculate BMI if weight and height are provided (if we add these fields)
        if (weightInput && heightInput) {
            [weightInput, heightInput].forEach(input => {
                input.addEventListener('input', () => this.calculateBMI());
            });
        }
    }
    
    initializeValidation() {
        // Set up form validation rules
        this.validationRules = {
            age: { min: 1, max: 120, required: true },
            gender: { required: true },
            blood_pressure: { min: 80, max: 200, required: true },
            cholesterol_level: { min: 100, max: 400, required: true },
            smoking: { required: true },
            bmi: { min: 10, max: 50, required: true },
            fasting_blood_sugar: { min: 60, max: 300 },
            triglyceride_level: { min: 50, max: 500 },
            crp_level: { min: 0, max: 20 },
            sleep_hours: { min: 3, max: 12 },
            homocysteine_level: { min: 5, max: 30 }
        };
    }
    
    async checkAPIHealth() {
        try {
            const response = await fetch('/api/health');
            const data = await response.json();
            
            if (!data.model_loaded) {
                this.showNotification('Model is loading, please wait...', 'warning');
            }
        } catch (error) {
            console.error('API health check failed:', error);
            this.showNotification('API connection failed. Please refresh the page.', 'error');
        }
    }
    
    validateField(field) {
        const name = field.name;
        const value = field.value.trim();
        const rules = this.validationRules[name];
        
        // Remove existing error states
        field.classList.remove('error', 'success');
        this.removeErrorMessage(field);
        
        if (!rules) return true;
        
        // Required field validation
        if (rules.required && !value) {
            this.showFieldError(field, 'This field is required');
            return false;
        }
        
        // Skip further validation if field is empty and not required
        if (!value && !rules.required) {
            return true;
        }
        
        // Numeric validation
        if (field.type === 'number') {
            const numValue = parseFloat(value);
            
            if (isNaN(numValue)) {
                this.showFieldError(field, 'Please enter a valid number');
                return false;
            }
            
            if (rules.min !== undefined && numValue < rules.min) {
                this.showFieldError(field, `Value must be at least ${rules.min}`);
                return false;
            }
            
            if (rules.max !== undefined && numValue > rules.max) {
                this.showFieldError(field, `Value must be no more than ${rules.max}`);
                return false;
            }
        }
        
        // Success state
        field.classList.add('success');
        return true;
    }
    
    showFieldError(field, message) {
        field.classList.add('error');
        
        const errorElement = document.createElement('div');
        errorElement.className = 'error-message';
        errorElement.textContent = message;
        
        field.parentNode.appendChild(errorElement);
    }
    
    removeErrorMessage(field) {
        const existingError = field.parentNode.querySelector('.error-message');
        if (existingError) {
            existingError.remove();
        }
    }
    
    validateForm() {
        const inputs = this.form.querySelectorAll('input[required], select[required]');
        let isValid = true;
        
        inputs.forEach(input => {
            if (!this.validateField(input)) {
                isValid = false;
            }
        });
        
        return isValid;
    }
    
    async handleFormSubmit(e) {
        e.preventDefault();
        
        if (!this.validateForm()) {
            this.showNotification('Please correct the errors in the form', 'error');
            return;
        }
        
        this.showLoading();
        
        try {
            const formData = this.collectFormData();
            const result = await this.makePrediction(formData);
            
            this.hideLoading();
            this.displayResults(result);
            
        } catch (error) {
            this.hideLoading();
            console.error('Prediction error:', error);
            this.showNotification('Prediction failed. Please try again.', 'error');
        }
    }
    
    collectFormData() {
        const formData = new FormData(this.form);
        const data = {};
        
        // Collect all form fields
        for (let [key, value] of formData.entries()) {
            if (value.trim()) {
                // Convert numeric fields
                const field = this.form.querySelector(`[name="${key}"]`);
                if (field && field.type === 'number') {
                    data[key] = parseFloat(value);
                } else {
                    data[key] = value;
                }
            }
        }
        
        return data;
    }
    
    async makePrediction(data) {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data)
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const result = await response.json();
        
        if (!result.success) {
            throw new Error(result.error || 'Prediction failed');
        }
        
        return result;
    }
    
    displayResults(result) {
        // Hide form and show results
        document.querySelector('.form-container').style.display = 'none';
        this.resultsContainer.style.display = 'block';
        
        // Scroll to results
        this.resultsContainer.scrollIntoView({ behavior: 'smooth' });
        
        // Display result summary
        this.displayResultSummary(result);
        
        // Display detailed results
        this.displayResultDetails(result);
    }
    
    displayResultSummary(result) {
        const { prediction, risk_level, risk_percentage, risk_color, recommendation } = result;
        
        // Set risk level class
        this.resultSummary.className = `result-summary ${risk_level.toLowerCase()}-risk`;
        
        const riskIcon = this.getRiskIcon(risk_level);
        
        this.resultSummary.innerHTML = `
            <div class="risk-icon">
                <i class="${riskIcon}" style="color: ${risk_color}; font-size: 4rem;"></i>
            </div>
            <div class="risk-level" style="color: ${risk_color};">
                ${risk_level} Risk
            </div>
            <div class="risk-percentage" style="color: ${risk_color};">
                ${risk_percentage}
            </div>
            <div class="risk-description">
                <strong>Prediction:</strong> ${prediction === 'Yes' ? 'Heart Disease Detected' : 'No Heart Disease Detected'}
            </div>
            <div class="recommendation">
                <i class="fas fa-lightbulb"></i> ${recommendation}
            </div>
        `;
    }
    
    displayResultDetails(result) {
        const { confidence, risk_probability, input_data } = result;
        
        this.resultDetails.innerHTML = `
            <div class="detail-card">
                <h4><i class="fas fa-percentage"></i> Risk Probability</h4>
                <p>${(risk_probability * 100).toFixed(1)}%</p>
            </div>
            <div class="detail-card">
                <h4><i class="fas fa-certificate"></i> Confidence Level</h4>
                <p>${confidence}</p>
            </div>
            <div class="detail-card">
                <h4><i class="fas fa-user"></i> Age</h4>
                <p>${input_data.Age} years</p>
            </div>
            <div class="detail-card">
                <h4><i class="fas fa-heartbeat"></i> Blood Pressure</h4>
                <p>${input_data['Blood Pressure']} mmHg</p>
            </div>
            <div class="detail-card">
                <h4><i class="fas fa-weight"></i> BMI</h4>
                <p>${input_data.BMI.toFixed(1)} kg/m²</p>
            </div>
            <div class="detail-card">
                <h4><i class="fas fa-smoking"></i> Smoking Status</h4>
                <p>${input_data.Smoking}</p>
            </div>
        `;
        
        // Add recommendation section
        const recommendationSection = document.createElement('div');
        recommendationSection.className = 'recommendation-section';
        recommendationSection.innerHTML = `
            <h4><i class="fas fa-medical-kit"></i> Health Recommendations</h4>
            <div class="recommendations">
                ${this.generateRecommendations(result)}
            </div>
        `;
        
        this.resultDetails.appendChild(recommendationSection);
    }
    
    generateRecommendations(result) {
        const { risk_level, input_data } = result;
        const recommendations = [];
        
        // General recommendations based on risk level
        if (risk_level === 'High') {
            recommendations.push('Schedule an immediate consultation with a cardiologist');
            recommendations.push('Consider comprehensive cardiac screening tests');
        } else if (risk_level === 'Medium') {
            recommendations.push('Schedule a check-up with your primary care physician');
            recommendations.push('Monitor your cardiovascular health regularly');
        } else {
            recommendations.push('Continue maintaining your healthy lifestyle');
            recommendations.push('Schedule routine health check-ups annually');
        }
        
        // Specific recommendations based on input data
        if (input_data.Smoking === 'Yes') {
            recommendations.push('Consider smoking cessation programs');
        }
        
        if (input_data.BMI > 30) {
            recommendations.push('Consult with a nutritionist for weight management');
        }
        
        if (input_data['Blood Pressure'] > 140) {
            recommendations.push('Monitor blood pressure regularly');
        }
        
        if (input_data['Exercise Habits'] === 'Low') {
            recommendations.push('Gradually increase physical activity levels');
        }
        
        return recommendations.map(rec => `<div class="recommendation-item"><i class="fas fa-check-circle"></i> ${rec}</div>`).join('');
    }
    
    getRiskIcon(riskLevel) {
        const icons = {
            'Low': 'fas fa-check-circle',
            'Medium': 'fas fa-exclamation-triangle',
            'High': 'fas fa-exclamation-circle'
        };
        return icons[riskLevel] || 'fas fa-question-circle';
    }
    
    toggleAdvancedOptions() {
        const isHidden = this.advancedSection.style.display === 'none';
        
        if (isHidden) {
            this.advancedSection.style.display = 'block';
            this.toggleAdvancedBtn.innerHTML = '<i class="fas fa-minus"></i> Hide Advanced Options';
        } else {
            this.advancedSection.style.display = 'none';
            this.toggleAdvancedBtn.innerHTML = '<i class="fas fa-plus"></i> Show Advanced Options';
        }
    }
    
    resetForm() {
        // Reset form
        this.form.reset();
        
        // Remove validation states
        const inputs = this.form.querySelectorAll('input, select');
        inputs.forEach(input => {
            input.classList.remove('error', 'success');
            this.removeErrorMessage(input);
        });
        
        // Show form and hide results
        document.querySelector('.form-container').style.display = 'block';
        this.resultsContainer.style.display = 'none';
        
        // Scroll to top of form
        document.querySelector('.form-container').scrollIntoView({ behavior: 'smooth' });
    }
    
    async shareResults() {
        const resultText = this.generateShareText();
        
        if (navigator.share) {
            try {
                await navigator.share({
                    title: 'Heart Disease Risk Assessment Results',
                    text: resultText,
                    url: window.location.href
                });
            } catch (error) {
                this.fallbackShare(resultText);
            }
        } else {
            this.fallbackShare(resultText);
        }
    }
    
    fallbackShare(text) {
        // Copy to clipboard
        navigator.clipboard.writeText(text).then(() => {
            this.showNotification('Results copied to clipboard!', 'success');
        }).catch(() => {
            // Fallback for older browsers
            const textarea = document.createElement('textarea');
            textarea.value = text;
            document.body.appendChild(textarea);
            textarea.select();
            document.execCommand('copy');
            document.body.removeChild(textarea);
            this.showNotification('Results copied to clipboard!', 'success');
        });
    }
    
    generateShareText() {
        const summary = this.resultSummary.textContent.replace(/\s+/g, ' ').trim();
        return `Heart Disease Risk Assessment Results:\n\n${summary}\n\nGenerated by FDM Mini Project 2025 - Heart Disease Prediction System`;
    }
    
    showLoading() {
        this.predictBtn.disabled = true;
        this.predictBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processing...';
    }
    
    hideLoading() {
        this.predictBtn.disabled = false;
        this.predictBtn.innerHTML = '<i class="fas fa-chart-line"></i> Analyze Risk';
    }
    
    showNotification(message, type = 'info') {
        // Create notification element
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.innerHTML = `
            <div class="notification-content">
                <i class="fas ${this.getNotificationIcon(type)}"></i>
                <span>${message}</span>
                <button class="notification-close">
                    <i class="fas fa-times"></i>
                </button>
            </div>
        `;
        
        // Add styles
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background: ${this.getNotificationColor(type)};
            color: white;
            padding: 1rem 1.5rem;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
            z-index: 1001;
            animation: slideIn 0.3s ease;
            max-width: 400px;
        `;
        
        // Add to DOM
        document.body.appendChild(notification);
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (notification.parentNode) {
                notification.remove();
            }
        }, 5000);
        
        // Manual close
        const closeBtn = notification.querySelector('.notification-close');
        closeBtn.addEventListener('click', () => notification.remove());
    }
    
    getNotificationIcon(type) {
        const icons = {
            'success': 'fa-check-circle',
            'error': 'fa-exclamation-circle',
            'warning': 'fa-exclamation-triangle',
            'info': 'fa-info-circle'
        };
        return icons[type] || 'fa-info-circle';
    }
    
    getNotificationColor(type) {
        const colors = {
            'success': '#10b981',
            'error': '#dc2626',
            'warning': '#f59e0b',
            'info': '#2563eb'
        };
        return colors[type] || '#2563eb';
    }
    
    calculateBMI() {
        const weight = parseFloat(document.getElementById('weight')?.value);
        const height = parseFloat(document.getElementById('height')?.value);
        
        if (weight && height) {
            const bmi = weight / (height * height);
            const bmiInput = document.getElementById('bmi');
            if (bmiInput) {
                bmiInput.value = bmi.toFixed(1);
                this.validateField(bmiInput);
            }
        }
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new HeartDiseasePredictor();
});

// Add CSS animations
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from {
            transform: translateX(100%);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    .notification-content {
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .notification-close {
        background: none;
        border: none;
        color: inherit;
        cursor: pointer;
        padding: 0.25rem;
        margin-left: auto;
    }
    
    .recommendation-section {
        grid-column: 1 / -1;
        background: #f8fafc;
        padding: 1.5rem;
        border-radius: 8px;
        border-left: 4px solid var(--primary-color);
    }
    
    .recommendation-section h4 {
        color: var(--text-primary);
        margin-bottom: 1rem;
        font-size: 1.1rem;
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .recommendations {
        display: grid;
        gap: 0.75rem;
    }
    
    .recommendation-item {
        display: flex;
        align-items: flex-start;
        gap: 0.75rem;
        padding: 0.75rem;
        background: white;
        border-radius: 6px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }
    
    .recommendation-item i {
        color: var(--success-color);
        margin-top: 0.1rem;
        flex-shrink: 0;
    }
`;
document.head.appendChild(style);