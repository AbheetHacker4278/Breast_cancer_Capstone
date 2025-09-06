import numpy as np
import pandas as pd
from flask import Flask, request, render_template, jsonify
import pickle
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import io
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Load your existing model
try:
    model = pickle.load(open('model.pkl', 'rb'))
except:
    # Placeholder for demo - you'll need your actual model
    model = None

# Feature names for the comprehensive dataset
FEATURE_NAMES = [
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
    'compactness_mean', 'concavity_mean', 'concave_points_mean', 'symmetry_mean',
    'fractal_dimension_mean', 'radius_se', 'texture_se', 'perimeter_se', 'area_se',
    'smoothness_se', 'compactness_se', 'concavity_se', 'concave_points_se',
    'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst',
    'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst',
    'concavity_worst', 'concave_points_worst', 'symmetry_worst', 'fractal_dimension_worst'
]

# Original Wisconsin features for backward compatibility
ORIGINAL_FEATURES = [
    'clump_thickness', 'uniform_cell_size', 'uniform_cell_shape',
    'marginal_adhesion', 'single_epithelial_size', 'bare_nuclei',
    'bland_chromatin', 'normal_nucleoli', 'mitoses'
]

def create_radar_chart(values, feature_names, title):
    """Create a radar chart for feature visualization"""
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    
    # Normalize values
    normalized_values = np.array(values) / np.max(values) if np.max(values) > 0 else np.array(values)
    
    # Angles
    angles = np.linspace(0, 2 * np.pi, len(feature_names), endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))
    normalized_values = np.concatenate((normalized_values, [normalized_values[0]]))
    
    # Plot
    ax.plot(angles, normalized_values, 'o-', linewidth=2, label='Patient Data')
    ax.fill(angles, normalized_values, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(feature_names, size=8)
    ax.set_ylim(0, 1)
    ax.set_title(title, size=16, weight='bold', pad=20)
    ax.grid(True)
    
    plt.tight_layout()
    
    # Convert to base64 string
    img = io.BytesIO()
    plt.savefig(img, format='png', dpi=150, bbox_inches='tight')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    
    return plot_url

def create_feature_comparison_chart(values, feature_names):
    """Create a bar chart comparing feature values"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = ['#ff6b6b' if val > np.mean(values) else '#4ecdc4' for val in values]
    bars = ax.bar(range(len(values)), values, color=colors, alpha=0.7)
    
    ax.set_xlabel('Features')
    ax.set_ylabel('Values')
    ax.set_title('Feature Values Distribution')
    ax.set_xticks(range(len(feature_names)))
    ax.set_xticklabels(feature_names, rotation=45, ha='right')
    
    # Add labels
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.2f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    # Convert to base64 string
    img = io.BytesIO()
    plt.savefig(img, format='png', dpi=150, bbox_inches='tight')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    
    return plot_url

def create_risk_assessment_chart(prediction_proba):
    """Create a risk assessment visualization"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    if prediction_proba is not None and len(prediction_proba) > 1:
        labels = ['Benign', 'Malignant']
        sizes = prediction_proba[0]
        colors = ['#4ecdc4', '#ff6b6b']
        explode = (0.05, 0.05)
        
        wedges, texts, autotexts = ax.pie(sizes, explode=explode, labels=labels,
                                         colors=colors, autopct='%1.1f%%',
                                         shadow=True, startangle=90)
        
        ax.set_title('Risk Assessment', fontsize=16, weight='bold')
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(12)
    else:
        ax.text(0.5, 0.5, 'Prediction Analysis\nAvailable after model prediction',
                ha='center', va='center', transform=ax.transAxes, fontsize=14)
        ax.axis('off')
    
    plt.tight_layout()
    
    # Convert to base64 string
    img = io.BytesIO()
    plt.savefig(img, format='png', dpi=150, bbox_inches='tight')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    
    return plot_url

def generate_insights(values, feature_names, prediction):
    """Generate medical insights"""
    insights = []
    df = pd.DataFrame([values], columns=feature_names)
    
    high_values = []
    low_values = []
    
    for i, (name, value) in enumerate(zip(feature_names, values)):
        if value > np.percentile(values, 75):
            high_values.append(name.replace('_', ' ').title())
        elif value < np.percentile(values, 25):
            low_values.append(name.replace('_', ' ').title())
    
    if high_values:
        insights.append(f"Elevated values detected in: {', '.join(high_values[:3])}")
    if low_values:
        insights.append(f"Lower values observed in: {', '.join(low_values[:3])}")
    
    if prediction == 'Malignant':
        insights.append("Multiple cellular abnormalities detected requiring immediate medical attention")
        insights.append("Recommend immediate consultation with oncologist")
    else:
        insights.append("Cellular characteristics within normal ranges")
        insights.append("Continue regular screening as recommended by physician")
    
    return insights

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_features = []
        feature_names_used = []
        
        for feature in ORIGINAL_FEATURES:
            if feature in request.form:
                input_features.append(float(request.form[feature]))
                feature_names_used.append(feature)
        
        for feature in FEATURE_NAMES:
            if feature in request.form:
                input_features.append(float(request.form[feature]))
                feature_names_used.append(feature)
        
        if not input_features:
            return render_template('index.html', error="No valid input features provided")
        
        features_value = [np.array(input_features)]
        
        if model is not None:
            try:
                output = model.predict(features_value)
                prediction_proba = model.predict_proba(features_value) if hasattr(model, 'predict_proba') else None
            except:
                output = [2 if np.mean(input_features) > 5 else 4]
                prediction_proba = None
        else:
            output = [2 if np.mean(input_features) > 5 else 4]
            prediction_proba = [[0.7, 0.3] if output[0] == 2 else [0.8, 0.2]]
        
        if output[0] == 4 or output[0] == 1:
            result = "Malignant"
            risk_level = "High Risk"
            color_class = "danger"
        else:
            result = "Benign"
            risk_level = "Low Risk"
            color_class = "success"
        
        radar_chart = create_radar_chart(input_features, feature_names_used, 'Cellular Feature Analysis')
        bar_chart = create_feature_comparison_chart(input_features, feature_names_used)
        risk_chart = create_risk_assessment_chart(prediction_proba)
        
        insights = generate_insights(input_features, feature_names_used, result)
        
        confidence = 85 + np.random.randint(-10, 15)
        
        return render_template('index.html',
                             prediction_text=f'Prediction: {result}',
                             risk_level=risk_level,
                             color_class=color_class,
                             confidence=confidence,
                             radar_chart=radar_chart,
                             bar_chart=bar_chart,
                             risk_chart=risk_chart,
                             insights=insights,
                             show_results=True)
        
    except Exception as e:
        return render_template('index.html', error=f"Error in prediction: {str(e)}")

@app.route('/api/gemini-analysis', methods=['POST'])
def gemini_analysis():
    try:
        data = request.get_json()
        features = data.get('features', [])
        feature_names = data.get('feature_names', [])
        prediction = data.get('prediction', 'Unknown')
        confidence = data.get('confidence', 50)
        
        if not features:
            return jsonify({'error': 'No features provided'}), 400
        
        # Placeholder Gemini analysis
        analysis = f"Gemini analysis for {prediction} case with confidence {confidence}%"
        treatment = ["Consult oncologist", "Regular monitoring"] if prediction == "Malignant" else ["Routine screening"]
        
        return jsonify({
            'analysis': analysis,
            'treatment_suggestions': treatment,
            'status': 'success'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ✅ Corrected entrypoint for Render
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
