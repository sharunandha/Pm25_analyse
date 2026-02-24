"""
Flask Web Application
PM2.5 Estimation System - Main Application

Author: PM2.5 Estimation System
"""

from flask import Flask, render_template, request, jsonify, url_for
import os
from werkzeug.utils import secure_filename
from datetime import datetime
import traceback
import json
import numpy as np

# Import our custom modules
from image_analysis import ImageAnalyzer
from pm25_estimator import PM25Estimator
from visualization import PM25Visualizer


# Custom JSON Encoder to handle numpy types
class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that can handle numpy data types."""
    def default(self, obj):
        if isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# Initialize Flask app
app = Flask(__name__)
app.json_encoder = NumpyEncoder
app.config['SECRET_KEY'] = 'pm25-estimation-secret-key-2026'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RESULTS_FOLDER'] = 'static/results'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'tif', 'tiff', 'bmp'}

# Ensure required directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)
os.makedirs('data', exist_ok=True)


def allowed_file(filename):
    """Check if uploaded file has allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    """
    Handle image upload and perform PM2.5 analysis.
    """
    try:
        # Check if file was uploaded
        if 'satellite_image' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['satellite_image']
        
        # Check if file is selected
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Validate file type
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Please upload an image file.'}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)
        
        print(f"✓ Image saved: {filepath}")
        
        # Step 1: Analyze image to extract atmospheric features
        print("Analyzing atmospheric features...")
        analyzer = ImageAnalyzer(filepath)
        features = analyzer.analyze()
        print(f"✓ Features extracted: {features}")
        
        # Step 2: Estimate PM2.5 from features
        print("Estimating PM2.5 concentration...")
        estimator = PM25Estimator()
        estimation_results = estimator.estimate_with_confidence(features)
        pm25_value = estimation_results['pm25']
        print(f"✓ PM2.5 estimated: {pm25_value} µg/m³")
        
        # Step 3: Create visualizations
        print("Generating visualizations...")
        visualizer = PM25Visualizer(app.config['RESULTS_FOLDER'])
        
        # Generate all visualizations
        vis_timestamp = timestamp
        heatmap_path = visualizer.create_heatmap(
            filepath, pm25_value, f'heatmap_{vis_timestamp}.png'
        )
        print(f"✓ Heatmap created: {heatmap_path}")
        
        before_after_path = visualizer.create_before_after(
            filepath, f'before_after_{vis_timestamp}.png'
        )
        print(f"✓ Before/After created: {before_after_path}")
        
        dehazed_path = visualizer.create_dehazed(
            filepath, f'dehazed_{vis_timestamp}.png'
        )
        print(f"✓ Dehazed image created: {dehazed_path}")
        
        timeseries_path = visualizer.create_timeseries_graph(
            pm25_value, output_name=f'timeseries_{vis_timestamp}.png'
        )
        print(f"✓ Time series created: {timeseries_path}")
        
        features_chart_path = visualizer.create_feature_chart(
            features, output_name=f'features_{vis_timestamp}.png'
        )
        print(f"✓ Feature chart created: {features_chart_path}")
        
        # Prepare response with all results
        response_data = {
            'success': True,
            'pm25': float(pm25_value),
            'confidence': float(estimation_results['confidence']),
            'aqi_category': estimation_results['aqi_category'],
            'aqi_color': estimation_results['aqi_color'],
            'health_advice': estimation_results['health_advice'],
            'features': {
                'haze_score': float(round(features['haze_score'], 2)),
                'turbidity': float(round(features['turbidity'], 2)),
                'visibility': float(round(features['visibility'], 2)),
                'contrast': float(round(features['contrast'], 2)),
                'brightness': float(round(features['brightness'], 2)),
                'saturation': float(round(features['saturation'], 2))
            },
            'images': {
                'original': url_for('static', filename=f'uploads/{unique_filename}'),
                'heatmap': url_for('static', filename=f'results/heatmap_{vis_timestamp}.png'),
                'dehazed': url_for('static', filename=f'results/dehazed_{vis_timestamp}.png'),
                'before_after': url_for('static', filename=f'results/before_after_{vis_timestamp}.png'),
                'timeseries': url_for('static', filename=f'results/timeseries_{vis_timestamp}.png'),
                'features_chart': url_for('static', filename=f'results/features_{vis_timestamp}.png')
            },
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        print("✓ Analysis complete!")
        return jsonify(response_data)
    
    except Exception as e:
        print(f"✗ Error during analysis: {str(e)}")
        print(traceback.format_exc())
        return jsonify({
            'error': f'Analysis failed: {str(e)}',
            'details': traceback.format_exc()
        }), 500


@app.route('/about')
def about():
    """Return information about the system."""
    info = {
        'title': 'PM2.5 Estimation System',
        'version': '1.0.0',
        'description': 'High-Resolution PM2.5 Estimation from Satellite Images Using Image Processing',
        'author': 'Final Year Engineering Project',
        'features': [
            'Image-based PM2.5 estimation (no ML training required)',
            'Real-time atmospheric feature extraction',
            'Multiple visualization outputs',
            'AQI category classification',
            'Historical trend analysis'
        ],
        'technology': {
            'backend': 'Flask + Python',
            'image_processing': 'OpenCV + NumPy',
            'visualization': 'Matplotlib + Seaborn',
            'frontend': 'HTML + CSS + JavaScript'
        }
    }
    return jsonify(info)


@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat()
    })


if __name__ == '__main__':
    print("=" * 60)
    print("PM2.5 ESTIMATION SYSTEM")
    print("High-Resolution PM2.5 Estimation from Satellite Images")
    print("=" * 60)
    print("\nStarting Flask application...")
    print("Server will be available at: http://127.0.0.1:5000")
    print("Press CTRL+C to stop the server\n")
    print("=" * 60)
    
    # Run Flask app
    app.run(debug=True, host='127.0.0.1', port=5000)
