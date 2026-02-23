# 🛰️ PM2.5 ESTIMATION SYSTEM - COMPLETE PROJECT EXPLANATION

## 📋 TABLE OF CONTENTS
1. [Project Overview](#project-overview)
2. [Technologies Used](#technologies-used)
3. [System Architecture](#system-architecture)
4. [Step-by-Step Workflow](#step-by-step-workflow)
5. [Core Modules Explained](#core-modules-explained)
6. [How Each Component Works](#how-each-component-works)
7. [Data Flow Diagram](#data-flow-diagram)
8. [Deployment Details](#deployment-details)

---

## 🎯 PROJECT OVERVIEW

### What is this project?
A **web-based PM2.5 Estimation System** that analyzes satellite/aerial images to estimate air pollution levels (PM2.5 - Particulate Matter 2.5 micrometers or smaller).

### Key Characteristics
- ✅ **No Machine Learning** - Uses formula-based image processing, not neural networks
- ✅ **No Training Data Required** - Works with pre-defined formulas and coefficients
- ✅ **Fully Offline** - No internet connection needed
- ✅ **Web Interface** - Clean, professional UI in the browser
- ✅ **Real-time Analysis** - Instant PM2.5 estimation from any image
- ✅ **Multiple Outputs** - Heatmaps, graphs, comparisons, and analysis charts

### Real-World Application
- Environmental monitoring
- Air quality assessment
- Pollution tracking
- Urban planning
- Climate research

---

## 💻 TECHNOLOGIES USED

### Backend Technologies

| Technology | Version | Purpose |
|-----------|---------|---------|
| **Python** | 3.11+ | Primary programming language |
| **Flask** | 3.0.0 | Web framework for routing and server |
| **Werkzeug** | 3.0.1 | WSGI utility library for Flask |
| **Gunicorn** | 21.2.0 | Production WSGI server (for Railway) |
| **OpenCV** | 4.8.1.78 | Computer vision library for image processing |
| **OpenCV-Headless** | 4.8.1.78 | Server version without display dependencies |
| **NumPy** | 1.26.3 | Numerical computing and array operations |
| **Pillow** | 11.0.0 | Image manipulation library |
| **Matplotlib** | 3.8.2 | Data visualization and graphing |
| **Seaborn** | 0.13.0 | Statistical data visualization |
| **Pandas** | 2.1.4 | Data manipulation and CSV handling |
| **Python-dateutil** | 2.8.2 | Date/time utilities |

### Frontend Technologies

| Technology | Purpose |
|-----------|---------|
| **HTML5** | Structure of web pages |
| **CSS3** | Styling and responsive design |
| **JavaScript (Vanilla)** | Client-side interactivity and AJAX |
| **Bootstrap Classes** | Grid and responsive layout |

### Deployment Platform

| Component | Details |
|-----------|---------|
| **Hosting** | Railway.app |
| **Container** | NIX packages (automatic) |
| **Server** | Gunicorn with 0.0.0.0:8000 |
| **Python Runtime** | 3.12/3.13 |

---

## 🏗️ SYSTEM ARCHITECTURE

### Project Structure

```
PM25_PROJECT/
│
├── 📄 BACKEND (Python Modules)
│   ├── app.py                    # Main Flask application (202 lines)
│   ├── image_analysis.py         # Feature extraction (269 lines)
│   ├── pm25_estimator.py         # PM2.5 calculation (235 lines)
│   └── visualization.py          # Graph generation (331 lines)
│
├── 🌐 FRONTEND (Web Interface)
│   ├── templates/
│   │   └── index.html            # Main web page
│   └── static/
│       ├── css/
│       │   └── style.css         # Styling
│       ├── uploads/              # User uploaded images
│       └── results/              # Generated visualizations
│
├── 📚 CONFIGURATION & DOCUMENTATION
│   ├── requirements.txt          # Python dependencies
│   ├── railway.json              # Railway deployment config
│   ├── Procfile                  # Gunicorn startup command
│   ├── runtime.txt               # Python version
│   └── data/
│       └── pm25_history.csv      # Historical PM2.5 data
│
├── 📖 DOCUMENTATION
│   ├── README.md                 # Comprehensive guide (400+ lines)
│   ├── QUICKSTART.md             # Fast setup guide
│   ├── START_HERE.md             # Getting started
│   ├── PROJECT_SUMMARY.md        # Project details
│   ├── VIVA_GUIDE.md             # Presentation guide
│   ├── RAILWAY_DEPLOYMENT.md     # Deployment instructions
│   └── COMPLETE_PROJECT_EXPLANATION.md  # This file
│
└── 🔧 SETUP SCRIPTS
    ├── setup_and_run.bat         # Windows quick setup
    └── setup_and_run.sh          # Linux/Mac quick setup
```

### System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     USER INTERFACE (Web Browser)             │
│               HTML + CSS + JavaScript (index.html)           │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  - Image Upload Form                                   │  │
│  │  - Results Display (PM2.5 value, category, advice)    │  │
│  │  - Visualizations (4 graphs/heatmaps)                 │  │
│  └────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                         ⬇️  HTTP/AJAX
┌─────────────────────────────────────────────────────────────┐
│                 FLASK WEB SERVER (app.py)                    │
│  Routes: /  /analyze  /about  /health                       │
│  Handles file upload, request routing, response formatting  │
└─────────────────────────────────────────────────────────────┘
                         ⬇️  Function Calls
┌─────────────────────────────────────────────────────────────┐
│              PYTHON PROCESSING MODULES                       │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ 1. ImageAnalyzer (image_analysis.py)                 │   │
│  │    - Load & preprocess image                         │   │
│  │    - Extract 6 atmospheric features                  │   │
│  │    - Returns feature dictionary                      │   │
│  └──────────────────────────────────────────────────────┘   │
│                         ⬇️                                    │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ 2. PM25Estimator (pm25_estimator.py)                 │   │
│  │    - Apply weighted formula                          │   │
│  │    - Calculate PM2.5 concentration                   │   │
│  │    - Determine AQI category & health advice          │   │
│  └──────────────────────────────────────────────────────┘   │
│                         ⬇️                                    │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ 3. PM25Visualizer (visualization.py)                 │   │
│  │    - Create heatmap from original image              │   │
│  │    - Generate before/after comparison                │   │
│  │    - Plot time-series graph                          │   │
│  │    - Create feature analysis chart                   │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                         ⬇️  File Storage
┌─────────────────────────────────────────────────────────────┐
│                    FILE SYSTEM STORAGE                       │
│  - static/uploads/      → Original satellite images         │
│  - static/results/      → Generated visualizations          │
│  - data/pm25_history.csv → Historical data                 │
└─────────────────────────────────────────────────────────────┘
```

---

## 📊 STEP-BY-STEP WORKFLOW

### Complete User Journey (6 Steps)

#### **Step 1: User Opens Web Application**
```
User opens browser → http://localhost:5000 (local) or Railway URL
          ⬇️
Flask receives GET request to '/' route
          ⬇️
Flask renders index.html from templates/
          ⬇️
Web page displays with upload form and interface
```

#### **Step 2: User Selects & Uploads Image**
```
User clicks "Choose File" button
User selects satellite/aerial image (PNG, JPG, TIFF, etc.)
User clicks "Analyze Image" button
          ⬇️
JavaScript collects file and makes POST request to /analyze
Request includes: satellite_image (binary file data)
```

#### **Step 3: Image Preprocessing (ImageAnalyzer)**
```
app.py receives POST request
          ⬇️
File validation:
  - Check file exists
  - Validate file extension (png, jpg, jpeg, tif, tiff, bmp)
  - Check file size (max 16MB)
          ⬇️
Save uploaded file with timestamp: static/uploads/20260123_084317_satellite.jpg
          ⬇️
ImageAnalyzer loads image:
  - Read image using OpenCV
  - Resize to standard 800x600 pixels (consistency)
  - Convert to grayscale (for edge detection, brightness)
  - Convert to HSV color space (for saturation analysis)
```

#### **Step 4: Feature Extraction (ImageAnalyzer - 6 Features)**
```
From preprocessed image, extract 6 atmospheric indicators:

┌─ Feature 1: HAZE SCORE (0-100) ─┐
│ How: Apply Laplacian edge detection
│      Calculate variance of edges
│      Low variance = more haze
│ Why: Haze reduces image sharpness
│      More haze → higher PM2.5
│ Formula: haze_score = 100 - (edge_variance/1000*100)
└────────────────────────────────────┘

┌─ Feature 2: TURBIDITY (0-100) ──┐
│ How: Split image to BGR channels
│      Find dark channel (min of R,G,B)
│      Calculate mean of dark channel
│ Why: Dark areas indicate atmospheric scattering
│      Particles cause more scattering
│ Formula: turbidity = (dark_channel_mean/255)*100
└─────────────────────────────────┘

┌─ Feature 3: VISIBILITY (0-100) ──┐
│ How: Analyze histogram distribution
│      Calculate entropy of pixel intensities
│      More spread = better visibility
│ Why: PM2.5 particles reduce visibility
│      Concentrated histogram = haze
│ Formula: visibility = histogram_entropy
└──────────────────────────────────┘

┌─ Feature 4: CONTRAST (0-100) ────┐
│ How: Calculate standard deviation
│      of all pixel values (grayscale)
│ Why: Pollution creates low contrast
│      Less variation = more pollution
│ Formula: contrast = (std_deviation/80)*100
└──────────────────────────────────┘

┌─ Feature 5: BRIGHTNESS (0-255) ──┐
│ How: Calculate mean of grayscale image
│ Why: Extreme brightness/darkness can
│      indicate atmospheric conditions
│ Formula: brightness = mean(grayscale_pixels)
└──────────────────────────────────┘

┌─ Feature 6: SATURATION (0-255) ──┐
│ How: Extract S channel from HSV
│      Calculate mean of saturation values
│ Why: Haze reduces color saturation
│      Lower saturation = more pollution
│ Formula: saturation = mean(HSV_S_channel)
└──────────────────────────────────┘

Returns: {
  'haze_score': 65.3,
  'turbidity': 48.2,
  'visibility': 72.1,
  'contrast': 55.8,
  'brightness': 145.6,
  'saturation': 110.2
}
```

#### **Step 5: PM2.5 Estimation (PM25Estimator)**
```
Receives feature dictionary
          ⬇️
Apply weighted formula:

PM2.5 = Base Offset + Weighted Features

PM2.5 = 20 (base)
      + 1.5 × 65.3   (haze_weight × haze_score)        = +97.95
      + 1.2 × 48.2   (turbidity_weight × turbidity)    = +57.84
      + (-0.8) × 72.1  (visibility_weight × visibility)  = -57.68
      + (-0.5) × 55.8  (contrast_weight × contrast)      = -27.9
      + 0.3 × (145.6/255)*100  (brightness_weight)       = +17.08
      + (-0.4) × (110.2/255)*100  (saturation_weight)    = -17.27
      ────────────────────────────────────────────────
      = Raw PM2.5 ≈ 90.02 µg/m³

          ⬇️
Apply Non-linear Correction:
(Prevent unrealistic extreme values)

Raw = 90.02 (in moderate range 50-150)
Corrected = 50 + (90.02 - 50) × 0.9 = 86.02

          ⬇️
Clamp to realistic range (0-300)
Final PM2.5 = 86.02 µg/m³

          ⬇️
Determine AQI Category:
PM2.5 = 86.02 falls in range 55.4-150.4
Category = "Unhealthy"
Color = Red (#FF0000)
Health Advice = "Everyone may begin to experience health effects"
Confidence = 82% (based on feature quality)

Returns: {
  'pm25': 86.02,
  'confidence': 82,
  'aqi_category': 'Unhealthy',
  'aqi_color': '#FF0000',
  'health_advice': 'Everyone may begin to experience health effects'
}
```

#### **Step 6: Visualization Generation (PM25Visualizer)**
```
Receives: original image, PM2.5 value, features

Generate 4 visualizations:

1️⃣ HEATMAP (heatmap_timestamp.png)
   - Apply color map to original image
   - Use COLORCV_JET or custom gradient
   - Color intensity represents PM2.5 concentration
   - Red = high PM2.5, Blue = low PM2.5
   - Add title, scale, timestamp

2️⃣ BEFORE/AFTER (before_after_timestamp.png)
   - Original image on left (Before)
   - Apply CLAHE enhancement on right (After)
   - CLAHE = Contrast Limited Adaptive Histogram Equalization
   - Shows what atmosphere would look without pollution
   - Split down middle for comparison

3️⃣ TIME SERIES GRAPH (timeseries_timestamp.png)
   - X-axis: Time (hourly from 0 to 24 hours)
   - Y-axis: PM2.5 concentration (µg/m³)
   - Current measurement marked with star
   - Line plot with trend
   - AQI categories as background bands
   - Reads/writes data/pm25_history.csv

4️⃣ FEATURE CHART (features_timestamp.png)
   - Bar chart of 6 atmospheric features
   - Shows which features contributed most
   - Color coded by severity
   - Helps understand PM2.5 composition

          ⬇️
Save all visualizations to static/results/
Return file paths and URLs
```

---

## 🔧 CORE MODULES EXPLAINED

### 1️⃣ **app.py** - Flask Web Application (202 lines)

**Purpose:** Main entry point, handles HTTP requests/responses

**Key Components:**

```python
# Configuration
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RESULTS_FOLDER'] = 'static/results'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit

# Routes (URL endpoints)
@app.route('/')                    # GET  - Main page
@app.route('/analyze', methods=['POST'])  # POST - Image analysis
@app.route('/about')              # GET  - API information
@app.route('/health')             # GET  - Health check
```

**Request Flow:**

```
User Action              → HTTP Method → Route Handler → Response
─────────────────────────────────────────────────────────────
Open website             → GET /       → render index.html
Upload image             → POST /analyze → JSON with results
Get app info             → GET /about  → JSON info
Check if alive           → GET /health → JSON status
```

**analyze() Function - Core Logic:**

```
1. Validate uploaded file
   └─ Check extension, size, filename

2. Save file with timestamp
   └─ static/uploads/20260123_084317_satellite.jpg

3. Create ImageAnalyzer instance
   └─ analyzer = ImageAnalyzer(filepath)
   └─ features = analyzer.analyze()

4. Create PM25Estimator instance
   └─ estimator = PM25Estimator()
   └─ results = estimator.estimate_with_confidence(features)

5. Create PM25Visualizer instance
   └─ visualizer = PM25Visualizer(results_folder)
   └─ Generate 4 visualization images

6. Build response JSON
   └─ Include PM2.5, AQI, features, image URLs, timestamp

7. Return JSON to frontend
   └─ Frontend displays results
```

### 2️⃣ **image_analysis.py** - Feature Extraction (269 lines)

**Purpose:** Extract 6 atmospheric features from satellite image

**Class: ImageAnalyzer**

```python
class ImageAnalyzer:
    def __init__(self, image_path):
        # Load image path
    
    def load_and_preprocess(self):
        # Read image → Resize to 800x600 → Convert to grayscale & HSV
    
    def calculate_haze_score(self):
        # Laplacian edge detection → Calculate variance → Invert
    
    def calculate_brightness(self):
        # Mean of grayscale pixels
    
    def calculate_contrast(self):
        # Standard deviation of pixel values → Normalize to 0-100
    
    def calculate_saturation(self):
        # Mean of HSV S-channel (0-255)
    
    def calculate_atmospheric_turbidity(self):
        # Dark channel prior (min of R,G,B) → Mean → Normalize
    
    def calculate_visibility_index(self):
        # Histogram entropy calculation
    
    def analyze(self):
        # Call all above functions → Return feature dictionary
```

**Feature Extraction Logic (Detailed):**

| Feature | Extraction Method | Why It Matters | Formula |
|---------|------------------|----------------|---------|
| **Haze Score** | Laplacian edge variance (inverted) | High haze = low edge sharpness = high PM2.5 | `100 - (variance/1000*100)` |
| **Turbidity** | Dark channel mean (BGR minimum) | Particles cause atmospheric scattering | `(dark_mean/255)*100` |
| **Visibility** | Histogram entropy | PM2.5 reduces vision distance | `entropy(histogram)` |
| **Contrast** | Grayscale std deviation | Pollution creates uniform, low-contrast images | `(std/80)*100` |
| **Brightness** | Mean pixel value | Extreme values indicate atmospheric conditions | `mean(grayscale)` |
| **Saturation** | HSV S-channel mean | Haze desaturates colors | `mean(HSV_S)` |

### 3️⃣ **pm25_estimator.py** - PM2.5 Calculation (235 lines)

**Purpose:** Convert features to PM2.5 concentration and AQI category

**Class: PM25Estimator**

```python
class PM25Estimator:
    COEFFICIENTS = {
        'haze_weight': 1.5,           # Haze is primary indicator
        'turbidity_weight': 1.2,      # Strong correlation
        'visibility_weight': -0.8,    # Inverse: more visibility = less PM2.5
        'contrast_weight': -0.5,      # Inverse
        'brightness_weight': 0.3,     # Weak correlation
        'saturation_weight': -0.4,    # Inverse
        'base_offset': 20             # Baseline pollution
    }
    
    def estimate(self, features):
        # Apply weighted formula
        # Apply non-linear correction
        # Return PM2.5 value (0-300)
    
    def estimate_with_confidence(self, features):
        # Calculate PM2.5
        # Assess feature quality
        # Calculate confidence score (%)
        # Determine AQI category
        # Return complete results dictionary
    
    def _apply_nonlinear_correction(self, raw_pm25):
        # Prevent unrealistic spikes
        # Use sigmoid-like curve
    
    def get_aqi_category(self, pm25):
        # EPA AQI categories:
        # 0-12: Good (Green)
        # 12-35.4: Moderate (Yellow)
        # 35.4-55.4: Unhealthy for Sensitive (Orange)
        # 55.4-150.4: Unhealthy (Red)
        # 150.4-250.4: Very Unhealthy (Purple)
        # 250.4+: Hazardous (Maroon)
```

**AQI Categories (EPA Standard):**

```
PM2.5 Range    Category                      Color    Health Effect
──────────────────────────────────────────────────────────────────
0-12           Good                          Green    Air quality satisfactory
12-35.4        Moderate                      Yellow   Acceptable risk
35.4-55.4      Unhealthy for Sensitive       Orange   Sensitive groups affected
55.4-150.4     Unhealthy                     Red      Everyone affected
150.4-250.4    Very Unhealthy                Purple   Serious health effects
250.4+         Hazardous                     Maroon   Emergency conditions
```

### 4️⃣ **visualization.py** - Graph Generation (331 lines)

**Purpose:** Create 4 types of visualizations

**Class: PM25Visualizer**

```python
class PM25Visualizer:
    def __init__(self, results_folder):
        # Initialize with output directory
    
    def create_heatmap(self, image_path, pm25_value, output_name):
        # Load original image
        # Apply color map (JET or custom)
        # Color intensity = PM2.5 level (red=high, blue=low)
        # Add title, scale, timestamp
        # Save as PNG
        # Return file path
    
    def create_before_after(self, image_path, output_name):
        # Left: Original image (Before)
        # Right: CLAHE enhanced image (After)
        # Show what pollution-free image would look like
        # Side-by-side comparison
    
    def create_timeseries_graph(self, pm25_value, output_name):
        # Read historical data from CSV
        # Generate 24-hour trend
        # Plot line graph
        # Mark current measurement
        # Add AQI background bands
        # Save and return path
    
    def create_feature_chart(self, features, output_name):
        # Create bar chart of 6 features
        # Color code by severity
        # Show contribution to PM2.5
        # Add value labels
        # Save and return path
```

**Visualization Details:**

```
1. HEATMAP VISUALIZATION
   ├─ Input: Satellite image + PM2.5 value
   ├─ Process:
   │  ├─ Convert image to HSV for better color mapping
   │  ├─ Normalize PM2.5 to 0-1 scale
   │  ├─ Apply color gradient:
   │  │  ├─ 0 PM2.5 → Blue (clear)
   │  │  ├─ 150 PM2.5 → Red (polluted)
   │  │  └─ 300+ PM2.5 → Dark Red (hazardous)
   │  ├─ Apply color map to entire image
   │  ├─ Add text annotations
   │  │  ├─ PM2.5 value
   │  │  ├─ Timestamp
   │  │  ├─ Location indicator
   │  │  └─ Color scale
   │  └─ Save high-quality PNG
   └─ Output: heatmap_timestamp.png

2. BEFORE/AFTER VISUALIZATION
   ├─ Input: Original satellite image
   ├─ Process:
   │  ├─ Left side: Original image
   │  ├─ Right side: CLAHE enhancement
   │  │  ├─ CLAHE = Contrast Limited Adaptive Histogram Equalization
   │  │  ├─ Increases contrast locally
   │  │  ├─ Simulates clearer atmosphere
   │  │  └─ Shows potential improvement
   │  ├─ Add middle divider line
   │  ├─ Label "Before" and "After"
   │  └─ Save side-by-side image
   └─ Output: before_after_timestamp.png

3. TIME SERIES GRAPH
   ├─ Input: Current PM2.5 + historical data (CSV)
   ├─ Process:
   │  ├─ Read data/pm25_history.csv
   │  ├─ Generate hourly data points for 24 hours
   │  ├─ Create line plot:
   │  │  ├─ X-axis: Hours (0-24)
   │  │  ├─ Y-axis: PM2.5 µg/m³
   │  │  ├─ Blue line: Trend
   │  │  └─ Red star: Current measurement
   │  ├─ Add background bands for AQI levels
   │  ├─ Add grid lines
   │  ├─ Add legend
   │  ├─ Add title and labels
   │  └─ Save as PNG
   └─ Output: timeseries_timestamp.png

4. FEATURE ANALYSIS CHART
   ├─ Input: 6 feature values
   ├─ Process:
   │  ├─ Create bar chart with 6 bars (one per feature)
   │  ├─ Color code:
   │  │  ├─ Green: Low contribution
   │  │  ├─ Yellow: Medium contribution
   │  │  └─ Red: High contribution
   │  ├─ Add feature names and values
   │  ├─ Add y-axis label: "Feature Score"
   │  ├─ Add title: "Atmospheric Feature Analysis"
   │  ├─ Sort by importance
   │  └─ Save as PNG
   └─ Output: features_timestamp.png
```

---

## 🔄 HOW EACH COMPONENT WORKS

### Image Processing Pipeline (Detailed)

```
STEP 1: IMAGE INPUT
┌──────────────────┐
│ Satellite Image  │ (JPG, PNG, TIFF, etc.)
│ Any resolution   │ (50x50 to 4000x4000 pixels)
└──────────────────┘
          ⬇️

STEP 2: PREPROCESSING
┌──────────────────────────────────────┐
│ 1. Load image file using OpenCV      │
│ 2. Resize to 800x600 (standardize)   │
│ 3. Convert RGB → Grayscale           │
│    └─ For edge detection & analysis  │
│ 4. Convert RGB → HSV                 │
│    └─ For saturation analysis        │
│ 5. Apply Gaussian blur (5x5)         │
│    └─ Reduce noise                   │
└──────────────────────────────────────┘
          ⬇️

STEP 3: FEATURE EXTRACTION (6 Features)
┌────────────────────────────────────────────────┐
│ From preprocessed image, extract:              │
│ 1. Haze Score      ← Edge sharpness            │
│ 2. Turbidity       ← Dark channel              │
│ 3. Visibility      ← Histogram entropy         │
│ 4. Contrast        ← Pixel std deviation       │
│ 5. Brightness      ← Mean pixel value          │
│ 6. Saturation      ← HSV S-channel             │
└────────────────────────────────────────────────┘
          ⬇️

STEP 4: PM2.5 CALCULATION
┌────────────────────────────────────────┐
│ Apply weighted formula:                │
│ PM2.5 = 20 + 1.5×H + 1.2×T            │
│         - 0.8×V - 0.5×C               │
│         + 0.3×B - 0.4×S               │
│                                        │
│ Apply non-linear correction            │
│ Clamp to 0-300 range                   │
└────────────────────────────────────────┘
          ⬇️

STEP 5: AQI CLASSIFICATION
┌────────────────────────────────────┐
│ Compare PM2.5 to EPA categories:  │
│ • Good (0-12)                     │
│ • Moderate (12-35.4)              │
│ • Unhealthy for Sensitive (35-55) │
│ • Unhealthy (55-150)              │
│ • Very Unhealthy (150-250)        │
│ • Hazardous (250+)                │
│                                   │
│ Assign:                           │
│ • Category name                   │
│ • Color code                      │
│ • Health advice                   │
└────────────────────────────────────┘
          ⬇️

STEP 6: VISUALIZATION
┌──────────────────────────────────────┐
│ Create 4 output images:              │
│ 1. Heatmap (color intensity map)    │
│ 2. Before/After (CLAHE comparison)  │
│ 3. Time Series (24-hour trend)      │
│ 4. Feature Chart (bar chart)        │
└──────────────────────────────────────┘
          ⬇️

OUTPUT
┌────────────────────────────┐
│ • PM2.5 value (µg/m³)     │
│ • AQI category & color    │
│ • Health advice           │
│ • Confidence score (%)    │
│ • All 4 visualizations    │
│ • Feature breakdown       │
└────────────────────────────┘
```

### Web Request/Response Flow

```
BROWSER                              SERVER
────────────────────────────────────────────

User loads page
GET / ───────────────────────→ Flask app.py
                               ├─ Check route '/'
                               ├─ Render templates/index.html
                               └─ Return HTML
                          ←─── HTML + CSS + JS

[HTML Page Displayed]

User selects image
[Image selected in form]

User clicks "Analyze"
JavaScript POST ──────────→ /analyze endpoint
multipart/form-data          File: satellite_image
                             
                               app.py receives request
                               ├─ Check file valid
                               ├─ Save to static/uploads/
                               ├─ Create ImageAnalyzer
                               ├─ Extract features
                               ├─ Create PM25Estimator
                               ├─ Calculate PM2.5
                               ├─ Create PM25Visualizer
                               ├─ Generate 4 images
                               ├─ Build JSON response
                               ├─ Return URLs
                          ←─── JSON response
                               {
                                 pm25: 86.02,
                                 aqi_category: "Unhealthy",
                                 images: {...},
                                 ...
                               }

JavaScript receives response
├─ Update DOM with results
├─ Display PM2.5 value
├─ Show AQI category + color
├─ Display 4 images
└─ Show feature chart

[Results displayed to user]
```

---

## 📈 DATA FLOW DIAGRAM

```
┌──────────────────────────────────────────────────────────────────┐
│                     USER INTERACTION LAYER                        │
│  (Browser: HTML/CSS/JavaScript - index.html)                    │
└──────────────────────────────────────────────────────────────────┘
                             ⬆️ ⬇️
                          HTTP/JSON
                             ⬆️ ⬇️
┌──────────────────────────────────────────────────────────────────┐
│                     WEB SERVER LAYER                              │
│  Flask Application (app.py)                                      │
│  ├─ Route: GET  /  → index.html                                  │
│  ├─ Route: POST /analyze → Process image                         │
│  ├─ Route: GET  /about → App info                                │
│  └─ Route: GET  /health → Status check                           │
└──────────────────────────────────────────────────────────────────┘
                             ⬇️
                    File I/O Operations
                             ⬇️
┌──────────────────────────────────────────────────────────────────┐
│                  IMAGE ANALYSIS LAYER                             │
│  ImageAnalyzer (image_analysis.py)                               │
│                                                                  │
│  Input: Satellite image file (static/uploads/)                  │
│                                                                  │
│  Processing:                                                     │
│  ├─ Load image (OpenCV)                                          │
│  ├─ Preprocess:                                                  │
│  │  ├─ Resize to 800x600                                         │
│  │  ├─ Convert BGR → Grayscale                                   │
│  │  └─ Convert BGR → HSV                                         │
│  ├─ Extract 6 features:                                          │
│  │  ├─ Haze Score (Laplacian variance)                           │
│  │  ├─ Turbidity (dark channel)                                  │
│  │  ├─ Visibility (histogram entropy)                            │
│  │  ├─ Contrast (std deviation)                                  │
│  │  ├─ Brightness (mean value)                                   │
│  │  └─ Saturation (HSV S-channel)                                │
│                                                                  │
│  Output: Feature dictionary                                      │
│  {                                                               │
│    'haze_score': 65.3,      (0-100)                              │
│    'turbidity': 48.2,       (0-100)                              │
│    'visibility': 72.1,      (0-100)                              │
│    'contrast': 55.8,        (0-100)                              │
│    'brightness': 145.6,     (0-255)                              │
│    'saturation': 110.2      (0-255)                              │
│  }                                                               │
└──────────────────────────────────────────────────────────────────┘
                             ⬇️
                      Function Call
                             ⬇️
┌──────────────────────────────────────────────────────────────────┐
│                 PM2.5 ESTIMATION LAYER                            │
│  PM25Estimator (pm25_estimator.py)                               │
│                                                                  │
│  Input: Feature dictionary                                       │
│                                                                  │
│  Processing:                                                     │
│  ├─ Apply weighted formula:                                      │
│  │  PM2.5 = 20 + 1.5×H + 1.2×T - 0.8×V - 0.5×C + 0.3×B - 0.4×S │
│  ├─ Apply non-linear correction                                  │
│  ├─ Clamp to 0-300 range                                         │
│  ├─ Calculate confidence score                                   │
│  ├─ Determine AQI category                                       │
│  │  (Good/Moderate/Unhealthy/etc.)                               │
│  └─ Get health advice                                            │
│                                                                  │
│  Output: Results dictionary                                      │
│  {                                                               │
│    'pm25': 86.02,                   (µg/m³)                      │
│    'confidence': 82,                (%)                          │
│    'aqi_category': 'Unhealthy',    (string)                      │
│    'aqi_color': '#FF0000',         (hex)                         │
│    'health_advice': 'Everyone...'  (string)                      │
│  }                                                               │
└──────────────────────────────────────────────────────────────────┘
                             ⬇️
                      Function Call
                             ⬇️
┌──────────────────────────────────────────────────────────────────┐
│               VISUALIZATION LAYER                                 │
│  PM25Visualizer (visualization.py)                               │
│                                                                  │
│  Input: Original image, PM2.5, features                          │
│                                                                  │
│  Processing - Generate 4 visualizations:                         │
│                                                                  │
│  1. HEATMAP                                                      │
│     ├─ Apply color map to image                                  │
│     ├─ Red = high PM2.5, Blue = low PM2.5                        │
│     └─ Save to static/results/heatmap_timestamp.png              │
│                                                                  │
│  2. BEFORE/AFTER                                                 │
│     ├─ Left: Original image                                      │
│     ├─ Right: CLAHE enhanced image                               │
│     └─ Save to static/results/before_after_timestamp.png         │
│                                                                  │
│  3. TIME SERIES                                                  │
│     ├─ Read historical data (CSV)                                │
│     ├─ Plot 24-hour trend                                        │
│     ├─ Mark current measurement                                  │
│     └─ Save to static/results/timeseries_timestamp.png           │
│                                                                  │
│  4. FEATURE CHART                                                │
│     ├─ Create bar chart of 6 features                            │
│     ├─ Color code by severity                                    │
│     └─ Save to static/results/features_timestamp.png             │
│                                                                  │
│  Output: File paths to 4 images                                  │
└──────────────────────────────────────────────────────────────────┘
                             ⬇️
                    Return to Flask app.py
                             ⬇️
┌──────────────────────────────────────────────────────────────────┐
│                      RESPONSE LAYER                               │
│  Build JSON response with:                                       │
│  - PM2.5 value                                                   │
│  - AQI category & color                                          │
│  - Health advice                                                 │
│  - Feature breakdown                                             │
│  - Image file paths/URLs                                         │
│  - Confidence score                                              │
│  - Timestamp                                                     │
│                                                                  │
│  Return JSON to browser via HTTP                                 │
└──────────────────────────────────────────────────────────────────┘
                             ⬆️
                          HTTP/JSON
                             ⬆️
┌──────────────────────────────────────────────────────────────────┐
│                  BROWSER PRESENTATION LAYER                       │
│  JavaScript processes response:                                  │
│  ├─ Display PM2.5 value (large text)                             │
│  ├─ Show AQI category with color                                 │
│  ├─ Display health advice                                        │
│  ├─ Show 4 generated images                                      │
│  └─ Display feature chart                                        │
│                                                                  │
│  User sees complete analysis results                             │
└──────────────────────────────────────────────────────────────────┘
```

---

## 🚀 DEPLOYMENT DETAILS

### Railway Deployment Configuration

**Files Created for Deployment:**

#### 1. **Procfile** (Simple format)
```
web: gunicorn --bind 0.0.0.0:${PORT:-8000} app:app
```
- Tells Railway how to start the application
- `web`: This is a web process
- `gunicorn`: WSGI server (production-ready)
- `--bind 0.0.0.0:${PORT}`: Listen on all interfaces, use PORT env variable
- `app:app`: Import Flask app from app.py module

#### 2. **railway.json** (NIXPACKS configuration)
```json
{
  "build": {"builder": "NIXPACKS"},
  "deploy": {
    "startCommand": "gunicorn --bind 0.0.0.0:${PORT:-8000} app:app",
    "restartPolicyType": "on_failure",
    "restartPolicyMaxRetries": 5
  },
  "services": {"web": {"public": true}}
}
```

#### 3. **runtime.txt** (Python version)
```
python-3.11.7
```
- Specifies which Python version to use
- Railway automatically detects from this file

#### 4. **requirements.txt** (Dependencies)
```
Flask==3.0.0
Werkzeug==3.0.1
gunicorn==21.2.0
opencv-python-headless==4.8.1.78  # No display dependencies
numpy==1.26.3
Pillow==11.0.0
pandas==2.1.4
matplotlib==3.8.2
seaborn==0.13.0
python-dateutil==2.8.2
```

### Deployment Process

```
Step 1: Push code to GitHub
├─ git add .
├─ git commit -m "message"
└─ git push origin main

Step 2: Railway auto-detects
├─ Reads Procfile
├─ Reads requirements.txt
├─ Identifies Python project
└─ Triggers build

Step 3: Build phase
├─ Install Python (3.11/3.12/3.13)
├─ Create virtual environment
├─ Install dependencies from requirements.txt
│  ├─ Flask
│  ├─ OpenCV (headless version)
│  ├─ NumPy, Pandas, Matplotlib, Seaborn
│  └─ Gunicorn
└─ Build completes

Step 4: Deploy phase
├─ Start Gunicorn server
├─ Listen on 0.0.0.0:8080 (Railway assigns PORT)
├─ Load Flask app (app:app)
└─ Routes ready for requests

Step 5: Public access
├─ Railway generates domain
├─ Example: https://pm25-project-xyz.railway.app
├─ Service exposed to internet
└─ Users can access from browser
```

### Why opencv-python-headless?

Railway runs Linux servers without display capabilities:
- **opencv-python**: Requires libGL.so.1, X11, display drivers
- **opencv-python-headless**: No display dependencies, works on servers
- Still has all image processing functions (cv2.imread, cv2.resize, etc.)

---

## 📊 COMPLETE EXAMPLE WALKTHROUGH

### Real-World Example: Analyzing a Polluted City Image

```
INPUT IMAGE: satellite_image_delhi.jpg
- Satellite photo of Delhi city
- 2000x1500 pixels
- Shows hazy atmosphere

┌─ Step 1: User uploads image ─┐
│ Browser sends POST to /analyze │
│ File size: 2.3 MB (within limit)│
│ Format: JPG (valid extension)   │
└────────────────────────────────┘
         ⬇️

┌─ Step 2: Save & Preprocess ─┐
│ Save: static/uploads/         │
│       20260123_084317_         │
│       satellite_image_delhi.jpg│
│                               │
│ Preprocess:                   │
│ ├─ Read with OpenCV           │
│ ├─ Resize to 800x600          │
│ ├─ Convert to grayscale       │
│ └─ Convert to HSV             │
└───────────────────────────────┘
         ⬇️

┌─ Step 3: Extract Features ─┐
│                             │
│ Haze Score:                │
│ ├─ Laplacian edge variance: │
│ ├─ High variance (clear): -10│
│ ├─ Low variance (haze): 800 │
│ ├─ In this image: 250       │
│ └─ Haze = 100-(250/1000*100) │
│     = 75 (significant haze)  │
│                             │
│ Turbidity:                  │
│ ├─ Dark channel mean: 110   │
│ ├─ (110/255)*100 = 43       │
│     (moderate scattering)   │
│                             │
│ Visibility:                 │
│ ├─ Histogram entropy: 5.8   │
│ ├─ Normalized: 58%          │
│     (reduced visibility)    │
│                             │
│ Contrast:                   │
│ ├─ Std deviation: 35        │
│ ├─ (35/80)*100 = 43%        │
│     (low contrast)          │
│                             │
│ Brightness:                 │
│ ├─ Mean pixel: 165          │
│     (moderate-bright)       │
│                             │
│ Saturation:                 │
│ ├─ HSV S-channel mean: 85   │
│     (desaturated)           │
│                             │
│ Features: {                 │
│   'haze_score': 75,         │
│   'turbidity': 43,          │
│   'visibility': 58,         │
│   'contrast': 43,           │
│   'brightness': 165,        │
│   'saturation': 85          │
│ }                           │
└─────────────────────────────┘
         ⬇️

┌─ Step 4: Calculate PM2.5 ─┐
│                            │
│ PM2.5 = Base + Weighted    │
│                            │
│ = 20                       │
│ + 1.5×75   (+112.5)        │
│ + 1.2×43   (+51.6)         │
│ - 0.8×58   (-46.4)         │
│ - 0.5×43   (-21.5)         │
│ + 0.3×65   (+19.5)         │
│ - 0.4×33   (-13.2)         │
│                            │
│ Raw PM2.5 = 122.5          │
│                            │
│ Non-linear correction:     │
│ 122.5 in range 50-150      │
│ = 50+(122.5-50)*0.9        │
│ = 115.25                   │
│                            │
│ Final PM2.5 = 115 µg/m³    │
│                            │
│ AQI Category:              │
│ 115 falls in 55.4-150.4    │
│ Category = "Unhealthy"     │
│ Color = Red (#FF0000)      │
│ Health Advice =            │
│ "Everyone may begin to     │
│  experience health effects"│
│                            │
│ Confidence = 78%           │
│ (based on feature quality) │
└────────────────────────────┘
         ⬇️

┌─ Step 5: Generate Visualizations ─┐
│                                    │
│ 1. HEATMAP:                        │
│    ├─ Apply red-blue colormap      │
│    ├─ Red intensity = 115/300      │
│    ├─ Overlay on original image    │
│    └─ Save: heatmap_...png         │
│                                    │
│ 2. BEFORE/AFTER:                   │
│    ├─ Left: Original (hazy)        │
│    ├─ Right: CLAHE enhanced        │
│    │          (clearer)            │
│    └─ Save: before_after_...png    │
│                                    │
│ 3. TIME SERIES:                    │
│    ├─ Plot 24-hour trend           │
│    ├─ Current point: 115 µg/m³     │
│    ├─ 0-12: "Good" band            │
│    ├─ 55-150: "Unhealthy" band     │
│    └─ Save: timeseries_...png      │
│                                    │
│ 4. FEATURE CHART:                  │
│    ├─ 6 bars for 6 features        │
│    ├─ Bar heights: 75, 43, 58, etc.│
│    ├─ Colors: Red (high), Yellow   │
│    └─ Save: features_...png        │
│                                    │
│ All 4 images saved to:             │
│ static/results/                    │
└────────────────────────────────────┘
         ⬇️

┌─ Step 6: Response to Browser ─┐
│                                │
│ JSON Response:                 │
│ {                              │
│   "success": true,             │
│   "pm25": 115,                 │
│   "confidence": 78,            │
│   "aqi_category": "Unhealthy", │
│   "aqi_color": "#FF0000",      │
│   "health_advice": "Everyone..│
│   "features": {                │
│     "haze_score": 75.2,        │
│     "turbidity": 43.1,         │
│     "visibility": 58.3,        │
│     "contrast": 43.8,          │
│     "brightness": 165,         │
│     "saturation": 85.3         │
│   },                           │
│   "images": {                  │
│     "original": "/static/...   │
│     "heatmap": "/static/...    │
│     "before_after": "/static/..│
│     "timeseries": "/static/..  │
│     "features_chart": "/static/│
│   },                           │
│   "timestamp": "2026-01-23..." │
│ }                              │
│                                │
└────────────────────────────────┘
         ⬇️

OUTPUT DISPLAYED TO USER:
┌──────────────────────────────────────┐
│     PM2.5: 115 µg/m³                 │
│     Category: Unhealthy (RED)        │
│                                      │
│     "Everyone may begin to          │
│      experience health effects"     │
│                                      │
│     Confidence: 78%                  │
│                                      │
│  [Heatmap Image]  [Before/After]    │
│  [Time Series Graph] [Feature Chart] │
│                                      │
│     Features:                        │
│     • Haze: 75.2 (High)             │
│     • Turbidity: 43.1 (Medium)      │
│     • Visibility: 58.3 (Low)        │
│     • Contrast: 43.8 (Low)          │
│     • Brightness: 165 (Normal)      │
│     • Saturation: 85.3 (Low)        │
│                                      │
│    [Analyze Another Image]           │
└──────────────────────────────────────┘
```

---

## ✅ COMPLETE TECHNOLOGY STACK SUMMARY

```
┌─────────────────────────────────────────────────────────────────┐
│                    TECHNOLOGY STACK                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│ BACKEND (Server-side Processing)                               │
│ ├─ Language: Python 3.11+                                      │
│ ├─ Framework: Flask 3.0.0 (Web framework)                      │
│ ├─ Image Processing:                                           │
│ │  ├─ OpenCV 4.8.1.78 (computer vision)                        │
│ │  ├─ NumPy 1.26.3 (numerical computing)                       │
│ │  └─ Pillow 11.0.0 (image manipulation)                       │
│ ├─ Data Processing:                                            │
│ │  ├─ Pandas 2.1.4 (CSV/data handling)                         │
│ │  └─ Python-dateutil 2.8.2 (timestamps)                       │
│ ├─ Visualization:                                              │
│ │  ├─ Matplotlib 3.8.2 (graphs/plots)                          │
│ │  └─ Seaborn 0.13.0 (statistical visualization)               │
│ └─ Server:                                                     │
│    ├─ Werkzeug 3.0.1 (WSGI utilities)                          │
│    └─ Gunicorn 21.2.0 (production WSGI server)                 │
│                                                                 │
│ FRONTEND (Client-side UI)                                      │
│ ├─ Language: HTML5 + CSS3 + JavaScript (Vanilla)               │
│ ├─ Architecture: Single Page Application (SPA)                 │
│ ├─ Communication: AJAX (XMLHttpRequest)                        │
│ └─ Data Format: JSON                                           │
│                                                                 │
│ DEPLOYMENT                                                     │
│ ├─ Platform: Railway.app                                      │
│ ├─ Container: NIX packages                                    │
│ ├─ Startup: Procfile (gunicorn)                               │
│ ├─ Python: 3.12/3.13                                          │
│ ├─ Port: 8000 (via $PORT environment variable)                │
│ ├─ Storage: Ephemeral (resets on redeploy)                    │
│ └─ Domain: Auto-generated Railway URL                         │
│                                                                 │
│ VERSION CONTROL                                                │
│ ├─ Repository: GitHub (vsiva763-git/PM25_PROJECT)             │
│ ├─ Branches: main                                              │
│ └─ Commits: Multiple for deployment fixes                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🎓 ACADEMIC RELEVANCE

### Why This Project is Academically Valid

```
✅ ENGINEERING PRINCIPLES:
   ├─ Signal Processing: Image preprocessing, filtering
   ├─ Feature Extraction: Computer vision techniques
   ├─ Algorithm Design: Weighted formula optimization
   ├─ Software Architecture: Modular, layered design
   ├─ Web Development: Full-stack application
   └─ Deployment: Cloud platform integration

✅ NO PROHIBITED TECHNIQUES:
   ├─ No Machine Learning training
   ├─ No Neural Networks
   ├─ No Large Datasets
   ├─ No API dependencies
   └─ No external paid services

✅ COMPLETE DOCUMENTATION:
   ├─ 7+ markdown files (1000+ lines)
   ├─ Code comments throughout
   ├─ Architecture diagrams
   ├─ Formula explanations
   ├─ User guides
   └─ Viva defense guide

✅ DEMONSTRABLE FUNCTIONALITY:
   ├─ Real image inputs
   ├─ Verifiable outputs (PM2.5 values)
   ├─ Professional UI
   ├─ Multiple visualizations
   └─ Deployable to cloud
```

---

## 🎯 CONCLUSION

This is a **complete, production-ready PM2.5 Estimation System** that:

1. **Analyzes satellite images** using pure image processing (no ML)
2. **Extracts 6 atmospheric features** through computer vision
3. **Calculates PM2.5** using weighted formula
4. **Generates visualizations** with Matplotlib/Seaborn
5. **Provides web interface** with Flask
6. **Deploys to Railway** with proper configuration
7. **Includes comprehensive documentation** for academics/viva

The project demonstrates solid understanding of:
- Image processing fundamentals
- Atmospheric science concepts
- Software engineering best practices
- Web application development
- Cloud deployment practices

All code is commented, documented, and ready for final-year submission! ✨
