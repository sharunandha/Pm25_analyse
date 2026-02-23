# 🛰️ PM2.5 Estimation System

**High-Resolution PM2.5 Estimation and Visualization from Satellite Images Using Image Processing Techniques**

A software-based final-year engineering project that estimates PM2.5 air pollution levels from satellite imagery using image processing techniques - **NO machine learning training required!**

---

## 📋 Project Overview

This system analyzes satellite images to estimate PM2.5 (Particulate Matter 2.5) concentration using atmospheric indicators such as haze, contrast, brightness, and saturation. It provides comprehensive visualizations including heatmaps, time-series graphs, and before/after comparisons.

### Key Features

✅ **No Training Required** - Uses formula-based estimation, not ML models  
✅ **Fully Offline** - Works completely without internet connection  
✅ **Web Interface** - Clean, professional web application  
✅ **Real-time Analysis** - Instant results from uploaded images  
✅ **Multiple Visualizations** - Heatmaps, graphs, and comparisons  
✅ **AQI Classification** - Automatic air quality categorization  
✅ **Historical Tracking** - Stores and displays PM2.5 trends  

---

## 🎯 Academic Validity

This project is designed for final-year engineering submissions with:

- **Clear scientific methodology** based on atmospheric research
- **Reproducible results** using image processing algorithms
- **No false claims** about ML/AI training
- **Viva-proof explanation** focusing on cost-effectiveness and software approach
- **Complete documentation** and commented code

---

## 🏗️ Project Structure

```
pm25_project/
│
├── app.py                      # Flask web application (main entry point)
├── image_analysis.py           # Atmospheric feature extraction
├── pm25_estimator.py           # PM2.5 calculation from features
├── visualization.py            # Graph and heatmap generation
├── requirements.txt            # Python dependencies
│
├── static/
│   ├── uploads/                # Uploaded satellite images
│   ├── results/                # Generated visualizations
│   └── css/
│       └── style.css           # Web interface styling
│
├── templates/
│   └── index.html              # Main web page
│
├── data/
│   └── pm25_history.csv        # Historical PM2.5 records
│
└── README.md                   # This file
```

---

## 🧠 Technical Approach

### Image Processing Pipeline

1. **Image Preprocessing**
   - Resize to standard dimensions (800x600)
   - Convert to grayscale and HSV color space
   - Apply Gaussian blur for noise reduction

2. **Feature Extraction**
   - **Haze Score**: Calculated using Laplacian edge detection variance
   - **Turbidity**: Dark channel prior for atmospheric scattering
   - **Visibility**: Histogram entropy analysis
   - **Contrast**: Standard deviation of pixel intensities
   - **Brightness**: Mean pixel value
   - **Saturation**: Average HSV saturation channel

3. **PM2.5 Estimation Formula**
   ```
   PM2.5 = α × haze + β × turbidity - γ × visibility 
           - δ × contrast + ε × brightness - ζ × saturation + offset
   ```
   
   Where α, β, γ, δ, ε, ζ are empirically calibrated coefficients

4. **Visualization Generation**
   - Spatial heatmap using OpenCV color mapping
   - Time-series graph with Matplotlib
   - Before/after using CLAHE enhancement
   - Feature bar charts with Seaborn

---

## 💻 Installation & Setup

### Prerequisites

- **Python 3.9 or higher**
- **Visual Studio Code** (recommended)
- **Windows/Linux/macOS**

### Step 1: Open Project in VS Code

1. Open Visual Studio Code
2. Click **File** → **Open Folder**
3. Navigate to `pm25_project` folder
4. Click **Select Folder**

### Step 2: Create Virtual Environment

Open terminal in VS Code (`Ctrl + `` ` or View → Terminal`) and run:

**Windows:**
```powershell
python -m venv venv
.\venv\Scripts\activate
```

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- Flask (web framework)
- OpenCV (image processing)
- NumPy (numerical operations)
- Matplotlib & Seaborn (visualizations)
- Pandas (data handling)

### Step 4: Run the Application

```bash
python app.py
```

You should see:
```
PM2.5 ESTIMATION SYSTEM
High-Resolution PM2.5 Estimation from Satellite Images
============================================================

Starting Flask application...
Server will be available at: http://127.0.0.1:5000
```

### Step 5: Open Web Application

1. Open your web browser
2. Go to: `http://127.0.0.1:5000`
3. You'll see the PM2.5 Estimation System interface

---

## 🚀 How to Use

### Using the Web Application

1. **Upload Image**
   - Click "Choose an image file..."
   - Select any satellite/aerial image (JPG, PNG, TIFF)
   - Any image works - the system will analyze atmospheric conditions

2. **Analyze**
   - Click "🔍 Analyze Image" button
   - Wait for processing (typically 5-10 seconds)

3. **View Results**
   - **PM2.5 Value**: Main concentration estimate in µg/m³
   - **AQI Category**: Color-coded air quality index
   - **Atmospheric Features**: All extracted indicators
   - **Heatmap**: Spatial distribution of PM2.5
   - **Before/After**: Pollution vs. clear comparison
   - **Time Series**: Historical trend graph
   - **Original Image**: Uploaded satellite image

### Sample Test Images

For testing, you can use:
- Google Earth screenshots
- Satellite images from NASA Worldview (https://worldview.earthdata.nasa.gov)
- Aerial photographs
- Any outdoor landscape images

---

## 📊 Output Descriptions

### 1. PM2.5 Numerical Value
- Range: 0-300 µg/m³
- Realistic estimates based on image analysis
- Displayed with confidence percentage

### 2. AQI Categories
- **Good** (0-12): Green
- **Moderate** (12-35.4): Yellow
- **Unhealthy for Sensitive Groups** (35.4-55.4): Orange
- **Unhealthy** (55.4-150.4): Red
- **Very Unhealthy** (150.4-250.4): Purple
- **Hazardous** (250.4+): Maroon

### 3. Spatial Heatmap
- Blue areas: Low PM2.5
- Green-Yellow: Moderate PM2.5
- Red areas: High PM2.5

### 4. Time Series Graph
- X-axis: Date/time of analysis
- Y-axis: PM2.5 concentration
- Color bands show AQI categories
- Stores last 30 measurements

### 5. Before/After Comparison
- Left: Current atmospheric conditions
- Right: Simulated clear conditions (contrast-enhanced)

---

## 🔬 Technical Details for Viva

### Q: How does the system work without ML training?

**Answer**: 
"The system uses classical image processing techniques to extract atmospheric indicators that naturally correlate with PM2.5 levels. We calculate features like haze (using edge detection variance), atmospheric turbidity (dark channel prior), and visibility (histogram entropy). These are combined using an empirically-calibrated weighted formula to estimate PM2.5 concentration. This approach is based on established atmospheric science research showing that pollution affects image clarity and color characteristics in measurable ways."

### Q: Why not use machine learning?

**Answer**:
"Our approach offers several advantages: (1) No need for large labeled datasets, (2) Computationally lightweight - runs on any computer, (3) Fully transparent and explainable results, (4) No dependency on cloud services or GPUs, (5) More suitable for resource-constrained educational projects. The formula-based method provides consistent, interpretable results ideal for understanding the underlying atmospheric physics."

### Q: How accurate is the estimation?

**Answer**:
"The system provides relative PM2.5 estimates useful for trend analysis and spatial distribution visualization. For absolute accuracy, ground-truth calibration with reference air quality stations would be needed. However, the extracted features (haze, contrast, visibility) have well-documented correlations with air quality in atmospheric research literature. The system is designed for educational demonstration and understanding of image-based environmental monitoring."

### Q: What are the key innovations?

**Answer**:
"(1) Multi-feature fusion combining six atmospheric indicators, (2) Spatial visualization through PM2.5 heatmaps, (3) Temporal tracking with historical graphs, (4) User-friendly web interface for accessibility, (5) Completely offline operation with no external dependencies, (6) Cost-effective software-only solution replacing expensive hardware sensors."

---

## 🛠️ Troubleshooting

### Issue: Port 5000 already in use

**Solution**:
```python
# In app.py, change the last line to:
app.run(debug=True, host='127.0.0.1', port=5001)
```
Then access at `http://127.0.0.1:5001`

### Issue: Module not found error

**Solution**:
```bash
# Ensure virtual environment is activated
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

### Issue: OpenCV import error

**Solution**:
```bash
pip uninstall opencv-python
pip install opencv-python-headless==4.8.1.78
```

### Issue: Images not displaying

**Solution**:
- Check browser console for errors (F12)
- Ensure `static/uploads/` and `static/results/` directories exist
- Clear browser cache and refresh

---

## 📚 Dependencies Explained

| Package | Purpose | Version |
|---------|---------|---------|
| Flask | Web framework for the application | 3.0.0 |
| opencv-python | Image processing and analysis | 4.8.1.78 |
| numpy | Numerical computations | 1.24.3 |
| matplotlib | Graph plotting | 3.8.2 |
| seaborn | Statistical visualizations | 0.13.0 |
| pandas | Data management for history | 2.1.4 |

---

## 📖 Code Modules Overview

### app.py
- Flask routes and web server
- File upload handling
- Coordinates analysis pipeline
- Serves HTML interface

### image_analysis.py
- `ImageAnalyzer` class
- Loads and preprocesses images
- Extracts 6 atmospheric features
- Returns feature dictionary

### pm25_estimator.py
- `PM25Estimator` class
- Applies weighted formula to features
- Calculates PM2.5 concentration
- Provides AQI classification

### visualization.py
- `PM25Visualizer` class
- Creates heatmap overlays
- Generates time-series graphs
- Produces before/after comparisons

---

## 🎓 Academic References

This project is based on concepts from:

1. **Atmospheric Remote Sensing**
   - Relationship between visibility and particulate matter
   - Dark channel prior for atmospheric scattering

2. **Image Quality Metrics**
   - Edge detection for haze quantification
   - Histogram analysis for visibility estimation

3. **Air Quality Index Standards**
   - EPA AQI categorization
   - PM2.5 health impact thresholds

---

## ⚖️ Limitations & Scope

### Current Limitations
- Relative estimates, not absolute measurements
- Requires calibration for specific geographic regions
- Weather conditions (rain, clouds) affect results
- Single-image analysis (no temporal comparison)

### Future Enhancements
- Integration with ground station data for calibration
- Multi-temporal analysis for trend detection
- Additional pollutant estimation (PM10, NO2)
- Mobile application development
- Real-time satellite feed integration

---

## 👨‍💻 Author & License

**Project**: PM2.5 Estimation System  
**Type**: Final Year Engineering Project  
**Year**: 2026  
**Purpose**: Academic demonstration of image processing for environmental monitoring  

**Note**: This is an educational project. For production air quality monitoring, use calibrated sensors and validated models.

---

## 📞 Support

For issues or questions:
1. Check the Troubleshooting section above
2. Review code comments in Python files
3. Verify all dependencies are installed correctly
4. Ensure Python 3.9+ is being used

---

## ✅ Project Checklist

Before submission/demo, verify:

- [ ] All dependencies installed successfully
- [ ] Flask server starts without errors
- [ ] Web interface loads at http://127.0.0.1:5000
- [ ] Image upload works
- [ ] PM2.5 estimation completes
- [ ] All visualizations generate correctly
- [ ] Results display properly
- [ ] Historical data saves to CSV
- [ ] Code is well-commented
- [ ] README is complete

---

## 🎯 Final Notes

This project demonstrates that **effective environmental monitoring solutions can be built using classical image processing techniques** without requiring expensive hardware, large datasets, or complex ML training. It's a perfect example of **software-based innovation** solving real-world problems cost-effectively.

**Good luck with your project presentation! 🚀**

---

**Remember**: The strength of this project lies in its **simplicity, transparency, and practical applicability** - emphasize these points during your viva!
