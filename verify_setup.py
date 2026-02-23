"""
Project Verification Script
Checks if all components are properly installed and configured.
Run this before the final demo to ensure everything works.
"""

import sys
import os

def check_python_version():
    """Check if Python version is 3.9+"""
    version = sys.version_info
    print(f"✓ Python version: {version.major}.{version.minor}.{version.micro}")
    if version.major < 3 or (version.major == 3 and version.minor < 9):
        print("  ⚠ Warning: Python 3.9+ recommended")
        return False
    return True

def check_dependencies():
    """Check if all required packages are installed"""
    print("\nChecking dependencies...")
    required = {
        'flask': 'Flask',
        'cv2': 'OpenCV',
        'numpy': 'NumPy',
        'matplotlib': 'Matplotlib',
        'seaborn': 'Seaborn',
        'pandas': 'Pandas'
    }
    
    all_installed = True
    for module, name in required.items():
        try:
            __import__(module)
            print(f"  ✓ {name} installed")
        except ImportError:
            print(f"  ✗ {name} NOT installed")
            all_installed = False
    
    return all_installed

def check_directories():
    """Check if all required directories exist"""
    print("\nChecking directory structure...")
    required_dirs = [
        'static/uploads',
        'static/results',
        'static/css',
        'templates',
        'data'
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"  ✓ {dir_path}/")
        else:
            print(f"  ✗ {dir_path}/ NOT FOUND")
            all_exist = False
    
    return all_exist

def check_files():
    """Check if all required files exist"""
    print("\nChecking required files...")
    required_files = [
        'app.py',
        'image_analysis.py',
        'pm25_estimator.py',
        'visualization.py',
        'requirements.txt',
        'templates/index.html',
        'static/css/style.css',
        'data/pm25_history.csv'
    ]
    
    all_exist = True
    for file_path in required_files:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            print(f"  ✓ {file_path} ({size} bytes)")
        else:
            print(f"  ✗ {file_path} NOT FOUND")
            all_exist = False
    
    return all_exist

def test_imports():
    """Test if custom modules can be imported"""
    print("\nTesting custom modules...")
    
    try:
        from image_analysis import ImageAnalyzer
        print("  ✓ ImageAnalyzer class imported")
    except Exception as e:
        print(f"  ✗ ImageAnalyzer import failed: {e}")
        return False
    
    try:
        from pm25_estimator import PM25Estimator
        print("  ✓ PM25Estimator class imported")
    except Exception as e:
        print(f"  ✗ PM25Estimator import failed: {e}")
        return False
    
    try:
        from visualization import PM25Visualizer
        print("  ✓ PM25Visualizer class imported")
    except Exception as e:
        print(f"  ✗ PM25Visualizer import failed: {e}")
        return False
    
    return True

def main():
    """Run all verification checks"""
    print("="*60)
    print("PM2.5 ESTIMATION SYSTEM - VERIFICATION SCRIPT")
    print("="*60)
    print()
    
    results = []
    
    # Run checks
    results.append(("Python Version", check_python_version()))
    results.append(("Dependencies", check_dependencies()))
    results.append(("Directory Structure", check_directories()))
    results.append(("Required Files", check_files()))
    results.append(("Module Imports", test_imports()))
    
    # Summary
    print("\n" + "="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{name}: {status}")
        if not passed:
            all_passed = False
    
    print("="*60)
    
    if all_passed:
        print("\n🎉 ALL CHECKS PASSED!")
        print("Your project is ready to run!")
        print("\nTo start the application:")
        print("  python app.py")
        print("\nThen open: http://127.0.0.1:5000")
    else:
        print("\n⚠ SOME CHECKS FAILED")
        print("Please fix the issues above before running the application.")
        print("\nTo install dependencies:")
        print("  pip install -r requirements.txt")
    
    print()

if __name__ == '__main__':
    main()
