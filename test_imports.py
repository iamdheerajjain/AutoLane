#!/usr/bin/env python3
"""
Test script to verify all imports work correctly
"""

import sys
import os

def test_imports():
    """Test all required imports"""
    print("Testing imports...")
    
    try:
        import streamlit as st
        print(f"✓ Streamlit {st.__version__}")
    except ImportError as e:
        print(f"✗ Streamlit: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✓ NumPy {np.__version__}")
    except ImportError as e:
        print(f"✗ NumPy: {e}")
        return False
    
    try:
        import tensorflow as tf
        print(f"✓ TensorFlow {tf.__version__}")
    except ImportError as e:
        print(f"✗ TensorFlow: {e}")
        return False
    
    try:
        from tensorflow import keras
        print(f"✓ Keras {keras.__version__}")
    except ImportError as e:
        print(f"✗ Keras: {e}")
        return False
    
    try:
        import cv2
        print(f"✓ OpenCV {cv2.__version__}")
    except ImportError as e:
        print(f"✗ OpenCV: {e}")
        return False
    
    try:
        import matplotlib.pyplot as plt
        print(f"✓ Matplotlib {plt.matplotlib.__version__}")
    except ImportError as e:
        print(f"✗ Matplotlib: {e}")
        return False
    
    try:
        import pandas as pd
        print(f"✓ Pandas {pd.__version__}")
    except ImportError as e:
        print(f"✗ Pandas: {e}")
        return False
    
    try:
        import plotly
        print(f"✓ Plotly {plotly.__version__}")
    except ImportError as e:
        print(f"✗ Plotly: {e}")
        return False
    
    try:
        from PIL import Image
        print(f"✓ Pillow")
    except ImportError as e:
        print(f"✗ Pillow: {e}")
        return False
    
    try:
        import psutil
        print(f"✓ psutil")
    except ImportError as e:
        print(f"✗ psutil: {e}")
        return False
    
    # Test custom modules
    try:
        from training import CustomLaneTrainer
        print("✓ CustomLaneTrainer")
    except ImportError as e:
        print(f"✗ CustomLaneTrainer: {e}")
        return False
    
    try:
        from prediction import CustomLanePredictor
        print("✓ CustomLanePredictor")
    except ImportError as e:
        print(f"✗ CustomLanePredictor: {e}")
        return False
    
    print("\n✓ All imports successful!")
    return True

def test_model_loading():
    """Test model loading if model exists"""
    print("\nTesting model loading...")
    
    if not os.path.exists('checkpoints'):
        print("ℹ️ No checkpoints directory found")
        return True
    
    model_files = [f for f in os.listdir('checkpoints') if f.endswith('.h5')]
    
    if not model_files:
        print("ℹ️ No model files found")
        return True
    
    print(f"Found {len(model_files)} model file(s)")
    
    for model_file in model_files:
        model_path = f"checkpoints/{model_file}"
        print(f"Testing {model_file}...")
        
        try:
            from prediction import CustomLanePredictor
            predictor = CustomLanePredictor(model_path)
            
            if predictor.model is not None:
                print(f"✓ {model_file} loaded successfully")
            else:
                print(f"✗ {model_file} failed to load")
                return False
                
        except Exception as e:
            print(f"✗ {model_file} error: {e}")
            return False
    
    print("✓ All models loaded successfully!")
    return True

if __name__ == "__main__":
    print("AutoLane Import Test")
    print("=" * 50)
    
    success = test_imports()
    
    if success:
        test_model_loading()
    
    print("\n" + "=" * 50)
    if success:
        print("✓ All tests passed! The application should work correctly.")
    else:
        print("✗ Some tests failed. Please check the errors above.")
        sys.exit(1)
