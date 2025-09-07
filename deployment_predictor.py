"""
Deployment-Specific Lane Predictor
Uses the DeploymentModelLoader for better compatibility in deployed environments
"""

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
from datetime import datetime
import matplotlib.pyplot as plt
from deployment_model_loader import DeploymentModelLoader

class DeploymentLanePredictor:
    def __init__(self, model_path, input_shape=(160, 320, 3)):
        self.input_shape = input_shape
        self.model_loader = DeploymentModelLoader()
        self.model = None
        self.load_successful = False
        
        # Try to load the model
        self.load_model(model_path)
    
    def load_model(self, model_path):
        """Load model using the deployment model loader"""
        if not os.path.exists(model_path):
            print(f"Model not found: {model_path}")
            return False
        
        print(f"Loading model from: {model_path}")
        
        # Use the deployment model loader
        success = self.model_loader.load_model(model_path, self.input_shape)
        
        if success:
            self.model = self.model_loader.model
            self.load_successful = True
            print("✓ Model loaded successfully!")
            
            # Test the model
            pred_success, pred_msg = self.model_loader.test_model_prediction(self.input_shape)
            if pred_success:
                print(f"✓ Model prediction test passed: {pred_msg}")
            else:
                print(f"⚠️ Model prediction test failed: {pred_msg}")
            
            return True
        else:
            print("✗ Model loading failed!")
            print("Error details:")
            for error in self.model_loader.error_details:
                print(f"  - {error}")
            return False
    
    def preprocess_image(self, img):
        """Preprocess image for model input"""
        # Resize image
        img_resized = cv2.resize(img, (self.input_shape[1], self.input_shape[0]))
        
        # Normalize to [0, 1]
        img_normalized = img_resized.astype(np.float32) / 255.0
        
        # Add batch dimension
        img_batch = np.expand_dims(img_normalized, axis=0)
        
        return img_batch
    
    def predict_lanes(self, img):
        """Predict lanes in image"""
        if self.model is None:
            raise ValueError("Model not loaded")
        
        # Preprocess image
        img_processed = self.preprocess_image(img)
        
        # Predict
        prediction = self.model.predict(img_processed, verbose=0)
        
        # Get lane mask (single channel output)
        lane_mask = prediction[0, :, :, 0]
        
        # Convert to binary mask
        lane_mask_binary = (lane_mask > 0.5).astype(np.uint8) * 255
        
        return lane_mask_binary, lane_mask
    
    def visualize_prediction(self, img, lane_mask, confidence_threshold=0.5):
        """Visualize prediction on original image"""
        # Resize prediction to original image size
        lane_mask_resized = cv2.resize(lane_mask, (img.shape[1], img.shape[0]))
        
        # Create colored mask
        colored_mask = np.zeros_like(img)
        colored_mask[lane_mask_resized > 128] = [0, 255, 0]  # Green for lanes
        
        # Combine with original image
        result = cv2.addWeighted(img, 0.7, colored_mask, 0.3, 0)
        
        # Add confidence information
        confidence = np.mean(lane_mask_resized) / 255.0
        cv2.putText(result, f'Confidence: {confidence:.2f}', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Add model info
        model_info = self.get_model_info()
        if model_info.get('is_fallback_model'):
            cv2.putText(result, 'AutoLane (Fallback Model)', (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        else:
            cv2.putText(result, 'AutoLane Enhanced', (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        return result
    
    def get_model_info(self):
        """Get detailed model information"""
        if self.model_loader:
            return self.model_loader.get_model_info()
        return {'loaded': False, 'error': 'No model loader available'}
    
    def is_model_loaded(self):
        """Check if model is successfully loaded"""
        return self.load_successful and self.model is not None

def test_deployment_predictor():
    """Test the deployment predictor"""
    model_path = "checkpoints/custom_lane_model_best.h5"
    
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return
    
    print("Testing Deployment Lane Predictor...")
    print("-" * 50)
    
    # Create predictor
    predictor = DeploymentLanePredictor(model_path)
    
    if predictor.is_model_loaded():
        print("✓ Deployment predictor created successfully!")
        
        # Get model info
        info = predictor.get_model_info()
        print(f"Model info: {info}")
        
        # Test with dummy image
        dummy_img = np.random.randint(0, 255, (160, 320, 3), dtype=np.uint8)
        try:
            lane_mask, confidence_map = predictor.predict_lanes(dummy_img)
            result = predictor.visualize_prediction(dummy_img, lane_mask)
            print("✓ Prediction test successful!")
            print(f"Lane mask shape: {lane_mask.shape}")
            print(f"Result shape: {result.shape}")
        except Exception as e:
            print(f"✗ Prediction test failed: {e}")
    else:
        print("✗ Deployment predictor creation failed!")

if __name__ == "__main__":
    test_deployment_predictor()
