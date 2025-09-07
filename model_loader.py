"""
Enhanced Model Loader with Version Compatibility
Handles various TensorFlow/Keras version compatibility issues
"""

import os
import sys
import warnings
import numpy as np
import tensorflow as tf
from tensorflow import keras
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelLoader:
    """Enhanced model loader with compatibility handling"""
    
    def __init__(self):
        self.tf_version = tf.__version__
        try:
            self.keras_version = keras.__version__
        except AttributeError:
            try:
                # For newer TensorFlow versions where Keras is integrated
                self.keras_version = tf.keras.__version__
            except AttributeError:
                # Fallback to TensorFlow version
                self.keras_version = f"integrated-{self.tf_version}"
        self.model = None
        self.load_successful = False
        self.error_details = []
        
        logger.info(f"ModelLoader initialized with TF {self.tf_version}, Keras {self.keras_version}")
    
    def load_model(self, model_path, input_shape=(160, 320, 3)):
        """
        Load model with multiple fallback strategies
        """
        if not os.path.exists(model_path):
            error_msg = f"Model file not found: {model_path}"
            logger.error(error_msg)
            self.error_details.append(error_msg)
            return False
        
        model_size = os.path.getsize(model_path) / (1024*1024)  # MB
        logger.info(f"Loading model: {model_path} ({model_size:.2f} MB)")
        
        # Try multiple loading strategies
        strategies = [
            self._load_standard,
            self._load_with_custom_objects,
            self._load_with_legacy_support,
            self._load_with_compile_false,
            self._load_with_safe_mode,
        ]
        
        for i, strategy in enumerate(strategies, 1):
            try:
                logger.info(f"Trying strategy {i}: {strategy.__name__}")
                success = strategy(model_path)
                if success:
                    self.load_successful = True
                    logger.info(f"✓ Model loaded successfully using {strategy.__name__}")
                    return True
            except Exception as e:
                error_msg = f"Strategy {i} failed: {str(e)}"
                logger.warning(error_msg)
                self.error_details.append(error_msg)
                continue
        
        # All strategies failed
        logger.error("All model loading strategies failed")
        return False
    
    def _load_standard(self, model_path):
        """Standard Keras model loading"""
        self.model = keras.models.load_model(model_path)
        return True
    
    def _load_with_custom_objects(self, model_path):
        """Load with custom objects for compatibility"""
        custom_objects = {
            'tf': tf,
            'Conv2DTranspose': keras.layers.Conv2DTranspose,
            'Conv2D': keras.layers.Conv2D,
            'MaxPooling2D': keras.layers.MaxPooling2D,
            'UpSampling2D': keras.layers.UpSampling2D,
            'BatchNormalization': keras.layers.BatchNormalization,
            'ReLU': keras.layers.ReLU,
            'LeakyReLU': keras.layers.LeakyReLU,
            'Dropout': keras.layers.Dropout,
            'Dense': keras.layers.Dense,
            'Flatten': keras.layers.Flatten,
            'GlobalAveragePooling2D': keras.layers.GlobalAveragePooling2D,
            'concatenate': keras.layers.concatenate,
            'Input': keras.layers.Input,
        }
        
        self.model = keras.models.load_model(
            model_path, 
            compile=False,
            custom_objects=custom_objects
        )
        return True
    
    def _load_with_legacy_support(self, model_path):
        """Load with legacy TensorFlow support"""
        # Try with different compatibility settings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # Set some legacy behavior
            tf.config.experimental.enable_op_determinism()
            
            self.model = keras.models.load_model(
                model_path, 
                compile=False,
                custom_objects={'tf': tf}
            )
        return True
    
    def _load_with_compile_false(self, model_path):
        """Load without compilation"""
        self.model = keras.models.load_model(model_path, compile=False)
        return True
    
    def _load_with_safe_mode(self, model_path):
        """Load with safe mode disabled"""
        self.model = keras.models.load_model(
            model_path, 
            compile=False,
            safe_mode=False
        )
        return True
    
    def test_model_prediction(self, input_shape=(160, 320, 3)):
        """Test if the loaded model can make predictions"""
        if self.model is None:
            return False, "No model loaded"
        
        try:
            # Create dummy input
            dummy_input = np.random.random((1, *input_shape)).astype(np.float32)
            
            # Make prediction
            prediction = self.model.predict(dummy_input, verbose=0)
            
            logger.info(f"Model prediction test successful. Output shape: {prediction.shape}")
            return True, f"Prediction successful. Output shape: {prediction.shape}"
            
        except Exception as e:
            error_msg = f"Model prediction test failed: {str(e)}"
            logger.error(error_msg)
            return False, error_msg
    
    def get_model_info(self):
        """Get detailed model information"""
        if self.model is None:
            return {
                'loaded': False,
                'error_details': self.error_details
            }
        
        try:
            return {
                'loaded': True,
                'model_type': type(self.model).__name__,
                'input_shape': self.model.input_shape if hasattr(self.model, 'input_shape') else 'Unknown',
                'output_shape': self.model.output_shape if hasattr(self.model, 'output_shape') else 'Unknown',
                'num_layers': len(self.model.layers) if hasattr(self.model, 'layers') else 'Unknown',
                'tf_version': self.tf_version,
                'keras_version': self.keras_version,
                'model_summary': str(self.model.summary()) if hasattr(self.model, 'summary') else 'Summary not available'
            }
        except Exception as e:
            return {
                'loaded': True,
                'error': f"Could not get model info: {str(e)}",
                'tf_version': self.tf_version,
                'keras_version': self.keras_version
            }
    
    def create_compatible_model(self, input_shape=(160, 320, 3)):
        """Create a new compatible model if loading fails"""
        logger.info("Creating new compatible model...")
        
        try:
            # Create a simple U-Net model
            inputs = keras.Input(shape=input_shape)
            
            # Encoder
            conv1 = keras.layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
            conv1 = keras.layers.Conv2D(32, 3, activation='relu', padding='same')(conv1)
            pool1 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)
            
            conv2 = keras.layers.Conv2D(64, 3, activation='relu', padding='same')(pool1)
            conv2 = keras.layers.Conv2D(64, 3, activation='relu', padding='same')(conv2)
            pool2 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
            
            # Bottleneck
            conv3 = keras.layers.Conv2D(128, 3, activation='relu', padding='same')(pool2)
            conv3 = keras.layers.Conv2D(128, 3, activation='relu', padding='same')(conv3)
            
            # Decoder
            up1 = keras.layers.Conv2DTranspose(64, 2, strides=(2, 2), padding='same')(conv3)
            up1 = keras.layers.concatenate([up1, conv2])
            conv4 = keras.layers.Conv2D(64, 3, activation='relu', padding='same')(up1)
            conv4 = keras.layers.Conv2D(64, 3, activation='relu', padding='same')(conv4)
            
            up2 = keras.layers.Conv2DTranspose(32, 2, strides=(2, 2), padding='same')(conv4)
            up2 = keras.layers.concatenate([up2, conv1])
            conv5 = keras.layers.Conv2D(32, 3, activation='relu', padding='same')(up2)
            conv5 = keras.layers.Conv2D(32, 3, activation='relu', padding='same')(conv5)
            
            # Output
            outputs = keras.layers.Conv2D(1, 1, activation='sigmoid')(conv5)
            
            model = keras.Model(inputs=inputs, outputs=outputs)
            
            # Compile model
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=0.001),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            self.model = model
            self.load_successful = True
            
            logger.info("✓ New compatible model created successfully")
            return True
            
        except Exception as e:
            error_msg = f"Failed to create compatible model: {str(e)}"
            logger.error(error_msg)
            self.error_details.append(error_msg)
            return False

def test_model_loading(model_path):
    """Test function to verify model loading"""
    loader = ModelLoader()
    
    print(f"Testing model loading: {model_path}")
    print(f"TensorFlow version: {loader.tf_version}")
    print(f"Keras version: {loader.keras_version}")
    print("-" * 50)
    
    # Try to load model
    success = loader.load_model(model_path)
    
    if success:
        print("✓ Model loaded successfully!")
        
        # Test prediction
        pred_success, pred_msg = loader.test_model_prediction()
        print(f"Prediction test: {'✓' if pred_success else '✗'} {pred_msg}")
        
        # Get model info
        info = loader.get_model_info()
        print(f"Model info: {info}")
        
    else:
        print("✗ Model loading failed!")
        print("Error details:")
        for error in loader.error_details:
            print(f"  - {error}")
        
        # Try to create compatible model
        print("\nAttempting to create compatible model...")
        if loader.create_compatible_model():
            print("✓ Compatible model created!")
        else:
            print("✗ Failed to create compatible model")
    
    return loader

if __name__ == "__main__":
    # Test the model loader
    model_path = "checkpoints/custom_lane_model_best.h5"
    if os.path.exists(model_path):
        test_model_loading(model_path)
    else:
        print(f"Model file not found: {model_path}")
