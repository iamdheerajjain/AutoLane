import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import argparse
import os
from datetime import datetime
import matplotlib.pyplot as plt

class CustomLanePredictor:
    def __init__(self, model_path, input_shape=(160, 320, 3)):
        self.input_shape = input_shape
        self.model = None
        self.load_model(model_path)
    
    def load_model(self, model_path):
        """Load the custom trained model with enhanced error handling"""
        if not os.path.exists(model_path):
            print(f"Model not found: {model_path}")
            print("Please train a model first using training.py")
            return False
        
        try:
            # Try different loading methods for compatibility
            print(f"Attempting to load model from: {model_path}")
            
            # Method 1: Standard Keras load_model
            try:
                self.model = keras.models.load_model(model_path, compile=False)
                print(f"✓ Model loaded successfully using standard method")
                return True
            except Exception as e1:
                print(f"Standard loading failed: {e1}")
                
                # Method 2: Load with custom objects
                try:
                    self.model = keras.models.load_model(
                        model_path, 
                        compile=False,
                        custom_objects={'tf': tf}
                    )
                    print(f"✓ Model loaded successfully with custom objects")
                    return True
                except Exception as e2:
                    print(f"Custom objects loading failed: {e2}")
                    
                    # Method 3: Load weights only (if model architecture is known)
                    try:
                        # This is a fallback - you might need to reconstruct the model architecture
                        print("Attempting to load model weights only...")
                        # Note: This requires knowing the model architecture
                        # For now, we'll raise the original error
                        raise e1
                    except Exception as e3:
                        print(f"Weights-only loading failed: {e3}")
                        
                        # Method 4: Try with different TensorFlow versions
                        try:
                            import tensorflow.compat.v1 as tf_v1
                            tf_v1.disable_eager_execution()
                            self.model = tf_v1.keras.models.load_model(model_path)
                            print(f"✓ Model loaded successfully with TF v1 compatibility")
                            return True
                        except Exception as e4:
                            print(f"TF v1 compatibility loading failed: {e4}")
                            
                            # Final error with detailed information
                            self._print_detailed_error_info(model_path, [e1, e2, e3, e4])
                            return False
        
        except Exception as e:
            print(f"Unexpected error during model loading: {e}")
            self._print_detailed_error_info(model_path, [e])
            return False
    
    def _print_detailed_error_info(self, model_path, errors):
        """Print detailed error information for troubleshooting"""
        print("\n" + "="*60)
        print("MODEL LOADING FAILED - TROUBLESHOOTING INFO")
        print("="*60)
        print(f"Model path: {model_path}")
        print(f"Model size: {os.path.getsize(model_path) / (1024*1024):.2f} MB")
        print(f"TensorFlow version: {tf.__version__}")
        print(f"Keras version: {keras.__version__}")
        print(f"Python version: {os.sys.version}")
        
        print("\nError details:")
        for i, error in enumerate(errors, 1):
            print(f"  {i}. {type(error).__name__}: {error}")
        
        print("\nPossible solutions:")
        print("1. Check if the model was saved with a different TensorFlow version")
        print("2. Try retraining the model with current TensorFlow version")
        print("3. Check if all custom layers are properly defined")
        print("4. Verify the model file is not corrupted")
        print("5. Try loading with compile=False parameter")
        print("6. Check if the model requires specific custom objects")
        
        print("\nTo fix this issue:")
        print("1. Run: python training.py to retrain the model")
        print("2. Or check the model compatibility in the Streamlit app")
        print("="*60)
    
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
        prediction = self.model.predict(img_processed)
        
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
        cv2.putText(result, 'AutoLane Model', (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        return result
    
    def process_image(self, input_path, output_path):
        """Process single image"""
        # Ensure outputs directory exists
        os.makedirs('outputs', exist_ok=True)
        
        # If output_path doesn't include outputs folder, add it
        if not output_path.startswith('outputs/'):
            output_path = f"outputs/{output_path}"
        
        # Load image
        img = cv2.imread(input_path)
        if img is None:
            raise ValueError(f"Could not load image: {input_path}")
        
        print(f"Processing image: {input_path}")
        print(f"Image size: {img.shape[1]}x{img.shape[0]}")
        
        # Predict lanes
        lane_mask, confidence_map = self.predict_lanes(img)
        
        # Visualize result
        result = self.visualize_prediction(img, lane_mask)
        
        # Save results
        cv2.imwrite(output_path, result)
        cv2.imwrite(output_path.replace('.jpg', '_mask.jpg'), lane_mask)
        
        # Create comparison visualization
        self.create_comparison_visualization(img, lane_mask, output_path.replace('.jpg', '_comparison.jpg'))
        
        print(f"Result saved to: {output_path}")
        print(f"Lane mask saved to: {output_path.replace('.jpg', '_mask.jpg')}")
        print(f"Comparison saved to: {output_path.replace('.jpg', '_comparison.jpg')}")
        
        return result, lane_mask
    
    def create_comparison_visualization(self, img, lane_mask, output_path):
        """Create side-by-side comparison visualization"""
        # Resize mask to original image size
        lane_mask_resized = cv2.resize(lane_mask, (img.shape[1], img.shape[0]))
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Original image
        axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        # Lane mask
        axes[1].imshow(lane_mask_resized, cmap='hot')
        axes[1].set_title('Detected Lanes (Heatmap)', fontsize=14, fontweight='bold')
        axes[1].axis('off')
        
        # Overlay
        overlay = img.copy()
        overlay[lane_mask_resized > 128] = [0, 255, 0]
        overlay = cv2.addWeighted(img, 0.7, overlay, 0.3, 0)
        axes[2].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        axes[2].set_title('Final Result', fontsize=14, fontweight='bold')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def process_video(self, input_path, output_path):
        """Process video file"""
        # Ensure outputs directory exists
        os.makedirs('outputs', exist_ok=True)
        
        # If output_path doesn't include outputs folder, add it
        if not output_path.startswith('outputs/'):
            output_path = f"outputs/{output_path}"
        
        # Open input video
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {input_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Processing video: {input_path}")
        print(f"Resolution: {width}x{height}, FPS: {fps}, Frames: {total_frames}")
        
        # Setup output video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
            raise ValueError(f"Could not create output video: {output_path}")
        
        # Process frames
        frame_count = 0
        start_time = datetime.now()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Predict lanes
                lane_mask, _ = self.predict_lanes(frame)
                
                # Visualize prediction
                result = self.visualize_prediction(frame, lane_mask)
                
                # Write frame
                out.write(result)
                
                frame_count += 1
                
                # Progress update
                if frame_count % 30 == 0:
                    elapsed = (datetime.now() - start_time).total_seconds()
                    fps_current = frame_count / elapsed if elapsed > 0 else 0
                    progress = (frame_count / total_frames) * 100
                    print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames}) - {fps_current:.1f} FPS")
        
        except KeyboardInterrupt:
            print("\nProcessing interrupted by user")
        
        finally:
            # Cleanup
            cap.release()
            out.release()
            
            # Final statistics
            total_time = (datetime.now() - start_time).total_seconds()
            avg_fps = frame_count / total_time if total_time > 0 else 0
            
            print(f"\nVideo processing complete!")
            print(f"Processed {frame_count} frames in {total_time:.1f} seconds")
            print(f"Average FPS: {avg_fps:.1f}")
            print(f"Output saved to: {output_path}")
    
    def create_comparison_video(self, input_path, output_path):
        """Create side-by-side comparison video"""
        # Ensure outputs directory exists
        os.makedirs('outputs', exist_ok=True)
        
        # If output_path doesn't include outputs folder, add it
        if not output_path.startswith('outputs/'):
            output_path = f"outputs/{output_path}"
        
        # Open input video
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {input_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Setup output video writer (side-by-side, so double width)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width * 2, height))
        
        if not out.isOpened():
            raise ValueError(f"Could not create output video: {output_path}")
        
        print(f"Creating comparison video: {input_path}")
        print(f"Resolution: {width*2}x{height}, FPS: {fps}, Frames: {total_frames}")
        
        # Process frames
        frame_count = 0
        start_time = datetime.now()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Predict lanes
                lane_mask, _ = self.predict_lanes(frame)
                
                # Visualize prediction
                result = self.visualize_prediction(frame, lane_mask)
                
                # Create side-by-side comparison
                comparison = np.hstack([frame, result])
                
                # Add labels
                cv2.putText(comparison, 'Original', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(comparison, 'AutoLane Detection', (width + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                # Write frame
                out.write(comparison)
                
                frame_count += 1
                
                # Progress update
                if frame_count % 30 == 0:
                    elapsed = (datetime.now() - start_time).total_seconds()
                    fps_current = frame_count / elapsed if elapsed > 0 else 0
                    progress = (frame_count / total_frames) * 100
                    print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames}) - {fps_current:.1f} FPS")
        
        except KeyboardInterrupt:
            print("\nProcessing interrupted by user")
        
        finally:
            # Cleanup
            cap.release()
            out.release()
            
            # Final statistics
            total_time = (datetime.now() - start_time).total_seconds()
            avg_fps = frame_count / total_time if total_time > 0 else 0
            
            print(f"\nComparison video complete!")
            print(f"Processed {frame_count} frames in {total_time:.1f} seconds")
            print(f"Average FPS: {avg_fps:.1f}")
            print(f"Output saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='AutoLane - Lane Detection with Custom Trained Model')
    parser.add_argument('--input', type=str, required=True, help='Input image or video path')
    parser.add_argument('--output', type=str, required=True, help='Output path')
    parser.add_argument('--model', type=str, default='custom_lane_model.h5', help='Custom model path')
    parser.add_argument('--comparison', action='store_true', help='Create side-by-side comparison video')
    
    args = parser.parse_args()
    
    # Create predictor
    predictor = CustomLanePredictor(model_path=args.model)
    
    if predictor.model is None:
        print("Failed to load model. Exiting.")
        return
    
    # Check if input is image or video
    if args.input.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        # Process image
        try:
            predictor.process_image(args.input, args.output)
        except Exception as e:
            print(f"Error processing image: {e}")
    
    elif args.input.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        # Process video
        try:
            if args.comparison:
                predictor.create_comparison_video(args.input, args.output)
            else:
                predictor.process_video(args.input, args.output)
        except Exception as e:
            print(f"Error processing video: {e}")
    
    else:
        print("Unsupported file format. Please use .jpg, .png, .mp4, .avi, .mov, or .mkv")

if __name__ == "__main__":
    main()
