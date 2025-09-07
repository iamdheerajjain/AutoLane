"""
Training Script for Your Custom Lane Detection Dataset
Optimized for real video datasets
"""

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import argparse
import json
from datetime import datetime
import glob
from tqdm import tqdm
import psutil
import gc

class CustomLaneTrainer:
    def __init__(self, input_shape=(160, 320, 3), num_classes=2):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        self.history = None
        
        # Configure TensorFlow for better memory management
        self._configure_tensorflow()
    
    def _configure_tensorflow(self):
        """Configure TensorFlow for better memory management and stability"""
        # Set memory growth for GPU if available
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print("GPU memory growth enabled")
            except RuntimeError as e:
                print(f"GPU configuration error: {e}")
        
        # Set CPU threads
        tf.config.threading.set_inter_op_parallelism_threads(0)
        tf.config.threading.set_intra_op_parallelism_threads(0)
        
        # Don't disable eager execution - it causes compatibility issues
        # tf.compat.v1.disable_eager_execution()
        
        # Set memory limit
        if gpus:
            try:
                tf.config.experimental.set_memory_limit('GPU:0', 1024)
            except:
                pass
    
    def get_memory_usage(self):
        """Get current memory usage in MB"""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    
    def print_memory_usage(self, stage=""):
        """Print current memory usage"""
        memory_mb = self.get_memory_usage()
        print(f"Memory usage {stage}: {memory_mb:.1f} MB")
        return memory_mb
        
    def create_advanced_model(self):
        """Create an advanced U-Net model for lane detection"""
        inputs = keras.Input(shape=self.input_shape)
        
        # Encoder
        # Block 1
        conv1 = layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
        conv1 = layers.Conv2D(32, 3, activation='relu', padding='same')(conv1)
        pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)
        
        # Block 2
        conv2 = layers.Conv2D(64, 3, activation='relu', padding='same')(pool1)
        conv2 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv2)
        pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)
        
        # Block 3
        conv3 = layers.Conv2D(128, 3, activation='relu', padding='same')(pool2)
        conv3 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv3)
        pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)
        
        # Block 4
        conv4 = layers.Conv2D(256, 3, activation='relu', padding='same')(pool3)
        conv4 = layers.Conv2D(256, 3, activation='relu', padding='same')(conv4)
        drop4 = layers.Dropout(0.5)(conv4)
        pool4 = layers.MaxPooling2D(pool_size=(2, 2))(drop4)
        
        # Bottleneck
        conv5 = layers.Conv2D(512, 3, activation='relu', padding='same')(pool4)
        conv5 = layers.Conv2D(512, 3, activation='relu', padding='same')(conv5)
        drop5 = layers.Dropout(0.5)(conv5)
        
        # Decoder with skip connections
        # Up Block 1
        up6 = layers.Conv2DTranspose(256, 2, strides=(2, 2), padding='same')(drop5)
        up6 = layers.concatenate([up6, conv4])
        conv6 = layers.Conv2D(256, 3, activation='relu', padding='same')(up6)
        conv6 = layers.Conv2D(256, 3, activation='relu', padding='same')(conv6)
        
        # Up Block 2
        up7 = layers.Conv2DTranspose(128, 2, strides=(2, 2), padding='same')(conv6)
        up7 = layers.concatenate([up7, conv3])
        conv7 = layers.Conv2D(128, 3, activation='relu', padding='same')(up7)
        conv7 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv7)
        
        # Up Block 3
        up8 = layers.Conv2DTranspose(64, 2, strides=(2, 2), padding='same')(conv7)
        up8 = layers.concatenate([up8, conv2])
        conv8 = layers.Conv2D(64, 3, activation='relu', padding='same')(up8)
        conv8 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv8)
        
        # Up Block 4
        up9 = layers.Conv2DTranspose(32, 2, strides=(2, 2), padding='same')(conv8)
        up9 = layers.concatenate([up9, conv1])
        conv9 = layers.Conv2D(32, 3, activation='relu', padding='same')(up9)
        conv9 = layers.Conv2D(32, 3, activation='relu', padding='same')(conv9)
        
        # Output layer - use sigmoid for binary segmentation
        outputs = layers.Conv2D(1, 1, activation='sigmoid')(conv9)
        
        model = keras.Model(inputs=inputs, outputs=outputs)
        return model
    
    def load_dataset_from_videos(self, video_paths, frames_per_video=200, skip_frames=5):
        """Load training data from video files with better lane detection"""
        print("Loading dataset from videos...")
        
        images = []
        masks = []
        
        for video_path in video_paths:
            print(f"Processing video: {os.path.basename(video_path)}")
            
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Warning: Could not open video {video_path}")
                continue
            
            frame_count = 0
            frames_processed = 0
            
            # Get video info
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            print(f"  Total frames: {total_frames}, FPS: {fps}")
            
            with tqdm(total=min(frames_per_video, total_frames // skip_frames), 
                     desc=f"Processing {os.path.basename(video_path)}") as pbar:
                
                while frames_processed < frames_per_video:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Process every skip_frames frame
                    if frame_count % skip_frames == 0:
                        # Resize frame
                        frame_resized = cv2.resize(frame, (self.input_shape[1], self.input_shape[0]))
                        
                        # Create enhanced lane mask
                        mask = self.create_enhanced_lane_mask(frame_resized)
                        
                        # Convert mask to binary (0 or 1) and add channel dimension
                        mask_binary = (mask > 128).astype(np.float32)
                        mask_binary = np.expand_dims(mask_binary, axis=-1)
                        
                        # Only add if we detected some lanes
                        if np.sum(mask_binary) > 100:  # At least 100 pixels of lanes
                            images.append(frame_resized.astype(np.float32) / 255.0)
                            masks.append(mask_binary)
                            frames_processed += 1
                            pbar.update(1)
                    
                    frame_count += 1
                    
                    # Break if we've processed enough frames
                    if frames_processed >= frames_per_video:
                        break
            
            cap.release()
            print(f"  Processed {frames_processed} frames from {os.path.basename(video_path)}")
        
        print(f"Total dataset: {len(images)} samples")
        return np.array(images), np.array(masks)
    
    def create_enhanced_lane_mask(self, img):
        """Create enhanced lane mask using multiple techniques"""
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Method 1: Canny edge detection
        edges = cv2.Canny(blur, 50, 150)
        
        # Method 2: Color-based lane detection (white and yellow)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # White lane detection
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 30, 255])
        white_mask = cv2.inRange(hsv, lower_white, upper_white)
        
        # Yellow lane detection
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([30, 255, 255])
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        # Combine color masks
        color_mask = cv2.bitwise_or(white_mask, yellow_mask)
        
        # Method 3: Hough transform
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=15, 
                               minLineLength=30, maxLineGap=20)
        
        # Create combined mask
        mask = np.zeros(gray.shape, dtype=np.uint8)
        
        # Add color-based detection
        mask = cv2.bitwise_or(mask, color_mask)
        
        # Add Hough line detection
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # Only add lines that are roughly horizontal (lanes)
                if abs(y2 - y1) < abs(x2 - x1):  # More horizontal than vertical
                    cv2.line(mask, (x1, y1), (x2, y2), 255, 8)
        
        # Morphological operations to clean up
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Dilate to make lines thicker
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=2)
        
        return mask
    
    def create_data_generator(self, images, masks, batch_size=32, augment=True):
        """Create a memory-efficient data generator with augmentation"""
        def generator():
            try:
                while True:
                    # Shuffle data
                    indices = np.random.permutation(len(images))
                    
                    for i in range(0, len(images), batch_size):
                        batch_indices = indices[i:i + batch_size]
                        batch_images = []
                        batch_masks = []
                        
                        for idx in batch_indices:
                            img = images[idx].copy()
                            mask = masks[idx].copy()
                            
                            # Original
                            batch_images.append(img)
                            batch_masks.append(mask)
                            
                            if augment and np.random.random() > 0.5:  # 50% chance of augmentation
                                # Randomly choose augmentation techniques
                                aug_choice = np.random.randint(0, 4)
                                
                                if aug_choice == 0:  # Horizontal flip
                                    img_aug = np.fliplr(img)
                                    mask_aug = np.fliplr(mask)
                                elif aug_choice == 1:  # Brightness adjustment
                                    img_aug = np.clip(img * np.random.uniform(0.8, 1.2) + np.random.uniform(-0.1, 0.1), 0, 1)
                                    mask_aug = mask
                                elif aug_choice == 2:  # Contrast adjustment
                                    img_aug = np.clip(img * np.random.uniform(0.8, 1.2), 0, 1)
                                    mask_aug = mask
                                else:  # Gaussian noise
                                    noise = np.random.normal(0, 0.05, img.shape)
                                    img_aug = np.clip(img + noise, 0, 1)
                                    mask_aug = mask
                                
                                batch_images.append(img_aug)
                                batch_masks.append(mask_aug)
                        
                        # Ensure proper data types
                        batch_images = np.array(batch_images, dtype=np.float32)
                        batch_masks = np.array(batch_masks, dtype=np.float32)
                        
                        yield batch_images, batch_masks
            except Exception as e:
                print(f"Error in data generator: {e}")
                # Return empty batch to prevent hanging
                yield np.array([]), np.array([])
        
        return generator
    
    def get_dataset_size(self, images, masks, augment=True):
        """Calculate the effective dataset size with augmentation"""
        if augment:
            return len(images) * 2  # Each image gets one augmentation
        return len(images)
    
    def train_model(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=16, use_generator=True):
        """Train the model with memory-efficient techniques"""
        print("Creating and training model...")
        
        # Create model
        self.model = self.create_advanced_model()
        
        # Compile model with advanced optimizer
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'binary_accuracy']
        )
        
        # Create checkpoints directory
        os.makedirs('checkpoints', exist_ok=True)
        
        # Advanced callbacks
        callbacks = [
            keras.callbacks.ModelCheckpoint(
                'checkpoints/custom_lane_model_best.h5',
                save_best_only=True,
                monitor='val_loss',
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=8,
                min_lr=1e-7,
                verbose=1
            ),
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.CSVLogger('checkpoints/training_log.csv'),
            keras.callbacks.TensorBoard(log_dir='checkpoints/logs', histogram_freq=1)
        ]
        
        if use_generator:
            # Use memory-efficient data generator
            print(f"Training with generator (effective samples: {self.get_dataset_size(X_train, y_train)})...")
            
            # Create data generators
            train_gen = self.create_data_generator(X_train, y_train, batch_size, augment=True)
            val_gen = self.create_data_generator(X_val, y_val, batch_size, augment=False)
            
            # Calculate steps per epoch
            steps_per_epoch = max(1, len(X_train) // batch_size)
            validation_steps = max(1, len(X_val) // batch_size)
            
            # Create generator instances
            train_generator = train_gen()
            val_generator = val_gen()
            
            try:
                self.history = self.model.fit(
                    train_generator,
                    steps_per_epoch=steps_per_epoch,
                    epochs=epochs,
                    validation_data=val_generator,
                    validation_steps=validation_steps,
                    callbacks=callbacks,
                    verbose=1
                )
            except Exception as e:
                print(f"Training error: {e}")
                # Clean up generators
                del train_generator, val_generator
                gc.collect()
                raise e
            finally:
                # Clean up generators
                del train_generator, val_generator
                gc.collect()
        else:
            # Use traditional method with Keras ImageDataGenerator
            datagen = keras.preprocessing.image.ImageDataGenerator(
                rotation_range=15,
                width_shift_range=0.1,
                height_shift_range=0.1,
                horizontal_flip=True,
                brightness_range=[0.7, 1.3],
                zoom_range=0.1,
                shear_range=0.1,
                fill_mode='nearest'
            )
            
            print(f"Training with {len(X_train)} samples...")
            self.history = self.model.fit(
                datagen.flow(X_train, y_train, batch_size=batch_size),
                steps_per_epoch=len(X_train) // batch_size,
                epochs=epochs,
                validation_data=(X_val, y_val),
                callbacks=callbacks,
                verbose=1
            )
        
        return self.history
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate the model with detailed metrics"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Evaluate
        results = self.model.evaluate(X_test, y_test, verbose=1)
        
        # Predict
        predictions = self.model.predict(X_test)
        
        # Calculate IoU
        iou_scores = []
        for i in range(len(X_test)):
            pred_mask = (predictions[i] > 0.5).astype(np.uint8)
            true_mask = y_test[i].squeeze() if y_test[i].ndim > 2 else y_test[i]
            
            # Ensure both masks have the same shape
            if pred_mask.ndim > 2:
                pred_mask = pred_mask.squeeze()
            if true_mask.ndim > 2:
                true_mask = true_mask.squeeze()
            
            # Calculate IoU for lane pixels
            intersection = np.logical_and(pred_mask == 1, true_mask == 1).sum()
            union = np.logical_or(pred_mask == 1, true_mask == 1).sum()
            
            if union > 0:
                iou = intersection / union
                iou_scores.append(iou)
        
        mean_iou = np.mean(iou_scores) if iou_scores else 0
        
        print(f"\nEvaluation Results:")
        print(f"Loss: {results[0]:.4f}")
        print(f"Accuracy: {results[1]:.4f}")
        print(f"Binary Accuracy: {results[2]:.4f}")
        print(f"Mean IoU: {mean_iou:.4f}")
        
        return results, mean_iou
    
    def plot_training_history(self, save_path='training_history.png'):
        """Plot comprehensive training history"""
        if self.history is None:
            print("No training history available")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss
        axes[0, 0].plot(self.history.history['loss'], label='Training Loss', linewidth=2)
        axes[0, 0].plot(self.history.history['val_loss'], label='Validation Loss', linewidth=2)
        axes[0, 0].set_title('Model Loss', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy
        axes[0, 1].plot(self.history.history['accuracy'], label='Training Accuracy', linewidth=2)
        axes[0, 1].plot(self.history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
        axes[0, 1].set_title('Model Accuracy', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Binary Accuracy
        if 'binary_accuracy' in self.history.history:
            axes[1, 0].plot(self.history.history['binary_accuracy'], 
                           label='Training Binary Accuracy', linewidth=2)
            axes[1, 0].plot(self.history.history['val_binary_accuracy'], 
                           label='Validation Binary Accuracy', linewidth=2)
            axes[1, 0].set_title('Binary Accuracy', fontsize=14, fontweight='bold')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Binary Accuracy')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Learning Rate
        if 'lr' in self.history.history:
            axes[1, 1].plot(self.history.history['lr'], linewidth=2, color='red')
            axes[1, 1].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].set_yscale('log')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Training history saved to: {save_path}")
    
    def save_model(self, filepath):
        """Save the trained model"""
        if self.model is not None:
            # Create checkpoints directory if it doesn't exist
            os.makedirs('checkpoints', exist_ok=True)
            
            # If filepath doesn't include checkpoints folder, add it
            if not filepath.startswith('checkpoints/'):
                filepath = f'checkpoints/{filepath}'
            
            self.model.save(filepath)
            print(f"Model saved to: {filepath}")
        else:
            print("No model to save")
    
    def test_on_sample(self, img_path, output_path):
        """Test the model on a sample image"""
        if self.model is None:
            print("Model not trained yet")
            return
        
        # Load image
        img = cv2.imread(img_path)
        if img is None:
            print(f"Could not load image: {img_path}")
            return
        
        # Preprocess
        img_resized = cv2.resize(img, (self.input_shape[1], self.input_shape[0]))
        img_normalized = img_resized.astype(np.float32) / 255.0
        img_batch = np.expand_dims(img_normalized, axis=0)
        
        # Predict
        prediction = self.model.predict(img_batch)
        lane_mask = (prediction[0, :, :, 0] > 0.5).astype(np.uint8) * 255
        
        # Resize back to original
        lane_mask_resized = cv2.resize(lane_mask, (img.shape[1], img.shape[0]))
        
        # Create visualization
        result = img.copy()
        result[lane_mask_resized > 128] = [0, 255, 0]  # Green for lanes
        result = cv2.addWeighted(img, 0.7, result, 0.3, 0)
        
        # Save result
        cv2.imwrite(output_path, result)
        cv2.imwrite(output_path.replace('.jpg', '_mask.jpg'), lane_mask_resized)
        
        print(f"Test result saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Train AutoLane - Custom Lane Detection Model')
    parser.add_argument('--videos', nargs='+', required=True, help='Training video paths')
    parser.add_argument('--output_model', type=str, default='custom_lane_model.h5',
                       help='Output model file path')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--frames_per_video', type=int, default=200,
                       help='Number of frames to extract per video')
    parser.add_argument('--skip_frames', type=int, default=5,
                       help='Skip every N frames to get variety')
    parser.add_argument('--augment', action='store_true', help='Apply data augmentation')
    parser.add_argument('--test_image', type=str, help='Test image path after training')
    parser.add_argument('--use_generator', action='store_true', default=True, 
                       help='Use memory-efficient data generator (recommended)')
    parser.add_argument('--reduce_frames', type=int, default=200,
                       help='Reduce frames per video to save memory')
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = CustomLaneTrainer()
    
    # Expand wildcards in video paths
    video_paths = []
    for video_pattern in args.videos:
        if '*' in video_pattern or '?' in video_pattern:
            # Expand wildcards
            expanded_paths = glob.glob(video_pattern)
            video_paths.extend(expanded_paths)
        else:
            video_paths.append(video_pattern)
    
    if not video_paths:
        print("No video files found. Please check your video paths.")
        return
    
    print(f"Found {len(video_paths)} video files:")
    for video_path in video_paths:
        print(f"  - {video_path}")
    
    # Load data with memory monitoring
    print("Loading dataset from videos...")
    trainer.print_memory_usage("before loading")
    
    # Use reduced frames if specified
    frames_per_video = min(args.frames_per_video, args.reduce_frames)
    print(f"Using {frames_per_video} frames per video to save memory")
    
    try:
        X, y = trainer.load_dataset_from_videos(video_paths, frames_per_video, args.skip_frames)
        
        if len(X) == 0:
            print("No data loaded. Please check your video files.")
            return
        
        print(f"Loaded {len(X)} training samples")
        trainer.print_memory_usage("after loading")
        
        # Ensure data is in correct format
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)
        
        print(f"Data shapes - X: {X.shape}, y: {y.shape}")
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")
    
    # Calculate effective dataset size with augmentation
    effective_train_size = trainer.get_dataset_size(X_train, y_train, augment=args.augment)
    print(f"Effective training samples with augmentation: {effective_train_size}")
    
    # Clear unused variables to free memory
    del X, y
    gc.collect()
    trainer.print_memory_usage("after data split")
    
    # Train model with memory-efficient generator
    print("Starting training...")
    history = trainer.train_model(X_train, y_train, X_val, y_val, 
                                epochs=args.epochs, batch_size=args.batch_size, 
                                use_generator=args.use_generator)
    
    # Evaluate
    print("Evaluating model...")
    results, mean_iou = trainer.evaluate_model(X_test, y_test)
    
    # Plot training history
    trainer.plot_training_history()
    
    # Save model
    trainer.save_model(args.output_model)
    
    # Test on sample image if provided
    if args.test_image:
        print(f"Testing on sample image: {args.test_image}")
        trainer.test_on_sample(args.test_image, 'test_result.jpg')
    
    # Save training info
    training_info = {
        'model_path': f'checkpoints/{args.output_model}',
        'training_samples': len(X_train),
        'validation_samples': len(X_val),
        'test_samples': len(X_test),
        'effective_training_samples': effective_train_size,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'frames_per_video': frames_per_video,
        'use_generator': args.use_generator,
        'augmentation_enabled': args.augment,
        'final_loss': results[0],
        'final_accuracy': results[1],
        'mean_iou': mean_iou,
        'timestamp': datetime.now().isoformat(),
        'videos_used': video_paths,
        'memory_optimized': True
    }
    
    # Create checkpoints directory if it doesn't exist
    os.makedirs('checkpoints', exist_ok=True)
    
    with open('checkpoints/custom_training_info.json', 'w') as f:
        json.dump(training_info, f, indent=2)
    
    print(f"\nTraining complete! Model saved to: checkpoints/{args.output_model}")
    print(f"Training info saved to: checkpoints/custom_training_info.json")

def main_wrapper():
    """Wrapper function to handle Python interpreter state properly"""
    try:
        # Run main function
        main()
        
    except Exception as e:
        print(f"Error in main wrapper: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up
        gc.collect()
        tf.keras.backend.clear_session()

if __name__ == "__main__":
    main_wrapper()
