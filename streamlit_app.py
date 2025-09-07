"""
Streamlit Web App for AutoLane
Professional deployment with training and prediction capabilities
"""

import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
import tempfile
import json
from datetime import datetime
import matplotlib.pyplot as plt
import plotly.express as px
from PIL import Image
import pandas as pd
import psutil
import gc

# Import OpenCV with error handling
try:
    import cv2
except ImportError as e:
    st.error(f"Failed to import OpenCV: {e}")
    st.error("Please ensure opencv-python-headless is installed correctly.")
    st.stop()

# Import our custom classes
try:
    from training import CustomLaneTrainer
    from prediction import CustomLanePredictor
except ImportError as e:
    st.error(f"Failed to import custom classes: {e}")
    st.error("Please ensure training.py and prediction.py are in the same directory.")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="AutoLane",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

def get_memory_usage():
    """Get current memory usage"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def clear_memory():
    """Clear memory"""
    gc.collect()

def main():
    try:
        # Header
        st.markdown('<h1 class="main-header">🚗 AutoLane</h1>', unsafe_allow_html=True)
        st.markdown("---")
        
        # Sidebar
        st.sidebar.title("Navigation")
        page = st.sidebar.selectbox(
            "Choose a page",
            ["🏠 Home", "🎯 Predict", "📚 Train Model", "📊 Analytics", "🔧 Model Diagnostics", "⚙️ Settings"]
        )
        
        # Memory usage in sidebar
        try:
            memory_usage = get_memory_usage()
            st.sidebar.metric("Memory Usage", f"{memory_usage:.1f} MB")
        except Exception as e:
            st.sidebar.warning(f"Memory info unavailable: {e}")
        
        # Route to appropriate page
        if page == "🏠 Home":
            show_home_page()
        elif page == "🎯 Predict":
            show_predict_page()
        elif page == "📚 Train Model":
            show_train_page()
        elif page == "📊 Analytics":
            show_analytics_page()
        elif page == "🔧 Model Diagnostics":
            show_model_diagnostics_page()
        elif page == "⚙️ Settings":
            show_settings_page()
            
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.error("Please check the Model Diagnostics page for troubleshooting information.")
        
        # Show basic system info even if main app fails
        with st.expander("System Information"):
            st.write(f"TensorFlow Version: {tf.__version__}")
            st.write(f"Python Version: {os.sys.version}")
            st.write(f"Streamlit Version: {st.__version__}")
            try:
                st.write(f"OpenCV Version: {cv2.__version__}")
            except:
                st.write("OpenCV: Not available")

def show_home_page():
    """Home page with overview and quick start"""
    st.markdown("## Welcome to AutoLane")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🎯 Features
        - **Real-time Lane Detection** using CNN
        - **Custom Model Training** with your datasets
        - **Multiple Input Formats** (Images & Videos)
        - **Advanced Analytics** and visualization
        - **Memory Optimized** for large datasets
        """)
    
    with col2:
        st.markdown("""
        ### 🚀 Quick Start
        1. **Upload** your video/image
        2. **Select** a trained model
        3. **Process** and get results
        4. **Download** the output
        """)
    
    # System status
    st.markdown("### 📊 System Status")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Available Models", len([f for f in os.listdir('checkpoints') if f.endswith('.h5')]) if os.path.exists('checkpoints') else 0)
    
    with col2:
        st.metric("Memory Usage", f"{get_memory_usage():.1f} MB")
    
    with col3:
        st.metric("GPU Available", "Yes" if tf.config.list_physical_devices('GPU') else "No")
    
    with col4:
        st.metric("Python Version", f"{os.sys.version.split()[0]}")
    
    # Recent activity
    if os.path.exists('checkpoints/custom_training_info.json'):
        with open('checkpoints/custom_training_info.json', 'r') as f:
            training_info = json.load(f)
        
        st.markdown("### 📈 Recent Training")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Last Training", training_info.get('timestamp', 'N/A')[:10])
        
        with col2:
            st.metric("Accuracy", f"{training_info.get('final_accuracy', 0):.3f}")
        
        with col3:
            st.metric("Epochs", training_info.get('epochs', 0))

def show_predict_page():
    """Prediction page"""
    st.markdown("## 🎯 AutoLane Prediction")
    
    # Model selection
    st.markdown("### 1. Select Model")
    
    if not os.path.exists('checkpoints'):
        st.error("No checkpoints folder found. Please train a model first.")
        return
    
    model_files = [f for f in os.listdir('checkpoints') if f.endswith('.h5')]
    
    if not model_files:
        st.error("No trained models found. Please train a model first.")
        return
    
    selected_model = st.selectbox("Choose a model", model_files)
    model_path = f"checkpoints/{selected_model}"
    
    # Load model info if available
    model_info_path = f"checkpoints/{selected_model.replace('.h5', '_info.json')}"
    if os.path.exists(model_info_path):
        with open(model_info_path, 'r') as f:
            model_info = json.load(f)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Training Accuracy", f"{model_info.get('final_accuracy', 0):.3f}")
        with col2:
            st.metric("Training Epochs", model_info.get('epochs', 0))
        with col3:
            st.metric("Model Size", f"{os.path.getsize(model_path) / (1024*1024):.1f} MB")
    
    # Input selection
    st.markdown("### 2. Upload Input")
    
    input_type = st.radio("Choose input type", ["Image", "Video"])
    
    if input_type == "Image":
        uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png', 'bmp'])
        
        if uploaded_file is not None:
            # Process image
            image = Image.open(uploaded_file)
            image_array = np.array(image)
            
            # Convert PIL to OpenCV format
            if len(image_array.shape) == 3:
                image_cv = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
            else:
                image_cv = image_array
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(image, caption="Original Image", use_column_width=True)
            
            with col2:
                if st.button("Detect Lanes", type="primary"):
                    with st.spinner("Processing image..."):
                        try:
                            # Load predictor with detailed error handling
                            with st.spinner("Loading model..."):
                                predictor = CustomLanePredictor(model_path)
                            
                            if predictor.model is None:
                                st.error("❌ Failed to load model")
                                st.error("Please check the troubleshooting section below for solutions.")
                                
                                # Show troubleshooting info
                                with st.expander("🔧 Troubleshooting Information"):
                                    st.markdown("""
                                    **Common Model Loading Issues:**
                                    
                                    1. **TensorFlow Version Mismatch**: The model was saved with a different TensorFlow version
                                    2. **Missing Dependencies**: Required packages are not installed
                                    3. **Corrupted Model File**: The model file may be damaged
                                    4. **Custom Objects**: Model contains custom layers not recognized
                                    
                                    **Solutions:**
                                    - Try retraining the model with current TensorFlow version
                                    - Check if all dependencies are installed: `pip install -r requirements.txt`
                                    - Verify the model file is not corrupted
                                    - Use the Training page to create a new model
                                    """)
                                return
                            
                            # Predict
                            lane_mask, confidence_map = predictor.predict_lanes(image_cv)
                            result = predictor.visualize_prediction(image_cv, lane_mask)
                            
                            # Convert back to PIL for display
                            result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
                            result_pil = Image.fromarray(result_rgb)
                            
                            st.image(result_pil, caption="AutoLane Detection Result", use_column_width=True)
                            
                            # Show confidence
                            confidence = np.mean(confidence_map) / 255.0
                            st.metric("Detection Confidence", f"{confidence:.3f}")
                            
                            # Download button
                            result_bytes = cv2.imencode('.jpg', result)[1].tobytes()
                            st.download_button(
                                label="Download Result",
                                data=result_bytes,
                                file_name=f"lane_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg",
                                mime="image/jpeg"
                            )
                            
                        except Exception as e:
                            st.error(f"Error processing image: {str(e)}")
    
    else:  # Video
        uploaded_file = st.file_uploader("Upload a video", type=['mp4', 'avi', 'mov', 'mkv'])
        
        if uploaded_file is not None:
            # Save uploaded video
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(uploaded_file.read())
                temp_video_path = tmp_file.name
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.video(uploaded_file)
            
            with col2:
                if st.button("Process Video", type="primary"):
                    with st.spinner("Processing video..."):
                        try:
                            # Load predictor with detailed error handling
                            with st.spinner("Loading model..."):
                                predictor = CustomLanePredictor(model_path)
                            
                            if predictor.model is None:
                                st.error("❌ Failed to load model")
                                st.error("Please check the troubleshooting section below for solutions.")
                                
                                # Show troubleshooting info
                                with st.expander("🔧 Troubleshooting Information"):
                                    st.markdown("""
                                    **Common Model Loading Issues:**
                                    
                                    1. **TensorFlow Version Mismatch**: The model was saved with a different TensorFlow version
                                    2. **Missing Dependencies**: Required packages are not installed
                                    3. **Corrupted Model File**: The model file may be damaged
                                    4. **Custom Objects**: Model contains custom layers not recognized
                                    
                                    **Solutions:**
                                    - Try retraining the model with current TensorFlow version
                                    - Check if all dependencies are installed: `pip install -r requirements.txt`
                                    - Verify the model file is not corrupted
                                    - Use the Training page to create a new model
                                    """)
                                return
                            
                            # Process video
                            output_path = f"outputs/output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
                            
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            # Custom video processing with progress
                            cap = cv2.VideoCapture(temp_video_path)
                            fps = int(cap.get(cv2.CAP_PROP_FPS))
                            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                            
                            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                            
                            frame_count = 0
                            
                            while True:
                                ret, frame = cap.read()
                                if not ret:
                                    break
                                
                                # Predict lanes
                                lane_mask, _ = predictor.predict_lanes(frame)
                                result = predictor.visualize_prediction(frame, lane_mask)
                                
                                # Write frame
                                out.write(result)
                                
                                frame_count += 1
                                progress = frame_count / total_frames
                                progress_bar.progress(progress)
                                status_text.text(f"Processing frame {frame_count}/{total_frames}")
                            
                            cap.release()
                            out.release()
                            
                            # Show result
                            st.success("Video processing complete!")
                            st.video(output_path)
                            
                            # Download button
                            with open(output_path, 'rb') as f:
                                video_bytes = f.read()
                            
                            st.download_button(
                                label="Download Processed Video",
                                data=video_bytes,
                                file_name=f"lane_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4",
                                mime="video/mp4"
                            )
                            
                            # Cleanup
                            os.unlink(temp_video_path)
                            os.unlink(output_path)
                            
                        except Exception as e:
                            st.error(f"Error processing video: {str(e)}")
                            if os.path.exists(temp_video_path):
                                os.unlink(temp_video_path)

def show_train_page():
    """Training page"""
    st.markdown("## 📚 Train Custom Model")
    
    # Training parameters
    st.markdown("### Training Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        epochs = st.slider("Epochs", 10, 300, 100)
        batch_size = st.selectbox("Batch Size", [16, 32, 64, 128, 256], index=2)
        frames_per_video = st.slider("Frames per Video", 50, 2000, 500)
    
    with col2:
        skip_frames = st.slider("Skip Frames", 1, 10, 2)
        use_generator = st.checkbox("Use Memory Generator", value=True)
        augment = st.checkbox("Enable Data Augmentation", value=True)
    
    # Video upload
    st.markdown("### Upload Training Videos")
    uploaded_videos = st.file_uploader(
        "Upload training videos", 
        type=['mp4', 'avi', 'mov', 'mkv'],
        accept_multiple_files=True
    )
    
    if uploaded_videos:
        st.success(f"Uploaded {len(uploaded_videos)} videos")
        
        # Save uploaded videos
        video_paths = []
        for i, video in enumerate(uploaded_videos):
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(video.read())
                video_paths.append(tmp_file.name)
        
        # Training button
        if st.button("Start Training", type="primary"):
            with st.spinner("Training model..."):
                try:
                    # Create trainer
                    trainer = CustomLaneTrainer()
                    
                    # Progress tracking
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Load data
                    status_text.text("Loading dataset...")
                    progress_bar.progress(0.1)
                    
                    X, y = trainer.load_dataset_from_videos(video_paths, frames_per_video, skip_frames)
                    
                    if len(X) == 0:
                        st.error("No data loaded. Please check your video files.")
                        return
                    
                    progress_bar.progress(0.3)
                    status_text.text(f"Loaded {len(X)} training samples")
                    
                    # Split data
                    from sklearn.model_selection import train_test_split
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
                    
                    progress_bar.progress(0.4)
                    status_text.text("Data split complete")
                    
                    # Train model
                    status_text.text("Training model...")
                    progress_bar.progress(0.5)
                    
                    history = trainer.train_model(X_train, y_train, X_val, y_val, 
                                                epochs=epochs, batch_size=batch_size, 
                                                use_generator=use_generator)
                    
                    progress_bar.progress(0.8)
                    status_text.text("Evaluating model...")
                    
                    # Evaluate
                    results, mean_iou = trainer.evaluate_model(X_test, y_test)
                    
                    progress_bar.progress(1.0)
                    status_text.text("Training complete!")
                    
                    # Save model
                    model_name = f"custom_lane_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.h5"
                    trainer.save_model(model_name)
                    
                    # Show results
                    st.success("Training completed successfully!")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Final Accuracy", f"{results[1]:.3f}")
                    with col2:
                        st.metric("Final Loss", f"{results[0]:.3f}")
                    with col3:
                        st.metric("Mean IoU", f"{mean_iou:.3f}")
                    with col4:
                        st.metric("Model Size", f"{os.path.getsize(f'checkpoints/{model_name}') / (1024*1024):.1f} MB")
                    
                    # Plot training history
                    fig = trainer.plot_training_history()
                    st.pyplot(fig)
                    
                except Exception as e:
                    st.error(f"Training failed: {str(e)}")
                
                finally:
                    # Cleanup
                    for video_path in video_paths:
                        if os.path.exists(video_path):
                            os.unlink(video_path)

def show_analytics_page():
    """Analytics page"""
    st.markdown("## 📊 Analytics & Monitoring")
    
    # System metrics
    st.markdown("### System Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Memory Usage", f"{get_memory_usage():.1f} MB")
    
    with col2:
        st.metric("CPU Usage", f"{psutil.cpu_percent():.1f}%")
    
    with col3:
        st.metric("Available Models", len([f for f in os.listdir('checkpoints') if f.endswith('.h5')]) if os.path.exists('checkpoints') else 0)
    
    with col4:
        st.metric("Disk Usage", f"{psutil.disk_usage('.').percent:.1f}%")
    
    # Training history
    if os.path.exists('checkpoints/custom_training_info.json'):
        st.markdown("### Training History")
        
        with open('checkpoints/custom_training_info.json', 'r') as f:
            training_info = json.load(f)
        
        # Create metrics dataframe
        metrics_data = {
            'Metric': ['Accuracy', 'Loss', 'Mean IoU', 'Epochs', 'Batch Size'],
            'Value': [
                training_info.get('final_accuracy', 0),
                training_info.get('final_loss', 0),
                training_info.get('mean_iou', 0),
                training_info.get('epochs', 0),
                training_info.get('batch_size', 0)
            ]
        }
        
        df = pd.DataFrame(metrics_data)
        st.dataframe(df, use_container_width=True)
        
        # Training curves if available
        if os.path.exists('checkpoints/training_log.csv'):
            st.markdown("### Training Curves")
            
            log_df = pd.read_csv('checkpoints/training_log.csv')
            
            # Plot accuracy
            fig_acc = px.line(log_df, x='epoch', y=['accuracy', 'val_accuracy'], 
                            title='Training Accuracy')
            st.plotly_chart(fig_acc, use_container_width=True)
            
            # Plot loss
            fig_loss = px.line(log_df, x='epoch', y=['loss', 'val_loss'], 
                             title='Training Loss')
            st.plotly_chart(fig_loss, use_container_width=True)
    
    # Model comparison
    if os.path.exists('checkpoints'):
        st.markdown("### Model Comparison")
        
        model_files = [f for f in os.listdir('checkpoints') if f.endswith('.h5')]
        
        if len(model_files) > 1:
            model_data = []
            for model_file in model_files:
                model_path = f"checkpoints/{model_file}"
                model_size = os.path.getsize(model_path) / (1024*1024)  # MB
                model_data.append({
                    'Model': model_file,
                    'Size (MB)': model_size,
                    'Created': datetime.fromtimestamp(os.path.getctime(model_path)).strftime('%Y-%m-%d %H:%M')
                })
            
            model_df = pd.DataFrame(model_data)
            st.dataframe(model_df, use_container_width=True)
        else:
            st.info("Only one model found. Train more models to see comparison.")

def show_model_diagnostics_page():
    """Model diagnostics page"""
    st.markdown("## 🔧 Model Diagnostics")
    
    # System information
    st.markdown("### System Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("TensorFlow Version", tf.__version__)
    
    with col2:
        st.metric("Keras Version", keras.__version__)
    
    with col3:
        st.metric("Python Version", f"{os.sys.version.split()[0]}")
    
    # Model files check
    st.markdown("### Model Files Check")
    
    if not os.path.exists('checkpoints'):
        st.error("❌ No checkpoints directory found")
        st.info("Create a checkpoints directory and train a model first")
        return
    
    model_files = [f for f in os.listdir('checkpoints') if f.endswith('.h5')]
    
    if not model_files:
        st.error("❌ No trained models found")
        st.info("Train a model first using the Training page")
        return
    
    st.success(f"✓ Found {len(model_files)} model file(s)")
    
    # Test each model
    st.markdown("### Model Loading Tests")
    
    for model_file in model_files:
        model_path = f"checkpoints/{model_file}"
        
        with st.expander(f"Test: {model_file}"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**File:** {model_file}")
                st.write(f"**Size:** {os.path.getsize(model_path) / (1024*1024):.2f} MB")
                st.write(f"**Created:** {datetime.fromtimestamp(os.path.getctime(model_path)).strftime('%Y-%m-%d %H:%M:%S')}")
            
            with col2:
                if st.button(f"Test Load {model_file}", key=f"test_{model_file}"):
                    with st.spinner("Testing model loading..."):
                        try:
                            # Test model loading
                            predictor = CustomLanePredictor(model_path)
                            
                            if predictor.model is not None:
                                st.success("✅ Model loaded successfully!")
                                
                                # Show model info
                                st.write("**Model Summary:**")
                                st.text(f"Input Shape: {predictor.input_shape}")
                                st.text(f"Model Type: {type(predictor.model).__name__}")
                                
                                # Test prediction with dummy data
                                dummy_img = np.random.randint(0, 255, (160, 320, 3), dtype=np.uint8)
                                try:
                                    lane_mask, confidence_map = predictor.predict_lanes(dummy_img)
                                    st.success("✅ Model prediction test passed!")
                                    st.write(f"Output shape: {lane_mask.shape}")
                                    st.write(f"Confidence range: {confidence_map.min():.3f} - {confidence_map.max():.3f}")
                                except Exception as pred_error:
                                    st.warning(f"⚠️ Model loaded but prediction failed: {pred_error}")
                                
                            else:
                                st.error("❌ Model loading failed")
                                
                        except Exception as e:
                            st.error(f"❌ Error testing model: {str(e)}")
                            
                            # Show detailed error info
                            with st.expander("Error Details"):
                                st.code(str(e))
    
    # Dependencies check
    st.markdown("### Dependencies Check")
    
    dependencies = {
        'tensorflow': tf.__version__,
        'keras': keras.__version__,
        'opencv-python': cv2.__version__,
        'numpy': np.__version__,
        'streamlit': st.__version__,
        'matplotlib': plt.matplotlib.__version__,
        'pandas': pd.__version__,
        'plotly': 'Available' if 'plotly' in globals() else 'Not Available',
        'psutil': 'Available' if 'psutil' in globals() else 'Not Available'
    }
    
    for dep, version in dependencies.items():
        if version == 'Available' or version == 'Not Available':
            if version == 'Available':
                st.success(f"✅ {dep}: {version}")
            else:
                st.error(f"❌ {dep}: {version}")
        else:
            st.info(f"ℹ️ {dep}: {version}")
    
    # GPU check
    st.markdown("### GPU Information")
    
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        st.success(f"✅ GPU Available: {len(gpus)} device(s)")
        for i, gpu in enumerate(gpus):
            st.write(f"  - GPU {i}: {gpu.name}")
    else:
        st.warning("⚠️ No GPU detected - using CPU")
    
    # Memory check
    st.markdown("### Memory Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Current Memory Usage", f"{get_memory_usage():.1f} MB")
    
    with col2:
        if st.button("Clear Memory"):
            clear_memory()
            st.success("Memory cleared!")
            st.rerun()
    
    # Troubleshooting guide
    st.markdown("### Troubleshooting Guide")
    
    with st.expander("Common Issues and Solutions"):
        st.markdown("""
        **1. Model Loading Failed**
        - **Cause**: TensorFlow version mismatch
        - **Solution**: Retrain the model with current TensorFlow version
        
        **2. CUDA/GPU Issues**
        - **Cause**: GPU drivers or CUDA not properly installed
        - **Solution**: Install CUDA-compatible TensorFlow or use CPU version
        
        **3. Memory Issues**
        - **Cause**: Insufficient RAM for model loading
        - **Solution**: Close other applications or use smaller batch sizes
        
        **4. OpenCV Issues**
        - **Cause**: Missing system libraries
        - **Solution**: Install opencv-python-headless instead of opencv-python
        
        **5. Custom Objects Error**
        - **Cause**: Model contains custom layers not recognized
        - **Solution**: Define custom objects or retrain without custom layers
        """)

def show_settings_page():
    """Settings page"""
    st.markdown("## ⚙️ Settings")
    
    # Model settings
    st.markdown("### Model Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        input_shape = st.selectbox("Input Shape", 
                                 ["160x320", "240x480", "320x640"], 
                                 index=0)
        
        confidence_threshold = st.slider("Confidence Threshold", 0.1, 0.9, 0.5)
    
    with col2:
        max_memory = st.slider("Max Memory Usage (MB)", 1000, 8000, 4000)
        auto_cleanup = st.checkbox("Auto Memory Cleanup", value=True)
    
    # System settings
    st.markdown("### System Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        log_level = st.selectbox("Log Level", ["DEBUG", "INFO", "WARNING", "ERROR"])
        save_logs = st.checkbox("Save Logs", value=True)
    
    with col2:
        max_upload_size = st.slider("Max Upload Size (MB)", 100, 1000, 500)
        parallel_processing = st.checkbox("Parallel Processing", value=True)
    
    # Save settings
    if st.button("Save Settings", type="primary"):
        settings = {
            'input_shape': input_shape,
            'confidence_threshold': confidence_threshold,
            'max_memory': max_memory,
            'auto_cleanup': auto_cleanup,
            'log_level': log_level,
            'save_logs': save_logs,
            'max_upload_size': max_upload_size,
            'parallel_processing': parallel_processing
        }
        
        with open('streamlit_settings.json', 'w') as f:
            json.dump(settings, f, indent=2)
        
        st.success("Settings saved successfully!")
    
    # Load existing settings
    if os.path.exists('streamlit_settings.json'):
        with open('streamlit_settings.json', 'r') as f:
            existing_settings = json.load(f)
        
        st.markdown("### Current Settings")
        st.json(existing_settings)
    
    # System actions
    st.markdown("### System Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Clear Memory"):
            clear_memory()
            st.success("Memory cleared!")
    
    with col2:
        if st.button("Restart App"):
            st.rerun()
    
    with col3:
        if st.button("Export Logs"):
            if os.path.exists('checkpoints/training_log.csv'):
                with open('checkpoints/training_log.csv', 'r') as f:
                    csv_data = f.read()
                
                st.download_button(
                    label="Download Training Log",
                    data=csv_data,
                    file_name=f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            else:
                st.warning("No training logs found")

if __name__ == "__main__":
    main()
