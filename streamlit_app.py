"""
Streamlit Web App for AutoLane
Professional deployment with training and prediction capabilities
"""

import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
import tempfile
import json
from datetime import datetime
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import pandas as pd
import time
import psutil
import gc

# Import our custom classes
from training import CustomLaneTrainer
from prediction import CustomLanePredictor

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
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #c3e6cb;
    }
    .error-message {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #f5c6cb;
    }
    .info-message {
        background-color: #d1ecf1;
        color: #0c5460;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #bee5eb;
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
    # Header
    st.markdown('<h1 class="main-header">🚗 AutoLane</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["🏠 Home", "🎯 Predict", "📚 Train Model", "📊 Analytics", "⚙️ Settings"]
    )
    
    # Memory usage in sidebar
    memory_usage = get_memory_usage()
    st.sidebar.metric("Memory Usage", f"{memory_usage:.1f} MB")
    
    if page == "🏠 Home":
        show_home_page()
    elif page == "🎯 Predict":
        show_predict_page()
    elif page == "📚 Train Model":
        show_train_page()
    elif page == "📊 Analytics":
        show_analytics_page()
    elif page == "⚙️ Settings":
        show_settings_page()

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
                            # Load predictor
                            predictor = CustomLanePredictor(model_path)
                            
                            if predictor.model is None:
                                st.error("Failed to load model")
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
                            # Load predictor
                            predictor = CustomLanePredictor(model_path)
                            
                            if predictor.model is None:
                                st.error("Failed to load model")
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
