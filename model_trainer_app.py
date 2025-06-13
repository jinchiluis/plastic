import streamlit as st
import os
from datetime import datetime
from model_trainer import ModelTrainer

# Page config
st.set_page_config(
    page_title="Defective Bag Detector Training",
    page_icon="🔍",
    layout="centered"
)

# Initialize session state
if 'trainer' not in st.session_state:
    st.session_state.trainer = ModelTrainer()
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'training_in_progress' not in st.session_state:
    st.session_state.training_in_progress = False

# Title and description
st.title("🔍 Defective Bag Detector")
st.markdown("### Training Interface")
st.markdown("Train an anomaly detection model using good plastic bag images")

# Sidebar for folder selection
with st.sidebar:
    st.header("📁 Training Data")
    
    # Folder path input
    folder_path = st.text_input(
        "Enter folder path containing good bag images:",
        placeholder="/path/to/good/images"
    )
    
    # Check if folder exists
    if folder_path:
        if os.path.exists(folder_path) and os.path.isdir(folder_path):
            # Count images in folder
            valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
            image_files = [f for f in os.listdir(folder_path) 
                          if f.lower().endswith(valid_extensions)]
            
            st.success(f"✅ Folder found: {len(image_files)} images")
            
            # Show sample images
            if len(image_files) > 0:
                st.markdown("#### Sample images:")
                cols = st.columns(3)
                for i, img_file in enumerate(image_files[:3]):
                    with cols[i]:
                        st.text(img_file[:15] + "...")
        else:
            st.error("❌ Folder not found")

# Main area
col1, col2 = st.columns(2)

with col1:
    # Train button
    if st.button(
        "🚀 Train Model", 
        type="primary", 
        disabled=not (folder_path and os.path.exists(folder_path)),
        use_container_width=True
    ):
        st.session_state.training_in_progress = True

# Training progress
if st.session_state.training_in_progress:
    with st.spinner("Training in progress... This may take a few minutes."):
        try:
            # Create progress placeholder
            progress_placeholder = st.empty()
            
            # Train the model
            progress_placeholder.info("📊 Extracting features from images...")
            model = st.session_state.trainer.train_autoencoder(folder_path, contamination=0.005) #Do little contamination
            
            # Generate model filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_filename = f"defect_detector_{timestamp}.pkl"
            model_path = os.path.join("models", model_filename)
            
            # Save the model
            progress_placeholder.info("💾 Saving model...")
            saved_path = st.session_state.trainer.save_model(model, model_path)
            
            # Update session state
            st.session_state.model_trained = True
            st.session_state.training_in_progress = False
            st.session_state.last_model_path = saved_path
            
            # Clear progress and show success
            progress_placeholder.empty()
            st.success(f"✅ Model trained and saved successfully!")
            st.info(f"📁 Model saved to: `{saved_path}`")
            
        except Exception as e:
            st.session_state.training_in_progress = False
            st.error(f"❌ Training failed: {str(e)}")

# Model info section
if st.session_state.model_trained:
    st.markdown("---")
    st.markdown("### 📊 Model Information")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Model Type", "Isolation Forest")
        st.metric("Feature Extractor", "MobileNetV2")
    with col2:
        st.metric("Status", "✅ Trained")
        if 'last_model_path' in st.session_state:
            st.metric("Location", st.session_state.last_model_path.split('/')[-1])

# Instructions
with st.expander("📖 How to use", expanded=True):
    st.markdown("""
    1. **Prepare your data**: Place all good (non-defective) bag images in a single folder
    2. **Enter folder path**: Type or paste the full path to your image folder
    3. **Train model**: Click the "Train Model" button
    4. **Wait**: The app will extract features and train the anomaly detector
    5. **Model saved**: The trained model will be saved in the "models" folder
    
    **Note**: 
    - Only good/normal bag images should be used for training
    - Supported formats: JPG, JPEG, PNG, BMP, TIFF
    - The model will learn what "normal" looks like and detect deviations
    """)

# Footer
st.markdown("---")
st.markdown("*Anomaly Detection using MobileNetV2 + Isolation Forest*")