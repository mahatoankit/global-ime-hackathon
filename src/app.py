import streamlit as st
import numpy as np
import os
import tensorflow as tf
import logging
from model.load_model import load_model
from model.predict import predict
from utils.preprocess import preprocess_document
import time
import warnings

# Initialize logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Suppress TensorFlow warnings and logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # 0=all, 1=info, 2=warning, 3=error
tf.get_logger().setLevel("ERROR")
warnings.filterwarnings("ignore")

# Configure Streamlit page
st.set_page_config(
    page_title="Document Classification System", page_icon="📄", layout="wide"
)

# Add custom CSS for styling
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        color: #F39C12;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-box {
        background-color: #F7F7F7;
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
    }
    .confidence-high {
        color: green;
        font-weight: bold;
    }
    .confidence-medium {
        color: orange;
        font-weight: bold;
    }
    .confidence-low {
        color: red;
        font-weight: bold;
    }
    .footer {
        text-align: center;
        margin-top: 3rem;
        color: #777777;
        font-size: 0.8rem;
    }
    .stAlert {
        background-color: #f8f9fa;
        padding: 10px;
        border-radius: 5px;
    }
    .gpu-status {
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    .gpu-available {
        background-color: #d4edda;
        color: #155724;
    }
    .gpu-unavailable {
        background-color: #fff3cd;
        color: #856404;
    }
    .system-info {
        font-size: 0.8rem;
        color: #6c757d;
    }
</style>
""",
    unsafe_allow_html=True,
)


# Cache the model loading to avoid reloading on every rerun
@st.cache_resource
def load_cached_model():
    with st.spinner("Loading models - this may take a moment..."):
        try:
            return load_model()
        except Exception as e:
            st.error(f"Error loading models: {str(e)}")
            return None


# Check GPU availability
def check_gpu_availability():
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        return True, f"GPU available: {len(gpus)} device(s) detected"
    else:
        return False, "No GPU detected. Using CPU for inference (slower)"


# Main function to run the app
def main():
    # Display header
    st.markdown(
        "<h1 class='main-header'>Document Classification System</h1>",
        unsafe_allow_html=True,
    )

    # Check GPU status
    gpu_available, gpu_message = check_gpu_availability()
    gpu_class = "gpu-available" if gpu_available else "gpu-unavailable"
    st.markdown(
        f"<div class='gpu-status {gpu_class}'>{gpu_message}</div>",
        unsafe_allow_html=True,
    )

    # Display system information
    with st.expander("System Information"):
        st.markdown("<div class='system-info'>", unsafe_allow_html=True)
        st.markdown(f"**TensorFlow Version:** {tf.__version__}")
        st.markdown(f"**NumPy Version:** {np.__version__}")
        st.markdown("**Model Information:** Image-Text Fusion Model using ResNet50")
        st.markdown("</div>", unsafe_allow_html=True)

    # Load models
    models = load_cached_model()

    if not models:
        st.error("Failed to load models. Please check the logs or contact support.")
        return

    # Sidebar information
    with st.sidebar:
        st.header("About")
        st.info(
            "This application uses deep learning to classify various types of documents. "
            "Upload a document image or file to classify it into one of the following categories:\n"
            "• Birth Certificate\n"
            "• Citizenship\n"
            "• National ID (NID)\n"
            "• PAN Card\n"
            "• Blank/Other"
        )

        st.header("Supported File Types")
        st.markdown(
            """
            - Images (JPG, PNG, JPEG)
            - PDF Documents
            - Word Documents (DOCX)
            - Text Files (TXT)
            """
        )

        st.header("How it Works")
        st.markdown(
            """
            This system uses a multimodal approach:
            1. **Text Analysis**: Extracts and processes text from documents
            2. **Image Analysis**: Analyzes visual features using ResNet50
            3. **Fusion Model**: Combines both features for accurate classification
            """
        )

    # Main content - File uploader
    st.subheader("Upload a Document")
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["pdf", "docx", "txt", "jpg", "jpeg", "png"],
        help="Upload a document file to classify",
    )

    # Process the uploaded file
    if uploaded_file is not None:
        col1, col2 = st.columns([1, 1])

        # Display file details
        with col1:
            st.write("File Details:")
            file_details = {
                "Name": uploaded_file.name,
                "Type": uploaded_file.type,
                "Size": f"{uploaded_file.size / 1024:.2f} KB",
            }
            for key, value in file_details.items():
                st.write(f"**{key}:** {value}")

        # Preview the file if it's an image
        with col2:
            if uploaded_file.type.startswith("image"):
                st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

        # Process document when button is clicked
        if st.button("Classify Document"):
            # Create a progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()

            try:
                # Step 1: Preprocessing
                status_text.text("Preprocessing document...")
                progress_bar.progress(10)
                time.sleep(0.5)  # Small delay for visual feedback

                document_data = preprocess_document(uploaded_file)
                progress_bar.progress(30)

                # Step 2: Feature extraction
                status_text.text("Extracting features...")
                progress_bar.progress(50)
                time.sleep(0.5)  # Small delay for visual feedback

                # Step 3: Model prediction
                status_text.text("Running classification model...")
                progress_bar.progress(70)

                # Make prediction
                result = predict(models, document_data)
                progress_bar.progress(90)

                # Step 4: Complete
                status_text.text("Classification complete!")
                progress_bar.progress(100)
                time.sleep(0.5)  # Small delay for visual feedback

                # Clear the status elements
                status_text.empty()

                # Display results in a nice format
                st.markdown("<div class='result-box'>", unsafe_allow_html=True)
                st.subheader("Classification Result")

                # Display the predicted class and confidence
                st.markdown(f"**Predicted Document Type:** {result['predicted_label']}")

                # Format confidence percentage and determine color based on confidence level
                confidence_pct = result["confidence"] * 100
                if confidence_pct >= 80:
                    confidence_class = "confidence-high"
                elif confidence_pct >= 50:
                    confidence_class = "confidence-medium"
                else:
                    confidence_class = "confidence-low"

                st.markdown(
                    f"**Confidence:** <span class='{confidence_class}'>{confidence_pct:.2f}%</span>",
                    unsafe_allow_html=True,
                )

                # Show all class probabilities as a bar chart
                st.subheader("Class Probabilities")
                probabilities_df = {
                    "Class": list(result["probabilities"].keys()),
                    "Probability": list(result["probabilities"].values()),
                }
                st.bar_chart(probabilities_df, x="Class", y="Probability")

                st.markdown("</div>", unsafe_allow_html=True)

            except Exception as e:
                # Clear progress indicators
                progress_bar.empty()
                status_text.empty()

                # Show error message
                st.error(f"An error occurred during processing: {str(e)}")
                st.info(
                    "Please try uploading a different file or contact support if the issue persists."
                )
                logging.error(f"Error processing document: {str(e)}", exc_info=True)

    # Footer
    st.markdown(
        "<div class='footer'>Global AI Hackathon 2025 - MergeDocs</div>",
        unsafe_allow_html=True,
    )


# Run the app
if __name__ == "__main__":
    main()
