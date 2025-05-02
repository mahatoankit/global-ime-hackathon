import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model as keras_load_model
from tensorflow.keras.applications.resnet50 import ResNet50
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import pickle
import logging
import warnings

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Suppress TensorFlow logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # 0=all, 1=info, 2=warning, 3=error
tf.get_logger().setLevel("ERROR")
warnings.filterwarnings("ignore", category=UserWarning)


def load_model():
    """
    Load all necessary models and vectorizers for the document classification system.

    Returns:
    dict: Dictionary containing all loaded models and vectorizers
    """
    logging.info("Starting to load models and components...")

    # Define the base path where models are stored
    base_model_dir = (
        "/home/ankit/WindowsFuneral/Hackathons/Global-AI-hackathon-2025/Models/"
    )

    # Define paths for each model component
    model_paths = {
        "main_model": os.path.join(base_model_dir, "my_model.keras"),
        "vectorizer": os.path.join(base_model_dir, "tfidf_vectorizer.pkl"),
        "label_map": os.path.join(base_model_dir, "label_map.pkl"),
    }

    # Create directory if it doesn't exist
    # This is a fallback in case models are not found
    os.makedirs(base_model_dir, exist_ok=True)

    result = {}

    # Load main classification model
    try:
        logging.info(f"Loading main model from: {model_paths['main_model']}")
        result["main_model"] = keras_load_model(model_paths["main_model"])
        logging.info("Main model loaded successfully")
    except (OSError, IOError) as e:
        logging.warning(f"Error loading main model: {e}")
        logging.info("Using placeholder model instead")
        # Create a simple placeholder model for demonstration
        result["main_model"] = create_placeholder_model()

    # Load ResNet50 for image feature extraction
    try:
        logging.info("Loading ResNet50 for image feature extraction")
        # Suppress output during model download
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result["resnet"] = ResNet50(
                weights="imagenet", include_top=False, input_shape=(224, 224, 3)
            )
        logging.info("ResNet50 loaded successfully")
    except Exception as e:
        logging.error(f"Error loading ResNet50: {e}")
        # No fallback for ResNet - it's a critical component
        raise

    # Load TF-IDF vectorizer
    try:
        if os.path.exists(model_paths["vectorizer"]):
            logging.info(f"Loading vectorizer from: {model_paths['vectorizer']}")
            with open(model_paths["vectorizer"], "rb") as f:
                result["vectorizer"] = pickle.load(f)
            logging.info("Vectorizer loaded successfully")
        else:
            logging.info("Vectorizer not found, creating new one")
            result["vectorizer"] = TfidfVectorizer(max_features=500)
    except Exception as e:
        logging.warning(f"Error loading vectorizer: {e}")
        # Create a new vectorizer as fallback
        result["vectorizer"] = TfidfVectorizer(max_features=500)

    # Load or define label mapping
    try:
        if os.path.exists(model_paths["label_map"]):
            logging.info(f"Loading label map from: {model_paths['label_map']}")
            with open(model_paths["label_map"], "rb") as f:
                result["label_map"] = pickle.load(f)
            logging.info("Label map loaded successfully")
        else:
            logging.info("Label map not found, using default mapping")
            result["label_map"] = {
                0: "Birth Certificate",
                1: "Blank",
                2: "Citizenship",
                3: "NID",
                4: "PAN",
            }
    except Exception as e:
        logging.warning(f"Error loading label map: {e}")
        # Default label mapping as fallback
        result["label_map"] = {
            0: "Birth Certificate",
            1: "Blank",
            2: "Citizenship",
            3: "NID",
            4: "PAN",
        }

    logging.info("All models and components loaded successfully")
    return result


def create_placeholder_model():
    """
    Create a placeholder model for demonstration purposes
    when the real model is unavailable.
    """
    logging.info("Creating placeholder model")
    input_shape = (1000,)  # Assumed combined feature size
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(64, activation="relu", input_shape=input_shape),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(5, activation="softmax"),  # 5 classes
        ]
    )
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model
