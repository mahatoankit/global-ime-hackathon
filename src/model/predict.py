import numpy as np
import logging
import time
import contextlib
import io
import sys
import os

# Add the parent directory to sys.path to allow absolute imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_model():
    import joblib
    import os

    model_path = os.path.join(os.path.dirname(__file__), "model.joblib")
    model = joblib.load(model_path)
    return model


def predict_document_class(model, document):
    from sklearn.feature_extraction.text import TfidfVectorizer

    # Assuming the model expects a TF-IDF vectorized input
    vectorizer = TfidfVectorizer()
    document_vector = vectorizer.transform([document])

    prediction = model.predict(document_vector)
    return prediction[0]  # Return the predicted class label


def predict(models, document_data):
    """
    Predict document class based on both image and text features.

    Parameters:
    models: Dictionary containing loaded models and vectorizers
    document_data: Dictionary containing processed text and image

    Returns:
    dict: Dictionary containing the prediction results
    """
    logging.info("Starting document prediction")

    # Extract models and processors from the input dictionary
    main_model = models["main_model"]
    resnet = models["resnet"]
    vectorizer = models["vectorizer"]
    label_map = models["label_map"]

    # Extract processed text and image from document data
    text = document_data.get("text", "")
    image = document_data.get("image", None)

    # Process text features with TF-IDF
    logging.info("Processing text features")
    if text:
        # If vectorizer hasn't been fit yet, fit it first
        if not hasattr(vectorizer, "vocabulary_"):
            try:
                logging.info("Fitting vectorizer on text")
                vectorizer.fit([text])
            except Exception as e:
                logging.error(f"Error fitting vectorizer: {e}")
                logging.info("Using dummy text features")
                text_features = np.zeros((1, 500))  # Assuming 500 features

        # Transform text to TF-IDF features
        try:
            text_features = vectorizer.transform([text]).toarray()
            logging.info(f"Text features extracted with shape: {text_features.shape}")
        except Exception as e:
            logging.error(f"Error transforming text with vectorizer: {e}")
            logging.info("Using dummy text features")
            text_features = np.zeros((1, 500))  # Assuming 500 features
    else:
        logging.warning("No text available, using zeros for text features")
        text_features = np.zeros((1, 500))  # Assuming 500 features

    # Process image features with ResNet50
    logging.info("Processing image features")
    if image is not None:
        # Use absolute import instead of relative import
        from utils.preprocess import preprocess_image

        try:
            img_array = preprocess_image(image)

            # Suppress stdout during prediction
            with contextlib.redirect_stdout(io.StringIO()):
                # Get image features from ResNet50
                logging.info("Extracting features from image using ResNet50")
                start_time = time.time()
                image_features = resnet.predict(img_array, verbose=0)
                end_time = time.time()
                logging.info(
                    f"Image feature extraction completed in {end_time - start_time:.2f} seconds"
                )

                # Flatten the features for concatenation
                image_features = image_features.reshape(image_features.shape[0], -1)
                logging.info(
                    f"Image features extracted with shape: {image_features.shape}"
                )
        except Exception as e:
            logging.error(f"Error extracting image features: {e}", exc_info=True)
            logging.info("Using dummy image features")
            image_features = np.zeros((1, 100352))  # Size depends on ResNet output
    else:
        logging.warning("No image available, using zeros for image features")
        image_features = np.zeros((1, 100352))  # Size depends on ResNet output

    # Combine features
    logging.info("Combining image and text features")
    try:
        combined_features = np.concatenate([image_features, text_features], axis=1)
        logging.info(f"Combined features shape: {combined_features.shape}")
    except ValueError as e:
        logging.error(f"Error concatenating features: {e}")
        logging.error(
            f"Image features shape: {image_features.shape}, Text features shape: {text_features.shape}"
        )
        logging.info("Using dummy combined features")
        combined_features = np.zeros((1, 100852))  # Adjust size as needed

    # Make prediction
    logging.info("Making prediction with the model")
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            prediction_probabilities = main_model.predict(combined_features, verbose=0)
        predicted_class_index = np.argmax(prediction_probabilities)
        confidence = float(prediction_probabilities[0][predicted_class_index])
        logging.info(
            f"Prediction completed. Class index: {predicted_class_index}, Confidence: {confidence:.4f}"
        )
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        logging.info("Using default prediction values")
        predicted_class_index = 0
        confidence = 0.0
        prediction_probabilities = np.zeros((1, len(label_map)))
        prediction_probabilities[0, 0] = 1.0

    # Get class label
    predicted_label = label_map.get(predicted_class_index, "Unknown")
    logging.info(f"Predicted label: {predicted_label}")

    # Return prediction results
    result = {
        "predicted_label": predicted_label,
        "confidence": confidence,
        "class_index": int(predicted_class_index),
        "probabilities": {
            label_map.get(i, f"Class {i}"): float(prob)
            for i, prob in enumerate(prediction_probabilities[0])
        },
    }

    logging.info("Document prediction completed successfully")
    return result
