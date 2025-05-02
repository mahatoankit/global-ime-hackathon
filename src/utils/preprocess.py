import numpy as np
import pytesseract
from PIL import Image
import io
from tensorflow.keras.applications.resnet50 import preprocess_input
import fitz  # PyMuPDF
import docx
import re


def preprocess_document(document):
    """
    Process uploaded document based on its type

    Parameters:
    document: The uploaded document file object

    Returns:
    dict: Dictionary containing extracted text and image (if available)
    """
    file_type = document.name.split(".")[-1].lower()

    # Read file content
    content = document.read()
    document.seek(0)  # Reset file pointer

    result = {"text": None, "image": None}

    # Extract based on file type
    if file_type == "pdf":
        result = extract_from_pdf(io.BytesIO(content))
    elif file_type == "docx":
        result = extract_from_docx(io.BytesIO(content))
    elif file_type == "txt":
        text = content.decode("utf-8")
        result["text"] = normalize_text(text)
    elif file_type in ["jpg", "jpeg", "png", "tiff", "bmp"]:
        result = extract_from_image(io.BytesIO(content))

    return result


def extract_from_pdf(pdf_file):
    """Extract text and first image from PDF"""
    result = {"text": "", "image": None}

    try:
        doc = fitz.open(stream=pdf_file, filetype="pdf")

        # Extract text from all pages
        for page in doc:
            result["text"] += page.get_text()

        # Try to extract first image if available
        for page_num in range(len(doc)):
            page = doc[page_num]
            image_list = page.get_images(full=True)
            if image_list:
                # Get first image
                xref = image_list[0][0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                result["image"] = Image.open(io.BytesIO(image_bytes))
                break

        result["text"] = normalize_text(result["text"])
    except Exception as e:
        print(f"Error extracting from PDF: {e}")

    return result


def extract_from_docx(docx_file):
    """Extract text and first image from DOCX"""
    result = {"text": "", "image": None}

    try:
        doc = docx.Document(docx_file)

        # Extract text
        for para in doc.paragraphs:
            result["text"] += para.text + "\n"

        # Images from DOCX are complex to extract directly
        # Would require additional libraries

        result["text"] = normalize_text(result["text"])
    except Exception as e:
        print(f"Error extracting from DOCX: {e}")

    return result


def extract_from_image(image_file):
    """Extract text from image using OCR and return the image itself"""
    result = {"text": "", "image": None}

    try:
        img = Image.open(image_file)
        result["image"] = img

        # Extract text using OCR
        result["text"] = pytesseract.image_to_string(img)
        result["text"] = normalize_text(result["text"])
    except Exception as e:
        print(f"Error extracting from image: {e}")

    return result


def normalize_text(text):
    """Normalize text by lowercasing and removing special characters"""
    if not text:
        return ""

    # Convert to lowercase
    text = text.lower()

    # Remove special characters but keep spaces
    text = re.sub(r"[^\w\s]", " ", text)

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def preprocess_image(image, target_size=(224, 224)):
    """
    Preprocess image for model input

    Parameters:
    image: PIL Image object
    target_size: Target size for resizing

    Returns:
    np.array: Preprocessed image array
    """
    if image is None:
        # Return empty image with zeros if no image available
        return np.zeros((1, target_size[0], target_size[1], 3))

    # Resize and convert to RGB if needed
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)

    # Convert to numpy array
    img_array = np.array(image)

    # Apply ResNet50 preprocessing
    img_array = preprocess_input(img_array)

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    return img_array
