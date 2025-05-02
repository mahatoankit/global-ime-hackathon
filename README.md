# Document Classification App

This project is a Streamlit web application designed for document classification using a pre-trained model developed with transfer learning and functional APIs for late fusion. The application allows users to upload documents and receive classification results based on the content of the documents.

## Project Structure

```
document-classification-app
├── src
│   ├── app.py                # Main entry point for the Streamlit application
│   ├── model
│   │   ├── __init__.py       # Initializes the model package
│   │   ├── load_model.py      # Functions to load the pre-trained model
│   │   └── predict.py         # Functions for document classification
│   ├── utils
│   │   ├── __init__.py       # Initializes the utils package
│   │   └── preprocess.py      # Functions for document preprocessing
│   └── components
│       ├── __init__.py       # Initializes the components package
│       └── ui_elements.py     # UI elements for the Streamlit application
├── requirements.txt           # Lists project dependencies
├── README.md                  # Documentation for the project
└── .streamlit
    └── config.toml           # Configuration settings for the Streamlit app
```

## Setup Instructions

1. Clone the repository:
   ```
   git clone <repository-url>
   cd document-classification-app
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the Streamlit application:
   ```
   streamlit run src/app.py
   ```

## Usage

- Upload a document using the provided file uploader in the application.
- The application will process the document and display the predicted class based on the pre-trained model.

## Model Information

The document classification model utilizes transfer learning techniques and is designed to handle various document types. The model is loaded and utilized through the `load_model.py` and `predict.py` modules.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.