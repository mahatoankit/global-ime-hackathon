from streamlit import file_uploader, button, write, text, container

def upload_document():
    uploaded_file = file_uploader("Upload a document", type=["pdf", "docx", "txt"])
    return uploaded_file

def display_classification_result(result):
    with container():
        text("Classification Result:")
        write(result)

def create_classification_button(callback):
    if button("Classify Document"):
        callback()