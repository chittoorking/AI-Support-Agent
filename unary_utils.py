from utils import *
from chat_utils import *
import gradio as gr
from qdrant_crud import *
import os
import shutil

## PROCESS AND LOAD THE UPLOADED FILE INTO VECTOR DATABASE

async def process_file(file, state):
    if file is None:
        return "No file uploaded", state
    output_dir_uploaded_files = "./uploaded_file"

    os.makedirs(output_dir_uploaded_files, exist_ok=True)

    file_name = f"{os.path.basename(file.name)}"
    file_extension = os.path.splitext(file_name)[1].lower()

    if file_extension != ".pdf":
        return "Uploaded file is not a PDF"

    print(file_name)
    state['file_name'] = file_name

    if filename_exists(file_name, collection_name="knowledge-base"):
        return "Processing Completed", state
    else:
        shutil.copy(file.name, output_dir_uploaded_files)
        upload_pdf_doc(file_path=f"{output_dir_uploaded_files}/{file_name}", source_name=file_name)
        return "Processing Completed", state

def reset_state(state):
    # Retrieve the file name from the state dictionary safely
    file_name = state.get('file_name', "")

    # Construct the full file path
    uploaded_file_path = os.path.join("./uploaded_file", file_name)
    
    # Check if the file name is not empty and if the file exists, then delete it
    if file_name and os.path.isfile(uploaded_file_path):
        try:
            os.remove(uploaded_file_path)
            print(f"Deleted file: {uploaded_file_path}")
        except Exception as e:
            print(f"Error deleting file {uploaded_file_path}: {e}")
    else:
        print(f"No file found: {uploaded_file_path}")
    
    delete_all_uploaded_files("uploaded_file")
    
    delete_chat_history("chat_memory", state['chat_id'])

    # Clear the file name in the state
    state['file_name'] = ""
    state['chat_id'] = None 

    return "",None,(), state