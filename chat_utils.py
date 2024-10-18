import os 
import glob
from utils import *

def load_chat_history(chat_id):
    chat_history = retrieve_chat_history(chat_id)
    messages = []
    for message in chat_history[0].chat_memory.messages:
        messages.append(message.content)
    paired_messages = [(messages[i], messages[i + 1]) for i in range(0, len(messages) - 1, 2)]
    return paired_messages

def delete_all_chats_history(folder_path):
    files = glob.glob(os.path.join(folder_path, '*'))
    for file in files:
        try:
            os.remove(file)
            print(f"Deleted file: {file}")
        except Exception as e:
            print(f"Error deleting file {file}: {e}")

def delete_all_uploaded_files(folder_path):
    files = glob.glob(os.path.join(folder_path, '*'))
    for file in files:
        try:
            os.remove(file)
            print(f"Deleted file: {file}")
        except Exception as e:
            print(f"Error deleting file {file}: {e}")

def delete_chat_history(folder_path, chat_id):
    # Find the specific file matching the chat_id
    file_path = os.path.join(folder_path, f"{chat_id}.json")
    
    # Check if the file exists and delete it
    if file_path and os.path.isfile(file_path):
        try:
            os.remove(file_path)
            print(f"Deleted file: {file_path}")
        except Exception as e:
            print(f"Error deleting file {file_path}: {e}")
    else:
        print(f"No file found with chat ID: {chat_id}")

def clear_chat(state):
    # Retrieve the file name from the state dictionary safely    
    delete_chat_history("chat_memory", state['chat_id'])
    state['chat_id'] = None

    return (), state