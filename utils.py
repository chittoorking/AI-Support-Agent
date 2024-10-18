from langchain.memory import ConversationBufferWindowMemory
from langchain.memory.chat_message_histories import FileChatMessageHistory
import os, uuid
import re
from eldar import Query
import os
import openai
from dotenv import load_dotenv
load_dotenv()
from qdrant_crud import upload_documents
from langchain_community.document_loaders import PyMuPDFLoader

OPENAI_API_KEY =  os.getenv("OPENAI_API_KEY")

MEM_DIR = os.getenv("MEM_DIR", "")

openai.api_key = OPENAI_API_KEY

def pdf_loader(file_path=None):
    loader = PyMuPDFLoader(file_path)
    documents = loader.load()
    return documents

def upload_pdf_doc(file_path, source_name):
    docs = pdf_loader(file_path=file_path)
    print(upload_documents(docs, source=source_name))
    return {"message": "Done", "file_name": source_name}

def retrieve_chat_history(chat_id=None, k=2):
    if not chat_id:
        chat_id = str(uuid.uuid4())
    
    file_path = f"./chat_memory/{chat_id}.json"
    
    # Create the file if it does not exist
    if not os.path.exists(file_path):
        with open(file_path, 'w') as f:
            f.write("[]")  # Initialize with an empty JSON array or appropriate initial content
    
    mem_store = FileChatMessageHistory(file_path)
    memory = ConversationBufferWindowMemory(
        memory_key="chat_history", chat_memory=mem_store, return_messages=True, k=k, output_key='answer'
    )
    
    return memory, chat_id