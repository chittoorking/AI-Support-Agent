import gradio as gr
from utils import *
from fastapi import FastAPI
from run_query import QueryIn_unary, search_docs
import asyncio
from chat_utils import *
from unary_utils import *
from transformers import pipeline
import numpy as np
import gtts  # Import gTTS
import os
import torch
from Text2Speech import *

# Add custom CSS to set the minimum height for the chatbot
CSS = """
<style>
    #upload_header {
        display: flex;
        justify-content: center;
        align-items: center;
    }
</style>
"""

# Initialize the FastAPI app
main_app = FastAPI(redoc_url=None, docs_url=None)

# Whisper model for automatic speech recognition (ASR)
transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-base.en")

# Function to transcribe audio input to text
def transcribe(audio):
    if audio is None:
        return "No audio provided."  # Handle case when user cancels the audio input

    sr, y = audio
    
    # Convert to mono if stereo
    if y.ndim > 1:
        y = y.mean(axis=1)
        
    y = y.astype(np.float32)
    y /= np.max(np.abs(y))

    return transcriber({"sampling_rate": sr, "raw": y})["text"]

# Refresh function to reset state and clear history
def refresh(state):
    delete_all_chats_history("chat_memory")
    delete_all_uploaded_files("uploaded_file")
    return reset_state(state)

# Run query function that handles chatbot interaction and generates TTS output
def run_query(query, state):
    try:
        data = QueryIn_unary(query=query, file_name=state['file_name'], chat_id=state['chat_id'])
        _, chat_id = search_docs(data.query, data.file_name, chat_id=data.chat_id)
        state['chat_id'] = chat_id
        chat_history = load_chat_history(chat_id)
        
        # Generate speech for the chatbot's latest response
        if chat_history:
            audio_path = Text_to_speech(chat_history[-1][1])  # Speak the chatbot's latest message
            return chat_history, state, audio_path
        return chat_history, state, None
    except Exception as e:
        print(e)
        return (), state, None

# Function to handle PDF file processing (using async)
def process_file_sync(file, state):
    result, new_state = asyncio.run(process_file(file, state))
    return (result, new_state)

# Create Gradio UI
with gr.Blocks(css=CSS) as demo:
    delete_all_chats_history("chat_memory")
    delete_all_uploaded_files("uploaded_file")

    with gr.Tab("Chat Single File"):
        state = gr.State({'chat_id': None, "file_name": ""})
        gr.Markdown("# Support Tool RAG")

        file_input = gr.File(label="Upload a PDF file", type="file", show_label=False)
        process_button = gr.Button("Process PDF File")
        pdf_process = gr.Markdown(label="PDF File Processing Status")

        chatbot = gr.Chatbot()
        msg = gr.Textbox(label="Enter your query")
        audio_input = gr.Audio(source="microphone", label="Speak your query")  # Audio input for query
        submit_audio_button = gr.Button("Submit Audio Query")  # Button to submit the transcribed audio query
                
        audio_output = gr.Audio(source='upload', label=" AI Response")  

        clear = gr.Button("Clear Chat")
        reset_button = gr.Button("Reset State")

        # Button to process uploaded PDF file
        process_button.click(
            process_file_sync, 
            inputs=[file_input, state], 
            outputs=[pdf_process, state]
        )

        # Submit the text query from textbox
        msg.submit(run_query, inputs=[msg, state], outputs=[chatbot, state])

        # Transcribe audio and then submit it as a query
        audio_input.change(
            lambda audio: transcribe(audio),  # Transcribe audio
            inputs=audio_input,
            outputs=msg  # Output transcribed text to msg Textbox
        )

        # Submit the transcribed audio as a query when button is pressed
        submit_audio_button.click(
            run_query, 
            inputs=[msg, state], 
            outputs=[chatbot, state, audio_output]
        )

        # Clear chat functionality
        clear.click(clear_chat, [state], [chatbot, state])

        # Reset state and clear inputs
        reset_button.click(reset_state, inputs=[state], outputs=[msg, file_input, chatbot, state])

    demo.load(refresh, [state], [msg, file_input, chatbot, state])

# Mount Gradio app with FastAPI
main_app = gr.mount_gradio_app(main_app, demo, path="/")
