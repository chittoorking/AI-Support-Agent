from gtts import gTTS
from playsound import playsound
import os
def Text_to_speech(Message):
    speech = gTTS(Message)
    speech.save('play.mp3')
    return "play.mp3"
def Play_speech(path):
    playsound(path)
def delete_file(file_path):
    if os.path.isfile(file_path):
        os.remove(file_path)
    else:
        print("File does not exist")
def Generate_text_to_speech(Message):
    file=Text_to_speech(Message)
    Play_speech(file)
    delete_file(file)