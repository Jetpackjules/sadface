import subprocess

import random
from PIL import Image
from io import BytesIO
from moviepy import editor
import time
from moviepy.editor import *

# get user input for conversation topic
question = input("What topic should the conversation be about? ")
prompt = f"Write a conversation between two AI robots. One is called Kelly and the other Max. Think of the conversation as a talk show. I will prompt a question, and Max and Kelly will anwer, commenting on the other's answer, using my question as a sort of conversation starter. Output NOTHING else (So no new interview question or anything like that!) \n\nFormat the question like this:\n\nSpeaker 1: text here\n\nSpeaker 2: text here\n\netc.\n\nINTERVIEWER'S QUESTION:\n{question}\n"
import GenTools


conversation = GenTools.TextGen(prompt, 100)[0]

print("INPUT:\n" + conversation)
# generate an image using Hugging Face's DALL-E API as input face
GenTools.generate_image("Portrait of a beautiful woman, stunning, high detail, asian,", "Kelly_face.png")

GenTools.generate_image("Portrait of a hansome man, stunning, high detail, american,", "Max_face.png")


# Initializing a dictionary to store the sentences spoken by each person
lines = conversation.split('\n\n')

# Initializing a dictionary to store the sentences spoken by each individual
spoken = {}

# Looping over the lines and appending the sentences to the corresponding lists
for line in lines:
    print("-----------------------------")
    print(line)
    speaker, sentence = line.split(':')
    speaker = speaker.strip() # Removing whitespace from the speaker's name
    sentence = sentence.strip() # Removing whitespace from the sentence
    if speaker not in spoken:
        spoken[speaker] = [] # Adding the speaker to the dictionary
    spoken[speaker].append(sentence) # Appending the sentence to the corresponding list

# Generating numbered MP3 files for each speaker's lines
for i, (speaker, sentences) in enumerate(spoken.items()):
    # Constructing the filename for the speaker's lines
    filename = f"{speaker}_{i}"
    # Generating the MP3 file for the speaker's lines
    if speaker == "Max":
        talker = "Olivia"
    else:
        talker = "Joanna"

    GenTools.speechifyV2("\n".join(sentences), talker, "Conversation\\", "text", filename)
    args = ["python", "SadTalker\inference.py", "--source_image", f"{speaker}_face.png", "--driven_audio", f"Conversation\\{filename}.mp3", "--result_dir", "", "--enhancer", "gfpgan", "--batch_size", "1"]
    subprocess.call(args)

# Combining the MP3 files into a single conversation
os.system("cat *.mp3 > conversation.mp3")
time.sleep(10)


