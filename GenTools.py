import openai
import boto3
import requests as r
from io import BytesIO
import io
import base64
import random
from PIL import Image
from mutagen.mp3 import MP3
from pathlib import Path
from moviepy import editor
import time
from moviepy.editor import *
from moviepy.video.tools.subtitles import SubtitlesClip
import subprocess
import os
import threading
import numpy as np
from polly_vtt import PollyVTT
import shutil
import glob
global p

# Text Generation:
def TextGen(INPUT:str, len:int, CHATGPT:bool=True, superhero:bool = False, temp:int=0.7, timeouts:int = 0):
    #max acceptable timeouts:
    max_timeouts = 4
        
    # OUTDATED?: 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
    openai.api_key = "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"
    try:
        if CHATGPT:
            # NEW CHAT GPT CODE:
            if not superhero:
                mess=[
                    {"role": "system", "content": "You are a genius youtuber robot who makes viral videos all the time."},
                    {"role": "assistant", "content": "Understood! I will do my best to generate funny and high quality content that will go Viral (Without ever using the word 'viral')! I will also make all my stories genuinely funny or interesting, and never start them cliche things like 'once upon a time!'"},
                    {"role": "user", "content": INPUT}
                    ]
            else:
                mess=[
                    {"role": "user", "content": INPUT}
                    ]
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=mess,
                max_tokens=len,
                temperature=temp,
                request_timeout=180
                )
            output = response.get("choices")[0].get("message").get("content")
            tokens_used = response.get("usage").get("total_tokens")
        else:
            #OLD DAVINCI 3 CODE:
            response = openai.Completion.create(model="text-davinci-003", prompt=INPUT, temperature=0.7, max_tokens=len)

            while ('Failed!' in str(response)):
                print("OPEN-AI ERROR ------- RETRYING in 5s... ")
                time.sleep(5)
                response = openai.Completion.create(model="text-davinci-003", prompt=INPUT, temperature=0.7, max_tokens=len)
            output = response.get("choices")[0].get("text")
            tokens_used = response.get("usage").get("total_tokens")
    except:
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!! -- OPENAI TIMED OUT? (I think?) -- !!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("Timeouts: " + str(timeouts))
        if (timeouts < max_timeouts):
            out = TextGen(INPUT, len, CHATGPT, superhero, temp, timeouts = timeouts+1)
            output = out[0]
            tokens_used [1]
        else:
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!! TOO MANY TIMEOUTS !!!!!!!!!!!!!!!!!!!!!!!!!")
    return [output, tokens_used]

# TTS Function:
def speechify(script:str, speaker:str, loc:str, typeOfText:str='text'):
    if (speaker=="random"):
        # Removed lea cuz of accent on e?
        speaker = random.choice(["Amy", "Ruth", "Joanna", "Ayanda", "Aria", "Olivia"])
    # load amazon polly API:
    polly_client = boto3.Session(
        aws_access_key_id="AKIA5KVWL6Q5JCGUWIL5",                     
        aws_secret_access_key="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",
        region_name='us-west-2').client('polly')

    VoiceResponse = polly_client.synthesize_speech(Engine='neural', OutputFormat='mp3', VoiceId=speaker, Text=script, TextType=typeOfText)
    print("Saving...")
    file = open(loc, 'wb')
    file.write(VoiceResponse['AudioStream'].read())
    file.close()

#NEW TTS function:
def speechifyV2(script:str, speaker:str, loc:str, typeOfText:str="ssml", file_name:str="Subtitles"):
    # CURRENTLY ANYTJNIG BUT SSML BREAKS, so using ssml for now...
    if typeOfText != "ssml":
        script = "<speak>"+script+"</speak>"
        typeOfText = "ssml"


    print("HERE IS THE SSML: \n" + script)
    if (speaker=="random"):
        # Removed lea cuz of accent on e?
        speaker = random.choice(["Amy", "Ruth", "Joanna", "Ayanda", "Aria", "Olivia"])
    # load amazon polly:
    polly_vtt = PollyVTT()
    polly_vtt.generate(
        loc+file_name,
        "srt",
        Text=script,
        TextType=typeOfText,
        VoiceId=speaker,
        OutputFormat="mp3",
    )

# GENERATING IMAGE FUNCTION:
def generate_image(IMGprompt:str, SavePath:str, negative_prompt:str=None, Model:str="runwayml/stable-diffusion-v1-5", local=False):
    # LOAD HuggingFace Image gen API:
    # MODEL_ID = "stabilityai/stable-diffusion-2"
    # MODEL_ID = "runwayml/stable-diffusion-v1-5"
    # MODEL_ID = "ogkalu/Superhero-Diffusion
    # MODEL_ID = "jetpackjules/supegen"

    ENDPOINT_URL="https://api-inference.huggingface.co/models/" + Model
    # HF_TOKEN="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"
    # ALTERNATE BUSINESS ACCOUNT 
    HF_TOKEN ="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"
    # ALT: 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

    if local:
        print("RUNNING LOCALLY!")
        payload = {
            "prompt": IMGprompt,
            # "negative_prompt": negative_prompt,
            "steps": 35,
            "parameters": {
                # "temperature": 0.5
                # "width": 512,
                # "height": 512
                # "guidance_scale": 9
            }
        }
        IMGresponse = r.post(url=f'http://127.0.0.1:7860/sdapi/v1/txt2img', json=payload).json()
        for i in IMGresponse['images']:
            img = Image.open(io.BytesIO(base64.b64decode(i.split(",",1)[0])))
    else:
        print("RUNNING ON HUGGINFACE!")
        payload = {
            "inputs": IMGprompt,
            "negative_prompt": negative_prompt,
            # , # <-- I'm not sure this does anything tbh
            "options": {
                "wait_for_model": True,
            },
            "parameters": {
                # "inference_steps": 100,
                # "temperature": 0.5
                # "width": 512,
                # "height": 512
                # "guidance_scale": 9
            }
        }
        headers = {
            "Authorization": f"Bearer {HF_TOKEN}",
            "Content-Type": "application/json",
            "Accept": "image/png" # important to get an image back
        }
        IMGresponse = r.post(ENDPOINT_URL, headers=headers, json=payload)
        failiures = 0
        maxFail = 30
        while not ('200' in str(IMGresponse)):
            print("ERROR -- recieved -->  " + str(IMGresponse) + " <-- RETRYING in 5s... ")
            time.sleep(5)
            failiures += 1
            if failiures>maxFail:
                print("------------------------------------------> SWITCHING ENDPOINT MODEL... <------------------------------------------")
                if ENDPOINT_URL=="https://api-inference.huggingface.co/models/runwayml/stable-diffusion-v1-5":
                    ENDPOINT_URL="https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-2"
                else:
                    ENDPOINT_URL="https://api-inference.huggingface.co/models/runwayml/stable-diffusion-v1-5"
                failiures -= 5
            IMGresponse = r.post(ENDPOINT_URL, headers=headers, json=payload)
        print("ITS GOOD! here is non-error: " + str(IMGresponse))
        img = Image.open(BytesIO(IMGresponse.content))

    image_arr = np.array(img)
    # Check if image is mostly black
    black_threshold = 20 # Adjust this value to change the sensitivity of the check
    if np.mean(image_arr) < black_threshold:
        print("NSFW detected! Retrying image...")
        generate_image(IMGprompt+".", SavePath, negative_prompt, Model, local)
    else:
        img.save(mode=r,fp=SavePath)

# IF RUNNING LOCALLY: INIT:
def local_img_init():
    global p
    batchFileLocation = 'C:/Users/Jetpackjules/stable-diffusion-webui-directml'
    batchFileFullPath = os.path.join(batchFileLocation, 'webui-user.bat')
    def launch():
        global p
        p = subprocess.Popen(os.path.abspath(batchFileFullPath), stdin=subprocess.PIPE, cwd = batchFileLocation)
    plot_thread = threading.Thread(target=launch)
    plot_thread.daemon = True
    plot_thread.start()
    time.sleep(10)
    return p

# Delete all files in a folder:
def clear_folder(directory:str):
    for root, dirs, files in os.walk(directory):
        for file in files:
            os.remove(os.path.join(root, file))

# Crop image to head:
def crop_image(image_path, size=(230,410)):
    with Image.open(image_path) as img:
        width, height = img.size
        # calculate the left and right coordinates of the cropped image
        left = (width - size[0]) // 2
        right = left + size[0]
        # crop the top 408 pixels of the image
        top = 0
        bottom = size[1]
        cropped_img = img.crop((left, top, right, bottom))
        cropped_img.save(image_path.replace(".jpeg", "_cropped.jpeg"))
        return(image_path.replace(".jpeg", "_cropped.jpeg"))

#Upscaling Function:
def upscale(image_path, upscale_type="none", ImageFolderPath="output"):
    if upscale_type != "none":
        print("UPSCALE TYPE: "+ upscale_type)
        if upscale_type == "default":
            upscale_command = f"""AI_tools/realesrgan/realesrgan-ncnn-vulkan.exe -i {image_path} -o {image_path}"""
            subprocess.call(upscale_command, shell=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        elif (upscale_type == "GFPGAN"):
            # Upscale the background:
            upscale_command = f"""AI_tools/realesrgan/realesrgan-ncnn-vulkan.exe -i {image_path} -o {image_path} -n realesrgan-x4plus -t 125"""
            subprocess.call(upscale_command, shell=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            upscale_command = f"""python AI_tools/GFPGAN/inference_gfpgan.py -i {image_path} -o {ImageFolderPath+"upscaled"} -s 1 -v 1.3 --bg_upsampler realesrgan-x4plus"""
            subprocess.call(upscale_command, shell=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            #Move the upscaled image to main output folder
            upscaled_folder = ImageFolderPath+'upscaled'+'/restored_imgs'
            filename = os.path.basename(image_path)
            upscaled_path = upscaled_folder + '/' + filename
            output_path = ImageFolderPath+filename
            shutil.copyfile(upscaled_path, output_path)
        else:
            upscale_command = f"""AI_tools/realesrgan/realesrgan-ncnn-vulkan.exe -i {image_path} -o {image_path} -n {upscale_type} -t 125 -s 4"""
            print(" -------------------------------------------------------------------------------------- RUNNING UPSCALE:")
            subprocess.call(upscale_command, shell=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            # , stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL

# Combining mp3 with Images:
def create_video(ImageFolderPath:str, subbed:bool=True, pixels=(512,512), upscale_type="default"):
    # Not sure if this is worth the time cost...else
    # if upscale_type != "none":
    #     pixels = (2*pixels[0], 2*pixels[1])

    AudioPath = ImageFolderPath + "/Subtitles.mp3"

    # Finding the length of the audio file...
    length_audio = int(MP3(AudioPath).info.length)

    # Checking for images in image folder:
    path_images = Path(ImageFolderPath)
    images = list(path_images.glob('*.jpeg'))
    image_list = list()

    for image_name in images:
        image_list.append(image_name)

    # Matching length of image video with mp3 file?
    duration = int(length_audio / len(image_list))
    
    # OLD SETTINGS: image_path, output_file, duration=5, screensize=pixels, fps=35, zoom_ratio=0.000375, zoom_smooth=10
    def run_ffmpeg_zoom(image_path, output_file, duration=5, screensize=pixels, fps=35, zoom_ratio=0.0005, zoom_smooth=10):
        ffmpeg_command = f"""ffmpeg -framerate {fps} -loop 1 -i {image_path} -filter_complex "[0:v]scale={screensize[0] * zoom_smooth}x{screensize[1] * zoom_smooth}, zoompan=z='min(zoom+{zoom_ratio},1.5)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d={duration * fps},trim=duration={duration}[v1];[v1]scale={screensize[0]}:{screensize[1]}[v]" -map "[v]" -y {output_file}"""
        subprocess.call(ffmpeg_command, shell=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        # If you want output, replace none with 'subprocess.PIPE'  ^^^^^
    
    # -------------------------------------------------

    # split the image list into two halves
    half = len(image_list) // 2
    first_half = image_list[0:half]
    second_half = image_list[half:]

    # upscale the first half of images using threads
    threads = []
    for i in range(len(first_half)):
        print("----- QUEUING UPSCALING FOR IMAGE " + str(i+1) + " -----")
        print("IMAGE: "+ str(first_half[i]))
        t = threading.Thread(target=upscale, args=[first_half[i], upscale_type, ImageFolderPath])
        threads.append(t)
        t.start()
    for t in threads:
        t.join()

    # upscale the second half of images using threads
    threads2 = []
    for i in range(len(second_half)):
        print("----- QUEUING UPSCALING FOR IMAGE " + str(i+half+1) + " -----")
        print("IMAGE: "+str(second_half[i]))
        t2 = threading.Thread(target=upscale, args=[second_half[i], upscale_type, ImageFolderPath])
        threads2.append(t2)
        t2.start()

    for t2 in threads2:
        t2.join()



    # Zoom and concatenate videos
    threads = []
    for i in range(0, len(image_list)):
        print("----- WORKING ON ZOOMED IMAGE " + str(i+1) + " -----")
        temp_video_path = ImageFolderPath + f"temp_{i}.mp4"
        t = threading.Thread(target=run_ffmpeg_zoom, args=[image_list[i], temp_video_path, duration])
        threads.append(t)
        t.start()
    for t in threads:
        t.join()
    
    # Combining all the temp zoom clips:
    print( "----------------------------------------------------- GETTING FILEPATH")
    file_paths = (sorted(glob.glob(ImageFolderPath + "temp_*.mp4"), key=lambda x: int(os.path.basename(x).split("_")[1].split(".")[0])))
    
    clips = []
    for file_path in file_paths:
        clip = editor.VideoFileClip(file_path)
        clips.append(clip)
    final_clip = editor.concatenate_videoclips(clips)

    # ---------------------------------------------

    # Combining zoomed video with audio
    audio = editor.AudioFileClip(AudioPath)
    final_video = final_clip.set_audio(audio)

    # Writing final video to file
    final_video.write_videofile(
        ImageFolderPath+"Video.mp4",
        fps=60,
        ffmpeg_params=['-lavfi', '[0:v]scale=-1:iw*16/9,boxblur=luma_radius=min(h\,w)/20:luma_power=1:chroma_radius=min(cw\,ch)/20:chroma_power=1[bg];[bg][0:v]overlay=(W-w)/2:(H-h)/2,crop=w=ih*9/16']
    )

    # Adding subtitle if subbed is True
    if subbed:
        final_videoBLURRED = VideoFileClip(ImageFolderPath+"Video.mp4")     

        # WITH BLOCK BACKGROUND AND MONOSPACE:
        # generator = lambda txt: TextClip(txt, bg_color="black", stroke_color="", font='Noto-Mono', fontsize=85, color='white', stroke_width=1.5, size=(len(txt)*54, 95))
        # WITH TEXT OUTLINE: 
        # generator = lambda txt: TextClip(txt, stroke_color="black", font='Source-Code-Pro-Black', fontsize=340, color='yellow', stroke_width=5).resize(0.23)    
        # MADE BY CHATGPT:
        generator = lambda txt: TextClip(txt, fontsize=90, font='Impact', color='white', stroke_color='black', stroke_width=2)
        
        subs = SubtitlesClip(ImageFolderPath+"Subtitles.mp3.srt", generator)
        subtitles = SubtitlesClip(subs, generator)

        # MIGHT NEED TO RE-INSTALL IMAGE MAGICK WITH ALL BOXES CHECKED (except last 2, but LEGACY FEATURES needs ot be installed)
        # Useful info: https://moviepy-tburrows13.readthedocs.io/en/improve-docs/ref/VideoClip/TextClip.html
        result = CompositeVideoClip([final_videoBLURRED, subtitles.set_position(('center', 0.7), relative=True)])
        result.write_videofile(ImageFolderPath + "VideoSUBBED.mp4")

