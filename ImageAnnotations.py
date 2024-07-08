from time import sleep

from openai import OpenAI
import base64
import requests
import os
import parsecharacters

API_KEY = os.getenv('API_KEY')
client = OpenAI(api_key=API_KEY)


# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

file_object = open("charactersFromImage.txt", "a")
folder = "illustrations/"
for image in os.listdir(folder):
    print(f"image: {image}", flush=True)

    if image not in parsecharacters.get_processed() and image[-4:] == 'jpeg':

        # if os.path.isfile(os.path.join(folder, image)):
        image_path = "illustrations/" + image
        # Getting the base64 string
        base64_image = encode_image(image_path)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

        payload = {
            "model": "gpt-4-vision-preview",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "List each character in this image. List the character by the best-fitting specific title (e.g. businessman, woman, scientist, teenager, dog). The characters should be separated by commas."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 300
        }

        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

        json = response.json()
        print(json)

        content = json['choices'][0]['message']['content']

        print(f"{image}: {content}")
        file_object.write(f"{image}: {content}")
        file_object.write("\n")
        sleep(6)
