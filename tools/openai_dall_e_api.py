import os
import requests
from openai import OpenAI
from langchain_core.tools import tool

client = OpenAI()

@tool
def openai_dall_e_api(prompt: str, save_to_filepath: str = "image.jpeg") -> str:
    """Sends a request to the OpenAI DALL-E API and saves the resulting image at the given filepath. Prompt must fewer than 1000 chars"""
    if len(prompt) > 1000:
        return Exception(f"Expected no more than 1000 characters in the prompt, received {len(prompt)}")

    response = client.images.generate(
        prompt=prompt,
        model="dall-e-3",
        n=1,
        size="1024x1024",
        response_format="url",
    )
    image_url = response.data[0].url

    # Remove the initial filepath extension if present
    save_to_filepath = os.path.splitext(save_to_filepath)[0]

    image_response = requests.get(image_url)
    if image_response.status_code == 200:
        # Determine the file extension from the Content-Type header
        content_type = image_response.headers['Content-Type']
        extension = content_type.split('/')[-1]
        save_to_filepath = f"{save_to_filepath}.{extension}"

        # Save the image to the specified filepath
        with open(save_to_filepath, 'wb') as image_file:
            image_file.write(image_response.content)
    else:
        raise Exception(f"Failed to download image. Status code: {image_response.status_code}")
    
    return f"Image was saved to {save_to_filepath}"