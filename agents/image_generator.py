from agents import BaseAgent
from tools.openai_dall_e_api import openai_dall_e_api

class ImageGenerator(BaseAgent):
    def __init__(self):
        system_prompt = '''You are an image_generator, a ReAct agent that can generate images using the OpenAI DALL-E 3 API.
        You have a tool to send prompts to the DALL-E API which will save the generated images.
        As necessary, you can choose to not generate the image if you feel you have not been given sufficient description. In that case, ask for more information.
        '''
        tools = [openai_dall_e_api]
        super().__init__('image_generator', tools, system_prompt)

image_generator = ImageGenerator().invoke