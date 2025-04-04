from groq import Groq
import base64
from pydantic_ai import Agent
from pathlib import Path
from pydantic import BaseModel
import json

LLM_CLIENT = Groq()

class VideoDescription(BaseModel):
    description: str

# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')


def get_system_prompt(location:str)->str:
  return f"""
You are a helpful AI assistant, who is an expert in describing the image. 
Carefully describe the given image. 
Additionally, incorporate the location information ({location}) to provide relevant contextual details.
"""

def get_image_description(system_prompt: str, base64_image: bytes, llm_client=LLM_CLIENT) -> str:
    chat_completion = llm_client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": [
                    # {"type": "text", "text": f"Describe the given image. This image is taken at {metadata['location']}. Use the location information also in the description"},
                    {"type": "text", "text": system_prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                        },
                    },
                ],
            }
        ],
        model="llama-3.2-11b-vision-preview",
    )

    return chat_completion.choices[0].message.content

def write_all_frame_descriptions(file_directory: Path):

    frames_directory = file_directory / "frames"
    frame_description_save_directory = file_directory / "descriptions"
    frame_description_save_directory.mkdir(exist_ok=True)

    with open(file_directory / "metadata.json", "r") as f:
        metadata = json.load(f)

    for image_path in frames_directory.iterdir():
        base64_image = encode_image(image_path)
        location = metadata['location']
        system_prompt = get_system_prompt(location)
        frame_description = get_image_description(system_prompt, base64_image)
        frame_description_file_path = frame_description_save_directory / f"{image_path.stem}.txt"
        with open(frame_description_file_path, "w") as f:
            f.write(frame_description)
        print(f"Frame description written to {frame_description_file_path}")
    
    return None


def get_video_description(file_directory, llm_model_name="groq:llama-3.3-70b-versatile") -> str:

    class VideoDescription(BaseModel):
        description: str

    system_prompt = """You are an AI assistant specializing in generating coherent and engaging video descriptions using frame-by-frame image descriptions.
    Your task is to analyze a sequence of frame descriptions and craft a structured, natural-sounding video description that flows smoothly. 
    Ensure the description captures key elements such as objects, actions, emotions, transitions, and location details while maintaining logical continuity between frames."""

    agent = Agent(llm_model_name, system_prompt=system_prompt, result_type=VideoDescription)

    frame_description_directory = file_directory / "descriptions"

    user_prompt = ""
    for frame_description_filepath in frame_description_directory.iterdir():
        frame_number = frame_description_filepath.stem[-2:]
        with open(frame_description_filepath, "r") as f:
            frame_description = f.read()
        user_prompt += f"Frame {frame_number} : \n{frame_description}\n\n"

    video_description = agent.run_sync(user_prompt)
    print(video_description)
    print(video_description.data.description)
    return video_description.data.description