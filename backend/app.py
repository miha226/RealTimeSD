import cv2 as cv
import numpy as np
from PIL import Image
from diffusers import (AutoPipelineForImage2Image, StableDiffusionControlNetPipeline,
                       ControlNetModel)
import torch
import os
import datetime
from fastapi import FastAPI, WebSocket
from fastapi.websockets import WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to your needs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#################
################# CONSTANTS
#################

# Global variables to hold models
pipeline = None
processor = None
run_model = None

DEFAULT_PROMPT = "portrait of a minion, wearing goggles, yellow skin, wearing a beanie, despicable me movie, in the style of pixar movie"
MODEL = "sdxlturbo"  # "lcm" or "sdxlturbo"
SDXLTURBO_MODEL_LOCATION = 'models/sdxl-turbo'
LCM_MODEL_LOCATION = 'models/LCM_Dreamshaper_v7'
CONTROLNET_CANNY_LOCATION = "models/control_v11p_sd15_canny"
TORCH_DEVICE = "cuda"
TORCH_DTYPE = torch.float16
GUIDANCE_SCALE = 3  # 0 for sdxl turbo (hardcoded already)
INFERENCE_STEPS = 2  # 4 for lcm (high quality), 2 for turbo
DEFAULT_NOISE_STRENGTH = 0.7  # 0.5 works well too
CONDITIONING_SCALE = 0.7  # 0.5 works well too
GUIDANCE_START = 0.0
GUIDANCE_END = 1.0
RANDOM_SEED = 21
HEIGHT = 512
WIDTH = 512

def prepare_seed():
    generator = torch.manual_seed(RANDOM_SEED)
    return generator

def convert_numpy_image_to_pil_image(image):
    return Image.fromarray(cv.cvtColor(image, cv.COLOR_BGR2RGB))

def process_lcm(image, lower_threshold=100, upper_threshold=100, aperture=3):
    image = np.array(image)
    image = cv.Canny(image, lower_threshold, upper_threshold, apertureSize=aperture)
    image = np.repeat(image[:, :, np.newaxis], 3, axis=2)
    return image

def process_sdxlturbo(image):
    return image

def prepare_lcm_controlnet_or_sdxlturbo_pipeline():
    if MODEL == "lcm":
        controlnet = ControlNetModel.from_pretrained(
            CONTROLNET_CANNY_LOCATION,
            torch_dtype=TORCH_DTYPE,
            use_safetensors=True
        )

        pipeline = StableDiffusionControlNetPipeline.from_pretrained(
            LCM_MODEL_LOCATION,
            controlnet=controlnet,
            torch_dtype=TORCH_DTYPE,
            safety_checker=None
        ).to(TORCH_DEVICE)

    elif MODEL == "sdxlturbo":
        pipeline = AutoPipelineForImage2Image.from_pretrained(
            SDXLTURBO_MODEL_LOCATION,
            torch_dtype=TORCH_DTYPE,
            safety_checker=None
        ).to(TORCH_DEVICE)

    return pipeline

def run_lcm(pipeline, ref_image):
    generator = prepare_seed()
    gen_image = pipeline(
        prompt=DEFAULT_PROMPT,
        num_inference_steps=INFERENCE_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        width=WIDTH,
        height=HEIGHT,
        generator=generator,
        image=ref_image,
        controlnet_conditioning_scale=CONDITIONING_SCALE,
        control_guidance_start=GUIDANCE_START,
        control_guidance_end=GUIDANCE_END,
    ).images[0]

    return gen_image

def run_sdxlturbo(pipeline, ref_image):
    generator = prepare_seed()
    gen_image = pipeline(
        prompt=DEFAULT_PROMPT,
        num_inference_steps=INFERENCE_STEPS,
        guidance_scale=0.0,
        width=WIDTH,
        height=HEIGHT,
        generator=generator,
        image=ref_image,
        strength=DEFAULT_NOISE_STRENGTH,
    ).images[0]

    return gen_image

# Initialize a global counter
image_counter = 0

@app.on_event("startup")
async def startup_event():
    global pipeline, processor, run_model

    print("FastAPI application is starting... Loading models into GPU.")
    # Load the models into GPU when the server starts
    pipeline = prepare_lcm_controlnet_or_sdxlturbo_pipeline()

    # Assign processor and run_model based on the model type
    processor = process_lcm if MODEL == "lcm" else process_sdxlturbo
    run_model = run_lcm if MODEL == "lcm" else run_sdxlturbo
    print("Models loaded successfully.")

    # Create 'pictures' directory if it doesn't exist
    if not os.path.exists('pictures'):
        os.makedirs('pictures')
        print("Created 'pictures' directory.")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    global image_counter  # Declare the counter as global
    try:
        while True:
            # Receive the frame as bytes from the client
            data = await websocket.receive_bytes()

            # Convert the received bytes to a numpy array
            nparr = np.frombuffer(data, np.uint8)

            # Decode the numpy array to an image (OpenCV)
            frame = cv.imdecode(nparr, cv.IMREAD_COLOR)

            # Check if the frame is valid
            if frame is None:
                continue

            # Process the frame (if any preprocessing is needed)
            numpy_image = processor(frame)
            pil_image = convert_numpy_image_to_pil_image(numpy_image)

            # Run Stable Diffusion on the image
            gen_image = run_model(pipeline, pil_image)

            # Convert the generated image back to a numpy array and encode it as JPEG
            result_image = np.array(gen_image)
            _, buffer = cv.imencode('.jpg', cv.cvtColor(result_image, cv.COLOR_RGB2BGR))

            # Send the processed frame back to the client
            print("Sending processed frame back to the client.")
            await websocket.send_bytes(buffer.tobytes())

            # Increment the image counter
            image_counter += 1

            # Save every tenth image
            if image_counter % 10 == 0:
                # Create a unique filename
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"image_{image_counter}_{timestamp}.jpg"
                filepath = os.path.join('pictures', filename)
                # Save the image
                gen_image.save(filepath)
                print(f"Saved image {filename} to 'pictures' folder.")

    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        await websocket.close()
