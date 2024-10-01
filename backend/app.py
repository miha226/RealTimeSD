import cv2 as cv
import numpy as np
from PIL import Image
from fastapi import FastAPI, WebSocket
from fastapi.websockets import WebSocketDisconnect
from diffusers import (AutoPipelineForImage2Image, StableDiffusionControlNetPipeline,
                       ControlNetModel)
import torch
import os
import asyncio

app = FastAPI()

### Constants and Functions ###

DEFAULT_PROMPT = "portrait of a minion, wearing goggles, yellow skin, wearing a beanie, despicable me movie, in the style of pixar movie"
MODEL = "sdxlturbo"
SDXLTURBO_MODEL_LOCATION = 'models/sdxl-turbo'
LCM_MODEL_LOCATION = 'models/LCM_Dreamshaper_v7'
CONTROLNET_CANNY_LOCATION = "models/control_v11p_sd15_canny"
TORCH_DEVICE, TORCH_DTYPE = None, None  # Will be set later
GUIDANCE_SCALE = 3
INFERENCE_STEPS = 4
DEFAULT_NOISE_STRENGTH = 0.7
CONDITIONING_SCALE = 0.7
GUIDANCE_START = 0.0
GUIDANCE_END = 1.0
RANDOM_SEED = 21
HEIGHT = 384
WIDTH = 384

# Helper functions for model initialization and processing
def choose_device(torch_device=None):
    global TORCH_DEVICE, TORCH_DTYPE
    if torch_device is None:
        if torch.cuda.is_available():
            torch_device = "cuda"
            torch_dtype = torch.float16
        elif torch.backends.mps.is_available() and not torch.cuda.is_available():
            torch_device = "mps"
            torch_dtype = torch.float16
        else:
            torch_device = "cpu"
            torch_dtype = torch.float32
    TORCH_DEVICE = torch_device
    TORCH_DTYPE = torch_dtype

def prepare_seed():
    generator = torch.manual_seed(RANDOM_SEED)
    return generator

def convert_numpy_image_to_pil_image(image):
    return Image.fromarray(cv.cvtColor(image, cv.COLOR_BGR2RGB))

def process_lcm(image):
    image = np.array(image)
    image = cv.Canny(image, 100, 200)
    image = np.repeat(image[:, :, np.newaxis], 3, axis=2)
    return image

def process_sdxlturbo(image):
    return image

def prepare_pipeline():
    if MODEL == "lcm":
        controlnet = ControlNetModel.from_pretrained(CONTROLNET_CANNY_LOCATION, torch_dtype=TORCH_DTYPE, use_safetensors=True)
        pipeline = StableDiffusionControlNetPipeline.from_pretrained(
            LCM_MODEL_LOCATION, controlnet=controlnet, torch_dtype=TORCH_DTYPE, safety_checker=None
        ).to(TORCH_DEVICE)
    elif MODEL == "sdxlturbo":
        pipeline = AutoPipelineForImage2Image.from_pretrained(
            SDXLTURBO_MODEL_LOCATION, torch_dtype=TORCH_DTYPE, safety_checker=None
        ).to(TORCH_DEVICE)
    return pipeline

def run_model(pipeline, ref_image):
    generator = prepare_seed()
    if MODEL == "lcm":
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
    else:
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


# WebSocket Endpoint for real-time image processing
@app.websocket("/ws/process")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    choose_device()  # Set up device for torch (CUDA/CPU/MPS)
    pipeline = prepare_pipeline()  # Initialize the pipeline

    try:
        while True:
            # Receive image from client
            data = await websocket.receive_bytes()
            np_arr = np.frombuffer(data, np.uint8)
            frame = cv.imdecode(np_arr, cv.IMREAD_COLOR)

            # Process the frame
            center_x = (frame.shape[1] - WIDTH) // 2
            center_y = (frame.shape[0] - HEIGHT) // 2
            cutout = frame[center_y:center_y + HEIGHT, center_x:center_x + WIDTH]

            processed_frame = process_sdxlturbo(cutout)  # You can switch to LCM model
            pil_image = convert_numpy_image_to_pil_image(processed_frame)
            generated_image = run_model(pipeline, pil_image)

            # Convert back to numpy and send to the client
            generated_image_np = np.array(generated_image)
            result_frame = frame.copy()
            result_frame[center_y:center_y + HEIGHT, center_x:center_x + WIDTH] = cv.cvtColor(generated_image_np, cv.COLOR_RGB2BGR)

            _, buffer = cv.imencode('.jpg', result_frame)
            await websocket.send_bytes(buffer.tobytes())

    except WebSocketDisconnect:
        print("Client disconnected")