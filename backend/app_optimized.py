import cv2 as cv
import numpy as np
from PIL import Image
from diffusers import (StableDiffusionXLControlNetImg2ImgPipeline, ControlNetModel, AutoencoderKL)
import torch
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import threading
from starlette.websockets import WebSocketState

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

DEFAULT_PROMPT = "photo of city glass skyscrapers, 4K, realistic, smooth transition"
MODEL = "sdxlturbo"  # "lcm" or "sdxlturbo"
SDXLTURBO_MODEL_LOCATION = 'models/sdxl-turbo'
TORCH_DEVICE = "cuda"
VARIANT = "fp16"
TORCH_DTYPE = torch.float16
GUIDANCE_SCALE = 3  # 0 for sdxl turbo (hardcoded already)
INFERENCE_STEPS = 2  # 4 for lcm (high quality), 2 for turbo
DEFAULT_NOISE_STRENGTH = 0.7  # 0.5 or 0.7 works well too
CONDITIONING_SCALE = 0.7  # 0.5 or 0.7 works well too
GUIDANCE_START = 0.0
GUIDANCE_END = 1.0
RANDOM_SEED = 21
HEIGHT = 512
WIDTH = 512

# Create a threading lock for the pipeline
pipeline_lock = threading.Lock()

def prepare_seed():
    generator = torch.Generator(device=TORCH_DEVICE)
    generator.manual_seed(RANDOM_SEED)
    print("Random seed prepared.")
    return generator

def convert_numpy_image_to_pil_image(image):
    try:
        pil_image = Image.fromarray(cv.cvtColor(image, cv.COLOR_BGR2RGB))
        print("Converted numpy array to PIL Image.")
        return pil_image
    except Exception as e:
        print(f"Error converting numpy image to PIL image: {e}")
        return None

def process_sdxlturbo(image, lower_threshold=100, upper_threshold=200, aperture=3):
    try:
        image = np.array(image)
        print(f"Original image shape: {image.shape}")
        image = cv.Canny(image, lower_threshold, upper_threshold, apertureSize=aperture)
        print("Canny edge detection applied.")
        image = np.repeat(image[:, :, np.newaxis], 3, axis=2)  # Ensure 3 channels
        print(f"Processed image shape after Canny: {image.shape}")
        return image
    except Exception as e:
        print(f"Error in process_sdxlturbo: {e}")
        return None

def prepare_sdxlturbo_pipeline():
    try:
        controlnet = ControlNetModel.from_pretrained(
            "diffusers/controlnet-canny-sdxl-1.0-small",
            torch_dtype=torch.float16
        )
        print("ControlNet model loaded.")
    except Exception as e:
        print(f"Error loading ControlNet model: {e}")
        return None

    try:
        vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
        print("VAE model loaded.")
    except Exception as e:
        print(f"Error loading VAE model: {e}")
        return None

    try:
        pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
            SDXLTURBO_MODEL_LOCATION,
            controlnet=controlnet,
            vae=vae,
            variant=VARIANT,
            use_safetensors=True,
            torch_dtype=TORCH_DTYPE,
        ).to(TORCH_DEVICE)
        print("Pipeline loaded and moved to device.")
        return pipe
    except Exception as e:
        print(f"Error loading pipeline: {e}")
        return None

def run_sdxlturbo(pipeline, source_image, control_image, generator):
    try:
        gen_image = pipeline(
            prompt=DEFAULT_PROMPT,
            num_inference_steps=INFERENCE_STEPS,
            guidance_scale=GUIDANCE_SCALE,  # Higher guidance scale for better prompt adherence
            width=WIDTH,
            height=HEIGHT,
            generator=generator,
            image=source_image,                # Source image
            control_image=control_image,        # Control image
            strength=DEFAULT_NOISE_STRENGTH,
            controlnet_conditioning_scale=CONDITIONING_SCALE,
        ).images[0]
        print("Pipeline successfully generated image.")
        return gen_image
    except Exception as e:
        print(f"Error during pipeline execution: {e}")
        return None

@app.on_event("startup")
async def startup_event():
    global pipeline, processor, run_model

    print("FastAPI application is starting... Loading models into GPU.")
    try:
        pipeline = prepare_sdxlturbo_pipeline()
        if pipeline is None:
            print("Failed to load pipeline.")
            raise RuntimeError("Pipeline loading failed.")
        else:
            print("Pipeline loaded successfully.")
    except Exception as e:
        print(f"Exception during pipeline loading: {e}")
        raise e

    # Assign processor and run_model based on the model type
    processor = process_sdxlturbo
    run_model = run_sdxlturbo
    print("Models loaded successfully.")

# Set a concurrency limit
MAX_CONCURRENT_PROCESSES = 4  # Adjust based on your GPU capacity
processing_semaphore = asyncio.Semaphore(MAX_CONCURRENT_PROCESSES)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket connection accepted.")
    tasks = []
    try:
        while True:
            # Receive the frame as bytes from the client
            data = await websocket.receive_bytes()
            print("Received data from client.")

            # Start a task to process the frame
            task = asyncio.create_task(process_and_send_frame(data, websocket))
            tasks.append(task)

    except WebSocketDisconnect:
        print("Client disconnected.")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Cancel all pending tasks
        for task in tasks:
            task.cancel()
        # Close the WebSocket if it's still open
        if websocket.application_state != WebSocketState.DISCONNECTED:
            await websocket.close()
        print("WebSocket closed.")

async def process_and_send_frame(data, websocket):
    try:
        async with processing_semaphore:
            # Run the processing in a separate thread to avoid blocking the event loop
            result = await asyncio.to_thread(process_frame, data)

            # Send the processed frame back to the client
            if result is not None:
                print("Sending processed frame back to the client.")
                try:
                    if websocket.application_state == WebSocketState.CONNECTED:
                        await websocket.send_bytes(result)
                        print("Processed frame sent to client.")
                    else:
                        print("WebSocket is closed. Cannot send data.")
                except Exception as e:
                    print(f"Failed to send data: {e}")
    except asyncio.CancelledError:
        print("Task was cancelled.")
    except Exception as e:
        print(f"Exception in process_and_send_frame: {e}")

def process_frame(data):
    try:
        # Convert the received bytes to a numpy array
        nparr = np.frombuffer(data, np.uint8)
        print("Converted received data to numpy array.")

        # Decode the numpy array to an image (OpenCV)
        frame = cv.imdecode(nparr, cv.IMREAD_COLOR)
        if frame is None:
            print("Failed to decode image from received data.")
            return None
        else:
            print("Image successfully decoded.")

        # Convert the source image to a PIL Image
        pil_source_image = convert_numpy_image_to_pil_image(frame)
        if pil_source_image is None:
            print("Conversion of source image to PIL image failed.")
            return None
        else:
            print("Conversion of source image to PIL image successful.")

        # Process the frame to generate the control image (Canny edges)
        control_image_np = processor(frame)  # Canny edge image as numpy array
        if control_image_np is None:
            print("Processor returned None.")
            return None
        else:
            print("Image processed by processor.")

        # Convert the control image to a PIL Image
        pil_control_image = convert_numpy_image_to_pil_image(control_image_np)
        if pil_control_image is None:
            print("Conversion of control image to PIL image failed.")
            return None
        else:
            print("Conversion of control image to PIL image successful.")

        # Prepare a per-thread random number generator
        generator = prepare_seed()
        print("Random seed prepared.")

        # Run Stable Diffusion on the image with thread safety
        with pipeline_lock:
            gen_image = run_model(pipeline, pil_source_image, pil_control_image, generator)
            if gen_image is None:
                print("Generated image is None.")
                return None
            else:
                print("Image successfully generated by pipeline.")

        # Convert the generated image back to a numpy array and encode it as JPEG
        result_image = np.array(gen_image)
        _, buffer = cv.imencode('.jpg', cv.cvtColor(result_image, cv.COLOR_RGB2BGR))
        print("Generated image encoded to JPEG format.")

        return buffer.tobytes()

    except Exception as e:
        print(f"Error in process_frame: {e}")
        return None
