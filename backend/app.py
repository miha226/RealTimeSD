import cv2 as cv
import numpy as np
from PIL import Image
from diffusers import (
    StableDiffusionXLAdapterPipeline,
    T2IAdapter,
    EulerAncestralDiscreteScheduler,
    AutoencoderTiny
)
from controlnet_aux import MidasDetector  # Changed depth estimator
import torch
from fastapi import FastAPI, WebSocket, HTTPException, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import threading
from starlette.websockets import WebSocketState
from DeepCache import DeepCacheSDHelper
from pydantic import BaseModel
from typing import Optional

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
depth_estimator = None
feature_extractor = None
run_model = None

DEFAULT_PROMPT = "photo of city glass skyscrapers, 4K, realistic, smooth transition"
NEGATIVE_PROMPT = "anime, cartoon, graphic, text, painting, crayon, graphite, abstract, glitch, deformed, mutated, ugly, disfigured"  # Added negative prompt
MODEL = "sdxlturbo"  # "lcm" or "sdxlturbo"
SDXLTURBO_MODEL_LOCATION = 'stabilityai/stable-diffusion-xl-base-1.0'  # Updated to model ID used in T2I Adapter
TORCH_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VARIANT = "fp16"
TORCH_DTYPE = torch.float16
GUIDANCE_SCALE = 0  # Increased guidance scale for better adherence
INFERENCE_STEPS = 2  # Increased inference steps from 2 to 30
DEFAULT_NOISE_STRENGTH = 0.5  # 0.5 or 0.7 works well too
CONDITIONING_SCALE = 1.0  # Updated conditioning scale based on example
RANDOM_SEED = 21
HEIGHT = 512
WIDTH = 512

# Create threading locks for the pipeline and settings
pipeline_lock = threading.Lock()
settings_lock = threading.Lock()

# Define the settings Pydantic model
class Settings(BaseModel):
    prompt: Optional[str] = None
    negative_prompt: Optional[str] = None  # Added negative prompt
    seed: Optional[int] = None
    inference_steps: Optional[int] = None
    noise_strength: Optional[float] = None
    conditioning_scale: Optional[float] = None

def prepare_seed(seed: int):
    generator = torch.Generator(device=TORCH_DEVICE)
    generator.manual_seed(seed)
    print(f"Random seed prepared: {seed}")
    return generator

def convert_numpy_image_to_pil_image(image):
    try:
        pil_image = Image.fromarray(cv.cvtColor(image, cv.COLOR_BGR2RGB))
        print("Converted numpy array to PIL Image.")
        return pil_image
    except Exception as e:
        print(f"Error converting numpy image to PIL image: {e}")
        return None

def get_depth_map(image):
    try:
        # Preprocess the image for depth estimation
        depth = depth_estimator(image, detect_resolution=512, image_resolution=1024)
        print("Depth map generated successfully.")
        return depth
    except Exception as e:
        print(f"Error generating depth map: {e}")
        return None

def prepare_t2i_adapter_pipeline():
    try:
        # Load the T2I Adapter
        adapter = T2IAdapter.from_pretrained(
            "TencentARC/t2i-adapter-depth-midas-sdxl-1.0",
            torch_dtype=torch.float16,
            variant="fp16"
        ).to(TORCH_DEVICE)
        print("T2I Adapter loaded.")
    except Exception as e:
        print(f"Error loading T2I Adapter: {e}")
        return None

    try:
        # Load the Euler Ancestral Discrete Scheduler
        model_id = 'stabilityai/stable-diffusion-xl-base-1.0'
        euler_a = EulerAncestralDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
        print("Scheduler loaded.")
    except Exception as e:
        print(f"Error loading scheduler: {e}")
        return None

    try:
        # Load the VAE
        vae = AutoencoderTiny.from_pretrained(
            'madebyollin/taesdxl',
            use_safetensors=True,
            torch_dtype=torch.float16,
        ).to(TORCH_DEVICE)
        print("VAE model loaded.")
    except Exception as e:
        print(f"Error loading VAE model: {e}")
        return None

    try:
        # Initialize the StableDiffusionXLAdapterPipeline
        pipe = StableDiffusionXLAdapterPipeline.from_pretrained(
            model_id,
            vae=vae,
            adapter=adapter,
            scheduler=euler_a,
            torch_dtype=TORCH_DTYPE,
            variant="fp16",
            use_safetensors=True
        ).to(TORCH_DEVICE)
        pipe.enable_xformers_memory_efficient_attention()
        print("StableDiffusionXLAdapterPipeline loaded and moved to device.")

        # Initialize DeepCache if needed (optional)
        # helper = DeepCacheSDHelper(pipe=pipe)
        # helper.set_params(cache_interval=3, cache_branch_id=0)
        # helper.enable()

        return pipe
    except Exception as e:
        print(f"Error loading pipeline: {e}")
        return None

def run_t2i_adapter(pipeline, source_image, control_image, generator, prompt, negative_prompt, num_inference_steps, conditioning_scale):
    try:
        gen_image = pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=source_image,  # Pass the source image here
            num_inference_steps=num_inference_steps,
            adapter_conditioning_scale=conditioning_scale,  # Utilize the depth map via adapter
            guidance_scale=GUIDANCE_SCALE,
            generator=generator
        ).images[0]
        print("Pipeline successfully generated image.")
        return gen_image
    except Exception as e:
        print(f"Error during pipeline execution: {e}")
        return None

@app.on_event("startup")
async def startup_event():
    global pipeline, depth_estimator, feature_extractor, run_model

    print("FastAPI application is starting... Loading models into GPU.")
    try:
        # Load depth estimator (MidasDetector)
        depth_estimator = MidasDetector.from_pretrained(
            "valhalla/t2iadapter-aux-models",
            filename="dpt_large_384.pt",
            model_type='dpt_large'
        ).to(TORCH_DEVICE)
        print("MidasDetector loaded.")
    except Exception as e:
        print(f"Error loading MidasDetector: {e}")
        raise e

    try:
        pipeline = prepare_t2i_adapter_pipeline()
        if pipeline is None:
            print("Failed to load pipeline.")
            raise RuntimeError("Pipeline loading failed.")
        else:
            print("Pipeline loaded successfully.")
    except Exception as e:
        print(f"Exception during pipeline loading: {e}")
        raise e

    # Assign run_model based on the model type
    run_model = run_t2i_adapter
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
        # Read current settings
        with settings_lock:
            prompt = DEFAULT_PROMPT
            negative_prompt = NEGATIVE_PROMPT  # Include negative prompt
            seed = RANDOM_SEED
            inference_steps = INFERENCE_STEPS
            noise_strength = DEFAULT_NOISE_STRENGTH
            conditioning_scale = CONDITIONING_SCALE

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

        # Generate the depth map as the control image
        control_image = get_depth_map(pil_source_image)
        if control_image is None:
            print("Depth map generation failed.")
            return None
        else:
            print("Depth map generated successfully.")

        # Prepare a per-thread random number generator
        generator = prepare_seed(seed)

        # Run Stable Diffusion on the image with thread safety
        with pipeline_lock:
            gen_image = run_model(
                pipeline,
                pil_source_image,      # Pass the source image here
                control_image,        # Ensure the adapter uses this control image
                generator,
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=inference_steps,
                conditioning_scale=conditioning_scale,
            )
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

#################
################# NEW ADDITIONS
#################

# POST endpoint to update settings
@app.post("/settings")
def update_settings(settings: Settings):
    """
    Update the generation settings. Clients can send any combination of the following:
    - prompt (str)
    - negative_prompt (str)
    - seed (int)
    - inference_steps (int)
    - noise_strength (float)
    - conditioning_scale (float)

    If a setting is not provided, the default value remains unchanged.
    """
    global DEFAULT_PROMPT, NEGATIVE_PROMPT, RANDOM_SEED, INFERENCE_STEPS, DEFAULT_NOISE_STRENGTH, CONDITIONING_SCALE
    with settings_lock:
        if settings.prompt is not None:
            DEFAULT_PROMPT = settings.prompt
            print(f"Updated prompt to: {DEFAULT_PROMPT}")
        if settings.negative_prompt is not None:
            NEGATIVE_PROMPT = settings.negative_prompt
            print(f"Updated negative prompt to: {NEGATIVE_PROMPT}")
        if settings.seed is not None:
            RANDOM_SEED = settings.seed
            print(f"Updated seed to: {RANDOM_SEED}")
        if settings.inference_steps is not None:
            INFERENCE_STEPS = settings.inference_steps
            print(f"Updated inference steps to: {INFERENCE_STEPS}")
        if settings.noise_strength is not None:
            DEFAULT_NOISE_STRENGTH = settings.noise_strength
            print(f"Updated noise strength to: {DEFAULT_NOISE_STRENGTH}")
        if settings.conditioning_scale is not None:
            CONDITIONING_SCALE = settings.conditioning_scale
            print(f"Updated conditioning scale to: {CONDITIONING_SCALE}")
    return {"status": "Settings updated successfully."}
