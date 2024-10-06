import cv2 as cv
import numpy as np
from PIL import Image
from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    UniPCMultistepScheduler
)
import torch
from fastapi import FastAPI, WebSocket, HTTPException, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import threading
from starlette.websockets import WebSocketState
from transformers import AutoImageProcessor, UperNetForSemanticSegmentation
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

# Define the color palette for segmentation
palette = np.asarray([
    [0, 0, 0],
    [120, 120, 120],
    [180, 120, 120],
    [6, 230, 230],
    [80, 50, 50],
    [4, 200, 3],
    [120, 120, 80],
    [140, 140, 140],
    [204, 5, 255],
    [230, 230, 230],
    [4, 250, 7],
    [224, 5, 255],
    [235, 255, 7],
    [150, 5, 61],
    [120, 120, 70],
    [8, 255, 51],
    [255, 6, 82],
    [143, 255, 140],
    [204, 255, 4],
    [255, 51, 7],
    [204, 70, 3],
    [0, 102, 200],
    [61, 230, 250],
    [255, 6, 51],
    [11, 102, 255],
    [255, 7, 71],
    [255, 9, 224],
    [9, 7, 230],
    [220, 220, 220],
    [255, 9, 92],
    [112, 9, 255],
    [8, 255, 214],
    [7, 255, 224],
    [255, 184, 6],
    [10, 255, 71],
    [255, 41, 10],
    [7, 255, 255],
    [224, 255, 8],
    [102, 8, 255],
    [255, 61, 6],
    [255, 194, 7],
    [255, 122, 8],
    [0, 255, 20],
    [255, 8, 41],
    [255, 5, 153],
    [6, 51, 255],
    [235, 12, 255],
    [160, 150, 20],
    [0, 163, 255],
    [140, 140, 140],
    [250, 10, 15],
    [20, 255, 0],
    [31, 255, 0],
    [255, 31, 0],
    [255, 224, 0],
    [153, 255, 0],
    [0, 0, 255],
    [255, 71, 0],
    [0, 235, 255],
    [0, 173, 255],
    [31, 0, 255],
    [11, 200, 200],
    [255, 82, 0],
    [0, 255, 245],
    [0, 61, 255],
    [0, 255, 112],
    [0, 255, 133],
    [255, 0, 0],
    [255, 163, 0],
    [255, 102, 0],
    [194, 255, 0],
    [0, 143, 255],
    [51, 255, 0],
    [0, 82, 255],
    [0, 255, 41],
    [0, 255, 173],
    [10, 0, 255],
    [173, 255, 0],
    [0, 255, 153],
    [255, 92, 0],
    [255, 0, 255],
    [255, 0, 245],
    [255, 0, 102],
    [255, 173, 0],
    [255, 0, 20],
    [255, 184, 184],
    [0, 31, 255],
    [0, 255, 61],
    [0, 71, 255],
    [255, 0, 204],
    [0, 255, 194],
    [0, 255, 82],
    [0, 10, 255],
    [0, 112, 255],
    [51, 0, 255],
    [0, 194, 255],
    [0, 122, 255],
    [0, 255, 163],
    [255, 153, 0],
    [0, 255, 10],
    [255, 112, 0],
    [143, 255, 0],
    [82, 0, 255],
    [163, 255, 0],
    [255, 235, 0],
    [8, 184, 170],
    [133, 0, 255],
    [0, 255, 92],
    [184, 0, 255],
    [255, 0, 31],
    [0, 184, 255],
    [0, 214, 255],
    [255, 0, 112],
    [92, 255, 0],
    [0, 224, 255],
    [112, 224, 255],
    [70, 184, 160],
    [163, 0, 255],
    [153, 0, 255],
    [71, 255, 0],
    [255, 0, 163],
    [255, 204, 0],
    [255, 0, 143],
    [0, 255, 235],
    [133, 255, 0],
    [255, 0, 235],
    [245, 0, 255],
    [255, 0, 122],
    [255, 245, 0],
    [10, 190, 212],
    [214, 255, 0],
    [0, 204, 255],
    [20, 0, 255],
    [255, 255, 0],
    [0, 153, 255],
    [0, 41, 255],
    [0, 255, 204],
    [41, 0, 255],
    [41, 255, 0],
    [173, 0, 255],
    [0, 245, 255],
    [71, 0, 255],
    [122, 0, 255],
    [0, 255, 184],
    [0, 92, 255],
    [184, 255, 0],
    [0, 133, 255],
    [255, 214, 0],
    [25, 194, 194],
    [102, 255, 0],
    [92, 0, 255],
])

# Global variables to hold models
pipeline = None
segmentation_model = None
segmentation_processor = None
run_model = None

DEFAULT_PROMPT = "photo of city glass skyscrapers, 4K, realistic, smooth transition"
MODEL = "sdxlturbo"  # "lcm" or "sdxlturbo"
SDXLTURBO_MODEL_LOCATION = 'models/sdxl-turbo'  # Update if necessary
TORCH_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VARIANT = "fp16"
TORCH_DTYPE = torch.float16
GUIDANCE_SCALE = 7.5  # Typically between 7.5 and 12 for better prompt adherence
INFERENCE_STEPS = 20  # Increased steps for higher quality
DEFAULT_NOISE_STRENGTH = 0.5  # 0.5 or 0.7 works well
CONDITIONING_SCALE = 1.0  # Adjust as needed
GUIDANCE_START = 0.0
GUIDANCE_END = 1.0
RANDOM_SEED = 21
HEIGHT = 512
WIDTH = 512

# Create threading locks for the pipeline and settings
pipeline_lock = threading.Lock()
settings_lock = threading.Lock()

# Define the settings Pydantic model
class Settings(BaseModel):
    prompt: Optional[str] = None
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

def get_segmentation_map(image):
    try:
        # Preprocess the image for segmentation
        inputs = segmentation_processor(images=image, return_tensors="pt").pixel_values.to(TORCH_DEVICE)
        with torch.no_grad():
            outputs = segmentation_model(inputs)
        
        # Post-process the segmentation output
        seg = segmentation_processor.post_process_semantic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
        
        # Move the segmentation tensor to CPU and convert to NumPy
        if isinstance(seg, torch.Tensor):
            seg = seg.cpu().numpy()
        
        # Create a color segmentation map
        color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)  # height, width, 3
        for label, color in enumerate(palette):
            color_seg[seg == label, :] = color[:3]  # Ensure color has 3 channels
        
        color_seg = color_seg.astype(np.uint8)
        segmentation_image = Image.fromarray(color_seg)
        print("Segmentation map generated successfully.")
        return segmentation_image
    except Exception as e:
        print(f"Error generating segmentation map: {e}")
        return None

def prepare_segmentation_pipeline():
    try:
        # Load the segmentation ControlNet model
        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-seg", 
            torch_dtype=torch.float16
        )
        print("Segmentation ControlNet model loaded.")
    except Exception as e:
        print(f"Error loading Segmentation ControlNet model: {e}")
        return None

    try:
        # Load the main Stable Diffusion pipeline
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", 
            controlnet=controlnet, 
            safety_checker=None, 
            torch_dtype=torch.float16
        ).to(TORCH_DEVICE)

        # Set the scheduler
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        print("Pipeline and scheduler loaded.")
    except Exception as e:
        print(f"Error loading Stable Diffusion ControlNet pipeline: {e}")
        return None

    # Optional: Enable memory-efficient attention if xformers is installed
    try:
        pipe.enable_xformers_memory_efficient_attention()
        print("xformers memory-efficient attention enabled.")
    except Exception:
        print("xformers not installed or could not be enabled. Skipping memory-efficient attention.")

    # Enable model CPU offloading to save GPU memory
    try:
        pipe.enable_model_cpu_offload()
        print("Model CPU offloading enabled.")
    except Exception as e:
        print(f"Error enabling model CPU offloading: {e}")

    # Compile the pipeline if desired (optional)
    # Uncomment and modify if you wish to use the 'sfast' compiler as in your original code
    """
    config = CompilationConfig.Default()
    # Configure compiler settings as needed
    try:
        pipe = compile(pipe, config)
        print("Pipeline compiled with optimizations.")
    except ImportError:
        print("Compiler module 'sfast' not found. Skipping pipeline compilation.")
    except Exception as e:
        print(f"Error during pipeline compilation: {e}")
    """

    return pipe

def run_segmentation_model(pipeline, source_image, control_image, generator, prompt, num_inference_steps, strength, conditioning_scale):
    try:
        gen_image = pipeline(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=GUIDANCE_SCALE,  # Adjust as needed
            width=WIDTH,
            height=HEIGHT,
            generator=generator,
            image=source_image,                # Source image
            control_image=control_image,        # Control image (segmentation map)
            strength=strength,
            controlnet_conditioning_scale=conditioning_scale,
        ).images[0]
        print("Pipeline successfully generated image using segmentation ControlNet.")
        return gen_image
    except Exception as e:
        print(f"Error during pipeline execution: {e}")
        return None

@app.on_event("startup")
async def startup_event():
    global pipeline, segmentation_model, segmentation_processor, run_model

    print("FastAPI application is starting... Loading models into GPU.")
    try:
        # Load segmentation model and processor
        segmentation_processor = AutoImageProcessor.from_pretrained("openmmlab/upernet-convnext-small")
        segmentation_model = UperNetForSemanticSegmentation.from_pretrained("openmmlab/upernet-convnext-small").to(TORCH_DEVICE)
        print("Segmentation model and processor loaded.")
    except Exception as e:
        print(f"Error loading segmentation model or processor: {e}")
        raise e

    try:
        pipeline = prepare_segmentation_pipeline()
        if pipeline is None:
            print("Failed to load pipeline.")
            raise RuntimeError("Pipeline loading failed.")
        else:
            print("Pipeline loaded successfully.")
    except Exception as e:
        print(f"Exception during pipeline loading: {e}")
        raise e

    # Assign run_model based on the model type
    run_model = run_segmentation_model
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

        # Generate the segmentation map as the control image
        control_image = get_segmentation_map(pil_source_image)
        if control_image is None:
            print("Segmentation map generation failed.")
            return None
        else:
            print("Segmentation map generated successfully.")

        # Prepare a per-thread random number generator
        generator = prepare_seed(seed)

        # Run Stable Diffusion on the image with thread safety
        with pipeline_lock:
            gen_image = run_model(
                pipeline,
                pil_source_image,
                control_image,
                generator,
                prompt=prompt,
                num_inference_steps=inference_steps,
                strength=noise_strength,
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
    - seed (int)
    - inference_steps (int)
    - noise_strength (float)
    - conditioning_scale (float)
    
    If a setting is not provided, the default value remains unchanged.
    """
    global DEFAULT_PROMPT, RANDOM_SEED, INFERENCE_STEPS, DEFAULT_NOISE_STRENGTH, CONDITIONING_SCALE
    with settings_lock:
        if settings.prompt is not None:
            DEFAULT_PROMPT = settings.prompt
            print(f"Updated prompt to: {DEFAULT_PROMPT}")
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
