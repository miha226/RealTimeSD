import cv2 as cv
import numpy as np
from PIL import Image
from diffusers import (
    StableDiffusionXLControlNetImg2ImgPipeline,
    ControlNetModel,
    AutoencoderKL
)
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import threading
from starlette.websockets import WebSocketState
from transformers import DPTImageProcessor, DPTForDepthEstimation
from DeepCache import DeepCacheSDHelper
from pydantic import BaseModel, Field  # Added Field for validation
from typing import Optional, List
import time
import random
import uuid

# Uncomment the following lines if you decide to use the 'sfast' compiler
from sfast.compilers.diffusion_pipeline_compiler import (compile, CompilationConfig)

# #########################
# 1. PyTorch Backend Settings
# #########################

# Disable gradient computations for inference
torch.set_grad_enabled(False)

# Enable TensorFloat-32 for faster matrix operations (supported on Ampere and newer GPUs)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# #########################
# 2. Initialize FastAPI
# #########################

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to your needs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# #########################
# 3. CONSTANTS AND GLOBAL VARIABLES
# #########################

# Global variables to hold models
pipeline = None
depth_estimator = None
feature_extractor = None
run_model = None

# Default settings
DEFAULT_PROMPT = "Beautiful modern urban landscape with a mix of sleek skyscrapers, green parks, and pedestrian-friendly streets. Buildings have glass facades and contemporary architecture. The area features well-designed open spaces, plazas, and pathways without any people or vehicles. Trees and gardens are integrated seamlessly into the city layout, with carefully placed greenery and water features. The scene is vibrant with sunlight reflecting off the buildings, highlighting the clean lines and modern design. The sky is clear and blue, emphasizing the harmony between nature and architecture."
MODEL = "sdxlturbo"  # "lcm" or "sdxlturbo"
SDXLTURBO_MODEL_LOCATION = 'models/sdxl-turbo'
TORCH_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VARIANT = "fp16"
TORCH_DTYPE = torch.float16
GUIDANCE_SCALE = 0.0  # Must be 0 for SDXL
INFERENCE_STEPS = 2  # 4 for lcm (high quality), 2 for turbo
DEFAULT_NOISE_STRENGTH = 0.5  # 0.5 or 0.7 works well too
CONDITIONING_SCALE = 0.5  # 0.5 or 0.7 works well too
RANDOM_SEED = 21
HEIGHT = 512
WIDTH = 512

# Batching parameters (default values)
BATCH_INTERVAL = 0.5  # seconds
MAX_BATCH_SIZE = 2    # Changed initial value from 12 to 2

# Create threading locks for the pipeline and settings
pipeline_lock = threading.Lock()
settings_lock = threading.Lock()

# Define the settings Pydantic model with validation for max_batch_size
class Settings(BaseModel):
    prompt: Optional[str] = None
    seed: Optional[int] = None
    inference_steps: Optional[int] = None
    noise_strength: Optional[float] = None
    conditioning_scale: Optional[float] = None
    batch_interval: Optional[float] = None  # New field
    max_batch_size: Optional[int] = Field(None, ge=1, le=10)    # New field with validation

# #########################
# 4. Helper Functions
# #########################

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
        inputs = feature_extractor(images=image, return_tensors="pt").pixel_values.to(TORCH_DEVICE)
        with torch.no_grad():
            with torch.autocast("cuda"):
                depth = depth_estimator(inputs).predicted_depth

        # Resize depth map to match desired size
        depth = torch.nn.functional.interpolate(
            depth.unsqueeze(1),
            size=(HEIGHT, WIDTH),
            mode="bicubic",
            align_corners=False,
        )
        # Normalize the depth map
        depth_min = torch.amin(depth, dim=[1, 2, 3], keepdim=True)
        depth_max = torch.amax(depth, dim=[1, 2, 3], keepdim=True)
        depth = (depth - depth_min) / (depth_max - depth_min)
        # Convert to 3 channels
        depth = torch.cat([depth] * 3, dim=1)
        # Convert to PIL Image
        depth = depth.permute(0, 2, 3, 1).cpu().numpy()[0]
        depth_image = Image.fromarray((depth * 255.0).clip(0, 255).astype(np.uint8))
        print("Depth map generated successfully.")
        return depth_image
    except Exception as e:
        print(f"Error generating depth map: {e}")
        return None

def prepare_sdxlturbo_pipeline():
    global pipeline
    try:
        # Load the depth ControlNet model
        controlnet = ControlNetModel.from_pretrained(
            "diffusers/controlnet-depth-sdxl-1.0-small",
            variant="fp16",
            use_safetensors=True,
            torch_dtype=torch.float16,
        )
        print("Depth ControlNet model loaded.")
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

        #helper = DeepCacheSDHelper(pipe=pipe)
        #helper.set_params(cache_interval=3, cache_branch_id=0)
        #helper.enable()

        print("Pipeline loaded and moved to device.")
    except Exception as e:
        print(f"Error loading pipeline: {e}")
        return None

    # #########################
    # 5. Enabling xformers and Triton (Optional)
    # #########################

    # Uncomment the following lines if you decide to use the 'sfast' compiler
   
    config = CompilationConfig.Default()

    # Attempt to enable xformers
    try:
        import xformers
        config.enable_xformers = True
        print("xformers enabled for pipeline.")
    except ImportError:
        print('xformers not installed, skipping xformers optimization.')

    # Attempt to enable triton
    try:
        import triton
        config.enable_triton = True
        print("Triton enabled for pipeline.")
    except ImportError:
        print('Triton not installed, skipping Triton optimization.')

    # Enable CUDA Graphs
    config.enable_cuda_graph = True
    print("CUDA Graphs enabled for pipeline.")

    # Additional compiler settings similar to maxperf.py
    config.enable_jit = True
    config.enable_jit_freeze = True
    config.trace_scheduler = True
    config.enable_cnn_optimization = True
    config.preserve_parameters = False
    config.prefer_lowp_gemm = True
   
    # #########################
    # 6. Compiling the Pipeline (Optional)
    # #########################

    try:
        pipe = compile(pipe, config)
        print("Pipeline compiled with optimizations.")
    except ImportError:
        print("Compiler module 'sfast' not found. Skipping pipeline compilation.")
    except Exception as e:
        print(f"Error during pipeline compilation: {e}")
    

    # If not using compiler, assign the pipeline directly
    pipeline = pipe
    return pipeline

def run_sdxlturbo_batch(
    pipeline,
    source_images: List[Image.Image],
    control_images: List[Image.Image],
    generators: List[torch.Generator],
    prompts: List[str],
    num_inference_steps: int,
    strengths: List[float],
    conditioning_scales: List[float]
):
    try:
        # Batch processing
        batch_size = len(source_images)
        assert batch_size == len(control_images) == len(generators) == len(prompts) == len(strengths) == len(conditioning_scales), "Batch size mismatch"

        # Ensure all strengths and conditioning_scales are identical
        if not (len(set(strengths)) == 1 and len(set(conditioning_scales)) == 1):
            raise ValueError("All strengths and conditioning scales must be identical for batched processing.")

        strength = strengths[0]
        conditioning_scale = conditioning_scales[0]
        guidance_scale = float(GUIDANCE_SCALE)  # Ensure it's a float

        # Use a single generator for the entire batch
        generator = generators[0]  # Assuming all generators are identical or you prefer the first one

        # Logging parameter details for debugging
        print(f"Batch size: {batch_size}")
        print(f"Prompts: {prompts}")
        print(f"Guidance scale: {guidance_scale}")
        print(f"Generator object: {generator}")
        print(f"Strength: {strength}")
        print(f"Conditioning scale: {conditioning_scale}")

        # Run the pipeline with scalar inputs for strength and guidance_scale
        with pipeline_lock:
            gen_images = pipeline(
                prompt=prompts,  # List[str]
                num_inference_steps=num_inference_steps,  # int
                guidance_scale=guidance_scale,  # Scalar float
                width=WIDTH,  # int
                height=HEIGHT,  # int
                generator=generator,  # Single torch.Generator
                image=source_images,  # List[Image.Image]
                control_image=control_images,  # List[Image.Image]
                strength=strength,  # Scalar float
                controlnet_conditioning_scale=conditioning_scale,  # Scalar float
            ).images
            print(f"Pipeline successfully generated {len(gen_images)} images.")
            return gen_images
    except Exception as e:
        import traceback
        print(f"Error during pipeline execution: {e}")
        print(traceback.format_exc())  # Print full traceback for debugging
        return [None] * len(source_images)

# #########################
# 7. FastAPI Event Handlers
# #########################

@app.on_event("startup")
async def startup_event():
    global pipeline, depth_estimator, feature_extractor, run_model

    print("FastAPI application is starting... Loading models into GPU.")
    try:
        # Load depth estimator and processor
        depth_estimator = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas").to(TORCH_DEVICE)
        feature_extractor = DPTImageProcessor.from_pretrained("Intel/dpt-hybrid-midas")
        print("Depth estimator and feature extractor loaded.")
    except Exception as e:
        print(f"Error loading depth estimator or feature extractor: {e}")
        raise e

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

    # Assign run_model based on the model type
    run_model = run_sdxlturbo_batch
    print("Models loaded successfully.")

    # Start the background batch processor
    asyncio.create_task(batch_processor())
    print("Background batch processor started.")

# #########################
# 8. Settings Endpoint
# #########################

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
    - batch_interval (float): Time window in seconds to collect batch requests
    - max_batch_size (int): Maximum number of images per batch

    If a setting is not provided, the default value remains unchanged.
    """
    global DEFAULT_PROMPT, RANDOM_SEED, INFERENCE_STEPS, DEFAULT_NOISE_STRENGTH, CONDITIONING_SCALE
    global BATCH_INTERVAL, MAX_BATCH_SIZE
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
        if settings.batch_interval is not None:
            BATCH_INTERVAL = settings.batch_interval
            print(f"Updated batch interval to: {BATCH_INTERVAL} seconds")
        if settings.max_batch_size is not None:
            if settings.max_batch_size > 10:
                raise HTTPException(status_code=400, detail="Max Batch Size cannot exceed 10.")
            MAX_BATCH_SIZE = settings.max_batch_size
            print(f"Updated max batch size to: {MAX_BATCH_SIZE}")
    return {"status": "Settings updated successfully."}

# #########################
# 9. Batching Implementation
# #########################

# Define a data structure for batch requests
class ImageRequest:
    def __init__(self, websocket: WebSocket, data: bytes, request_id: str):
        self.websocket = websocket
        self.data = data
        self.request_id = request_id  # Unique identifier for mapping responses

# Create a shared queue for batching
batch_queue = asyncio.Queue()

# #########################
# 10. Background Batch Processor
# #########################

async def batch_processor():
    while True:
        batch: List[ImageRequest] = []
        try:
            # Wait for the first request with a timeout
            first_request = await asyncio.wait_for(batch_queue.get(), timeout=BATCH_INTERVAL)
            batch.append(first_request)
            print(f"Batch processor started with 1 request. Collecting up to {MAX_BATCH_SIZE} requests.")

            # Collect additional requests until timeout or max batch size
            while len(batch) < MAX_BATCH_SIZE:
                try:
                    # Adjust the timeout based on the remaining time in BATCH_INTERVAL
                    remaining_time = BATCH_INTERVAL
                    request = await asyncio.wait_for(batch_queue.get(), timeout=remaining_time)
                    batch.append(request)
                    print(f"Added request to batch. Current batch size: {len(batch)}")
                except asyncio.TimeoutError:
                    print("Batch interval elapsed. Processing current batch.")
                    break

            # Process the batch
            if batch:
                print(f"Processing batch of size: {len(batch)}")
                await process_batch(batch)
        except asyncio.TimeoutError:
            # No requests received within the interval
            pass
        except Exception as e:
            print(f"Error in batch_processor: {e}")

async def process_batch(batch: List[ImageRequest]):
    # Prepare lists for batch processing
    source_images = []
    control_images = []
    generators = []
    prompts = []
    strengths = []
    conditioning_scales = []
    request_ids = []
    websockets = []

    # Extract data from the batch
    for request in batch:
        pil_source_image = convert_numpy_image_to_pil_image(cv.imdecode(np.frombuffer(request.data, np.uint8), cv.IMREAD_COLOR))
        if pil_source_image is None:
            print(f"Failed to convert image for request {request.request_id}. Skipping.")
            continue

        control_image = get_depth_map(pil_source_image)
        if control_image is None:
            print(f"Failed to generate depth map for request {request.request_id}. Skipping.")
            continue

        # Prepare seed and settings
        with settings_lock:
            seed = RANDOM_SEED if RANDOM_SEED != -1 else random.randint(0, 2147483647)
            prompt = DEFAULT_PROMPT
            inference_steps = INFERENCE_STEPS
            noise_strength = DEFAULT_NOISE_STRENGTH
            conditioning_scale = CONDITIONING_SCALE

        generator = prepare_seed(seed)

        # Append to batch lists
        source_images.append(pil_source_image)
        control_images.append(control_image)
        generators.append(generator)
        prompts.append(prompt)
        strengths.append(noise_strength)
        conditioning_scales.append(conditioning_scale)
        request_ids.append(request.request_id)
        websockets.append(request.websocket)

    if not source_images:
        print("No valid images to process in this batch.")
        return

    # Run the pipeline in batch
    gen_images = run_model(
        pipeline,
        source_images,
        control_images,
        generators,
        prompts,
        INFERENCE_STEPS,
        strengths,
        conditioning_scales
    )

    # Iterate over generated images and send back to respective clients
    for idx, gen_image in enumerate(gen_images):
        if gen_image is None:
            print(f"Generated image is None for request {request_ids[idx]}. Skipping sending.")
            continue

        # Convert the generated image back to a numpy array and encode it as JPEG
        try:
            result_image = np.array(gen_image)
            _, buffer = cv.imencode('.jpg', cv.cvtColor(result_image, cv.COLOR_RGB2BGR))
            result_bytes = buffer.tobytes()
            print(f"Sending processed image for request {request_ids[idx]} back to client.")
        except Exception as e:
            print(f"Error encoding generated image for request {request_ids[idx]}: {e}")
            continue

        # Send the processed image back to the client
        try:
            if websockets[idx].application_state == WebSocketState.CONNECTED:
                await websockets[idx].send_bytes(result_bytes)
                print(f"Processed image sent for request {request_ids[idx]}.")
            else:
                print(f"WebSocket for request {request_ids[idx]} is disconnected. Cannot send data.")
        except Exception as e:
            print(f"Failed to send processed image for request {request_ids[idx]}: {e}")

# #########################
# 11. WebSocket Endpoint
# #########################

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket connection accepted.")
    try:
        while True:
            # Receive the frame as bytes from the client
            data = await websocket.receive_bytes()
            print("Received data from client.")

            # Generate a unique request ID
            request_id = str(uuid.uuid4())

            # Create an ImageRequest and enqueue it
            image_request = ImageRequest(websocket=websocket, data=data, request_id=request_id)
            await batch_queue.put(image_request)
            print(f"Enqueued request {request_id} for batching.")
    except WebSocketDisconnect:
        print("Client disconnected.")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        await websocket.close()
        print("WebSocket closed.")

# #########################
# 12. Main
# #########################

# This section is optional if you're running the app using an external command like uvicorn.
# However, including it allows the script to be executed directly.
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8080)
