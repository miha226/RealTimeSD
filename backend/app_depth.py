import cv2 as cv
import numpy as np
from PIL import Image
from diffusers import (
    StableDiffusionXLControlNetImg2ImgPipeline,
    ControlNetModel,
    AutoencoderTiny
)
import torch
from fastapi import FastAPI, WebSocket, HTTPException, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import threading
from starlette.websockets import WebSocketState
from transformers import DPTImageProcessor, DPTForDepthEstimation
from DeepCache import DeepCacheSDHelper
from pydantic import BaseModel
from typing import Optional
from sfast.compilers.diffusion_pipeline_compiler import (compile, CompilationConfig)

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
MODEL = "sdxlturbo"  # "lcm" or "sdxlturbo"
SDXLTURBO_MODEL_LOCATION = 'models/sdxl-turbo'
TORCH_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VARIANT = "fp16"
TORCH_DTYPE = torch.float16
GUIDANCE_SCALE = 0  # Higher guidance scale for better prompt adherence
INFERENCE_STEPS = 2  # 4 for lcm (high quality), 2 for turbo
DEFAULT_NOISE_STRENGTH = 0.5  # 0.5 or 0.7 works well too
CONDITIONING_SCALE = 0.5  # 0.5 or 0.7 works well too
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

def run_sdxlturbo(pipeline, source_image, control_image, generator, prompt, num_inference_steps, strength, conditioning_scale):
    try:
        gen_image = pipeline(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=GUIDANCE_SCALE,  # Keeping it as default
            width=WIDTH,
            height=HEIGHT,
            generator=generator,
            image=source_image,                # Source image
            control_image=control_image,        # Control image (depth map)
            strength=strength,
            controlnet_conditioning_scale=conditioning_scale,
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
