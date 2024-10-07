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
import subprocess
import collections
import time
import sys

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

    # Start the FFmpeg streaming thread
    ffmpeg_thread = threading.Thread(target=ffmpeg_streamer, daemon=True)
    ffmpeg_thread.start()
    print("FFmpeg streaming thread started.")

# Set a concurrency limit
MAX_CONCURRENT_PROCESSES = 8  # Adjust based on your GPU capacity
processing_semaphore = asyncio.Semaphore(MAX_CONCURRENT_PROCESSES)

# Initialize the image buffer
image_buffer = collections.deque(maxlen=100)
buffer_lock = threading.Lock()

def ffmpeg_streamer():
    """
    This function runs in a separate thread and handles streaming images from the buffer using FFmpeg.
    """
    ffmpeg_command = [
        'ffmpeg',
        '-y',
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-pix_fmt', 'rgb24',
        '-s', f'{WIDTH}x{HEIGHT}',
        '-r', '30',  # 30 fps
        '-i', '-',  # Input comes from stdin
        '-c:v', 'mpeg1video',
        '-f', 'mpegts',
        '-listen', '1',  # Listen as a server
        'http://0.0.0.0:7000/stream.mpegts'  # Replace with your server's IP
    ]

    try:
        process = subprocess.Popen(ffmpeg_command, stdin=subprocess.PIPE)
        print("FFmpeg process started for streaming on port 7000.")

        while True:
            start_time = time.time()
            with buffer_lock:
                if image_buffer:
                    frame = image_buffer.popleft()
                    print("Popped image from buffer for streaming.")
                else:
                    # Create a black frame if buffer is empty
                    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
                    print("Buffer empty. Using black frame.")

            try:
                # Write frame to FFmpeg's stdin
                process.stdin.write(frame.tobytes())
                print("Frame written to FFmpeg stdin.")
            except BrokenPipeError:
                print("FFmpeg pipe broken. Exiting streamer thread.")
                break
            except Exception as e:
                print(f"Error writing frame to FFmpeg: {e}")
                break

            # Maintain 30 fps
            elapsed = time.time() - start_time
            sleep_time = max(0, (1/30) - elapsed)
            time.sleep(sleep_time)

    except Exception as e:
        print(f"FFmpeg streaming error: {e}")
    finally:
        if process.stdin:
            process.stdin.close()
        process.wait()
        print("FFmpeg process terminated.")


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
            task = asyncio.create_task(process_and_buffer_frame(data))
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

async def process_and_buffer_frame(data):
    """
    Process a single frame and add the result to the buffer.
    Implements a timeout of 200 ms for both acquiring the semaphore and processing the frame.
    If either step exceeds 200 ms, the frame is dropped.
    """
    try:
        # Attempt to acquire the semaphore with a 200 ms timeout
        await asyncio.wait_for(processing_semaphore.acquire(), timeout=0.2)
    except asyncio.TimeoutError:
        print("Processing queue is full. Dropping frame.")
        return

    try:
        try:
            # Attempt to process the frame with a 200 ms timeout
            result = await asyncio.wait_for(
                asyncio.to_thread(process_frame, data),
                timeout=0.2
            )
        except asyncio.TimeoutError:
            print("Processing frame timed out. Dropping frame.")
            return
        except Exception as e:
            print(f"Exception in processing frame: {e}")
            return

        # If processing was successful, add the frame to the buffer
        if result is not None:
            with buffer_lock:
                image_buffer.append(result)
                print("Processed frame added to buffer.")
    finally:
        # Release the semaphore regardless of processing outcome
        processing_semaphore.release()

def process_frame(data):
    """
    Process the received frame and generate the output image.
    This function runs in a separate thread.
    """
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

        # Convert the generated image back to a numpy array
        result_image = np.array(gen_image)
        if result_image is None:
            print("Failed to convert generated image to numpy array.")
            return None

        # Resize the image to match the desired dimensions
        if result_image.shape[0] != HEIGHT or result_image.shape[1] != WIDTH:
            result_image = cv.resize(result_image, (WIDTH, HEIGHT))
            print(f"Resized generated image to {WIDTH}x{HEIGHT}.")

        # Ensure the image is in RGB format
        if result_image.shape[2] == 4:
            result_image = cv.cvtColor(result_image, cv.COLOR_RGBA2RGB)
            print("Converted generated image from RGBA to RGB.")

        return result_image

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

#################
################# FFmpeg Streaming
#################

# Note:
# Ensure that FFmpeg is installed and accessible in your system's PATH.
# If FFmpeg is not installed, you can download it from https://ffmpeg.org/download.html

# The FFmpeg streamer thread is started during the startup event.
# It continuously reads images from the buffer and streams them at 30 fps.
# If the buffer is empty, it sends a black frame instead.

#################
################# RUNNING THE SERVER
#################

# To run the FastAPI server, use the following command:
# uvicorn your_script_name:app --host 0.0.0.0 --port 8000

# Replace 'your_script_name' with the actual name of your Python script.

# The video stream will be available on port 7000.
# You can view the stream using a media player like VLC:
# Open VLC and navigate to "Media" -> "Open Network Stream" and enter the URL:
# http://localhost:7000

# Ensure that the firewall allows traffic on port 7000 if accessing remotely.

#################
################# END OF CODE
#################
