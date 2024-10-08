# app.py
import cv2 as cv
import numpy as np
from PIL import Image, ImageEnhance
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
from transformers import AutoImageProcessor, AutoModelForDepthEstimation  # Updated Imports
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
depth_estimator = None  # Updated
image_processor = None   # Updated
run_model = None
last_generated_image = None

DEFAULT_PROMPT = "Aerial view of a futuristic residential area featuring a variety of modern homes, including a large, luxurious mansion and smaller houses. The architecture blends sleek glass, metal, and greenery, with solar panels and green rooftops on each building. Winding pathways and roads connect the homes, surrounded by landscaped gardens and small parks with trees and ponds. The area is designed with sustainability in mind, with electric vehicles on the streets and communal green spaces. The scene is well-lit with natural sunlight, showcasing the clean lines and innovative design of the neighborhood from above"
MODEL = "sdxlturbo"  # "lcm" or "sdxlturbo"
SDXLTURBO_MODEL_LOCATION = 'models/sdxl-turbo'
TORCH_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VARIANT = "fp16"
TORCH_DTYPE = torch.float16
GUIDANCE_SCALE = 0  # Higher guidance scale for better prompt adherence
INFERENCE_STEPS = 2  # 4 for lcm (high quality), 2 for turbo
DEFAULT_NOISE_STRENGTH = 0.5  # 0.5 or 0.7 works well too
CONDITIONING_SCALE = 0.7  # 0.5 or 0.7 works well too
GUIDANCE_START = 0.0
GUIDANCE_END = 1.0
RANDOM_SEED = 21
HEIGHT = 512
WIDTH = 512

# Create threading locks for the pipeline and settings
pipeline_lock = threading.Lock()
settings_lock = threading.Lock()
last_generated_image_lock = threading.Lock()

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

def get_depth_map(image, contrast_factor=1.5):
    """
    Generates a depth map using the Depth Anything model and enhances its contrast.

    Args:
        image (PIL.Image.Image): The input image.
        contrast_factor (float): The factor by which to enhance the contrast.

    Returns:
        PIL.Image.Image: The contrast-enhanced depth map.
    """
    try:
        # Preprocess the image for depth estimation using Depth Anything
        inputs = image_processor(images=image, return_tensors="pt")
        inputs = {k: v.to(TORCH_DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = depth_estimator(**inputs)
            predicted_depth = outputs.predicted_depth

        # Interpolate to original size
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=image.size[::-1],
            mode="bicubic",
            align_corners=False,
        )

        # Normalize the depth map
        prediction = prediction.squeeze().cpu().numpy()
        formatted = (prediction * 255 / np.max(prediction)).astype("uint8")
        depth_image = Image.fromarray(formatted)

        # Enhance Contrast
        enhancer = ImageEnhance.Contrast(depth_image)
        depth_image = enhancer.enhance(contrast_factor)
        print(f"Depth map generated and contrast enhanced by a factor of {contrast_factor}.")
        return depth_image
    except Exception as e:
        print(f"Error generating depth map with Depth Anything: {e}")
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

        pipe.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter_sdxl.bin")
        pipe.set_ip_adapter_scale(0.6)


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
        #pipe = compile(pipe, config)
        print("Pipeline compiled with optimizations.")
    except ImportError:
        print("Compiler module 'sfast' not found. Skipping pipeline compilation.")
    except Exception as e:
        print(f"Error during pipeline compilation: {e}")

    # If not using compiler, assign the pipeline directly
    pipeline = pipe
    return pipeline

def run_sdxlturbo(pipeline, source_image, control_image, generator, prompt, num_inference_steps, strength, conditioning_scale):
    global last_generated_image
    ipimage = None
    with last_generated_image_lock:
        if last_generated_image is not None:
            ipimage = last_generated_image
        else:
            ipimage = Image.new('RGB', (512, 512), 'white')

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
            ip_adapter_image=source_image
        ).images[0]
        print("Pipeline successfully generated image.")
        return gen_image
    except Exception as e:
        print(f"Error during pipeline execution: {e}")
        return None

@app.on_event("startup")
async def startup_event():
    global pipeline, depth_estimator, image_processor, run_model  # Updated

    print("FastAPI application is starting... Loading models into GPU.")
    try:
        # Load depth estimator and processor using Depth Anything
        depth_estimator = AutoModelForDepthEstimation.from_pretrained(
            "depth-anything/Depth-Anything-V2-Small-hf"
        ).to(TORCH_DEVICE)
        image_processor = AutoImageProcessor.from_pretrained(
            "depth-anything/Depth-Anything-V2-Small-hf"
        )
        print("Depth Anything model and processor loaded.")
    except Exception as e:
        print(f"Error loading Depth Anything model or processor: {e}")
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
MAX_CONCURRENT_PROCESSES = 10  # Adjust based on your GPU capacity
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
    """
    Process a single frame and send the result back to the client.
    Implements a timeout of 200 ms for both acquiring the semaphore and processing the frame.
    If either step exceeds 200 ms, the frame is dropped.
    """
    try:
        # Attempt to acquire the semaphore with a 200 ms timeout
        await asyncio.wait_for(processing_semaphore.acquire(), timeout=1)
    except asyncio.TimeoutError:
        print("Processing queue is full. Dropping frame.")
        return

    try:
        try:
            # Attempt to process the frame with a 200 ms timeout
            result = await asyncio.wait_for(
                asyncio.to_thread(process_frame, data),
                timeout=1
            )
        except asyncio.TimeoutError:
            print("Processing frame timed out. Dropping frame.")
            return
        except Exception as e:
            print(f"Exception in processing frame: {e}")
            return

        # If processing was successful, send the frame back to the client
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

        # Generate the depth map as the control image using Depth Anything
        control_image = get_depth_map(pil_source_image)
        if control_image is None:
            print("Depth map generation failed.")
            return None
        else:
            print("Depth map generated successfully using Depth Anything.")

        
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
            
            global last_generated_image
            with last_generated_image_lock:
                last_generated_image = gen_image

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
