import cv2 as cv
import numpy as np
from PIL import Image, ImageEnhance
from diffusers import AutoencoderTiny
import torch
from fastapi import FastAPI, WebSocket, HTTPException, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import threading
from starlette.websockets import WebSocketState
from controlnet_aux import NormalBaeDetector
from DeepCache import DeepCacheSDHelper
from pydantic import BaseModel
from typing import Optional, Union, List, Callable, Dict, Any, Tuple
from sfast.compilers.diffusion_pipeline_compiler import (compile, CompilationConfig)
from pipeline_controlnet_union_sd_xl_img2img import StableDiffusionXLControlNetUnionImg2ImgPipeline
from controlnet_union import ControlNetModel_Union
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from diffusers import StableDiffusionControlNetImg2ImgPipeline, ControlNetModel
from diffusers import (
    StableDiffusionControlNetPipeline,
    UniPCMultistepScheduler,
)

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
normal_detector = None
run_model = None
depth_estimator = None  # Updated
image_processor = None   # Updated


DEFAULT_PROMPT = "photo of city glass skyscrapers, 4K, realistic, smooth transition"
MODEL = "sdxlturbo"  # "lcm" or "sdxlturbo"
SDXLTURBO_MODEL_LOCATION = 'models/sd-turbo'
TORCH_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VARIANT = "fp16"
TORCH_DTYPE = torch.float16
GUIDANCE_SCALE = 0  # Adjusted to a typical value for guidance
INFERENCE_STEPS = 2  # Adjusted based on your example
DEFAULT_NOISE_STRENGTH = 0.5  # Adjusted to match the pipeline's strength parameter
CONDITIONING_SCALE = 0.7  # Adjusted to match the pipeline's controlnet_conditioning_scale
GUIDANCE_START = 0.0
GUIDANCE_END = 1.0
RANDOM_SEED = 21
HEIGHT = 512  # Adjusted based on your example
WIDTH = 512   # Adjusted based on your example

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

def get_normal_map(image):
    try:
        # Generate the normal map as a NumPy array using OpenCV
        normal = normal_detector(image, hand_and_face=False, output_type='cv2')
        
        # need to resize the image resolution to 1024 * 1024 or same bucket resolution to get the best performance
        height, width, _  = normal.shape
        print("Height: {}, Width: {}".format(height, width))
        ratio = np.sqrt(512. * 512. / (width * height))
        print("Ratio: {}".format(ratio))
        new_width, new_height = int(width * ratio), int(height * ratio)
        print("New Width: {}, New Height: {}".format(new_width, new_height))
        normal = cv.resize(normal, (new_width, new_height))
        normal = Image.fromarray(normal)
        print("normaly type: {}".format(type(normal)))
        print("Normal map generated, resized, and padded successfully.")
        
        return normal
    except Exception as e:
        print(f"Error generating normal map: {e}")
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
        #enhancer = ImageEnhance.Contrast(depth_image)
        #depth_image = enhancer.enhance(contrast_factor)
        #print(f"Depth map generated and contrast enhanced by a factor of {contrast_factor}.")
        return depth_image
    except Exception as e:
        print(f"Error generating depth map with Depth Anything: {e}")
        return None
    

def depth_to_normal_map(depth_image, bg_threshold=3):
    # Convert the depth map to a NumPy array
    depth_np = np.array(depth_image).astype(np.float32)
    
    # Normalize depth map to range [0, 1]
    depth_np -= np.min(depth_np)
    depth_np /= np.max(depth_np)
    
    # Calculate gradients using Sobel operator
    sobel_x = cv.Sobel(depth_np, cv.CV_32F, 1, 0, ksize=3)
    sobel_y = cv.Sobel(depth_np, cv.CV_32F, 0, 1, ksize=3)
    
    # Set a constant for z-axis
    sobel_z = np.ones_like(sobel_x) * np.pi * 2.0

    # Stack and normalize gradients to create normal map
    normal_map = np.stack([sobel_x, sobel_y, sobel_z], axis=2)
    normal_map /= np.sqrt(np.sum(normal_map ** 2, axis=2, keepdims=True) + 1e-8)
    
    # Convert normals to 0-255 range
    normal_map = ((normal_map + 1.0) / 2.0 * 255.0).astype(np.uint8)
    
    # Create PIL image from NumPy array
    normal_map_image = Image.fromarray(normal_map)
    
    return normal_map_image

def prepare_sdxlturbo_pipeline():
    global pipeline
    try:
        # Load the normal ControlNet model
        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/control_v11p_sd15_normalbae",
            use_safetensors=True,
            torch_dtype=torch.float16,
        ).to(TORCH_DEVICE)
        print("Normal ControlNet model loaded.")
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
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
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

    #########################
    # 5. Enabling xformers and Triton (Optional)
    #########################

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

    #########################
    # 6. Compiling the Pipeline (Optional)
    #########################

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

def run_sdxlturbo(
    pipeline,
    source_image,
    control_image,
    generator,
    prompt,
    num_inference_steps,
    strength,
    conditioning_scale
):
    try:
        # Define the union_control_type tensor
        # Assuming index 4 corresponds to normal maps based on your example
        union_control_type = torch.Tensor([0, 0, 0, 0, 1, 0]).to(TORCH_DEVICE)

        # Prepare the control_image_list
        # The list length should match the number of control types supported by the pipeline
        control_image_list = [None, None, None, None, control_image, None]
        print("Stage 1")
        gen_image = pipeline(
            prompt=prompt,
            image=control_image,  # Source image
            #control_image_list=control_image_list,  # Control images
            #union_control=True,
            #union_control_type=union_control_type,
            num_inference_steps=num_inference_steps,
            guidance_scale=GUIDANCE_SCALE,
            #strength=strength,
            controlnet_conditioning_scale=conditioning_scale,
            height=HEIGHT,
            width=WIDTH,
            generator=generator
        ).images[0]
        print("Pipeline successfully generated image.")
        print("Stage 2")
        return gen_image
    except Exception as e:
        print(f"Error during pipeline execution: {e}")
        return None

@app.on_event("startup")
async def startup_event():
    global pipeline, normal_detector, run_model, depth_estimator, image_processor

    print("FastAPI application is starting... Loading models into GPU.")
    try:
        # Load normal detector
        normal_detector = NormalBaeDetector.from_pretrained('lllyasviel/Annotators')
        print("Normal detector loaded.")
    except Exception as e:
        print(f"Error loading normal detector: {e}")
        raise e
    
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
MAX_CONCURRENT_PROCESSES = 1  # Adjust based on your GPU capacity
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

        # Generate the normal map as the control image
        #control_image = depth_to_normal_map(get_depth_map(pil_source_image))
        control_image=get_normal_map(pil_source_image)
        if control_image is None:
            print("Normal map generation failed.")
            return None
        else:
            print("Normal map generated successfully.")


        result_image = np.array(control_image)
        _, buffer = cv.imencode('.jpg', cv.cvtColor(result_image, cv.COLOR_RGB2BGR))
        print("Generated image encoded to JPEG format.")

        return buffer.tobytes()
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
