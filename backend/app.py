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
from transformers import DPTImageProcessor, DPTForDepthEstimation
from DeepCache import DeepCacheSDHelper
from pydantic import BaseModel
from typing import Optional
from sfast.compilers.diffusion_pipeline_compiler import (compile, CompilationConfig)
from transformers import AutoImageProcessor, AutoModelForDepthEstimation


from diffusers import FluxControlNetModel
from diffusers.pipelines import FluxControlNetPipeline, FluxImg2ImgPipeline

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
depth_image_processor = None


DEFAULT_PROMPT = "photo of city glass skyscrapers, 4K, realistic, smooth transition"
MODEL_LOCATION = "models/FLUX.1-schnell"
TORCH_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VARIANT = "fp16"
TORCH_DTYPE = torch.float16
GUIDANCE_SCALE = 2  # Higher guidance scale for better prompt adherence
INFERENCE_STEPS = 2  # 4 for lcm (high quality), 2 for turbo
DEFAULT_NOISE_STRENGTH = 0.5  # 0.5 or 0.7 works well too
CONDITIONING_SCALE = 0.5  # 0.5 or 0.7 works well too
GUIDANCE_START = 0.0
GUIDANCE_END = 1.0
RANDOM_SEED = 21
HEIGHT = 512
WIDTH = 512




# Define the settings Pydantic model
class Settings(BaseModel):
    prompt: Optional[str] = None
    seed: Optional[int] = None
    inference_steps: Optional[int] = None
    noise_strength: Optional[float] = None
    conditioning_scale: Optional[float] = None

class PipeSettings(BaseModel):
    load_depth_controlnet: Optional[bool] = None
    load_depth_estimator: Optional[bool] = None
    load_normals_controlnet: Optional[bool] = None





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
    
def load_depth_processor():
    global depth_image_processor, depth_estimator
    try:
        # Load depth estimator and processor using Depth Anything
        depth_estimator = AutoModelForDepthEstimation.from_pretrained(
            "depth-anything/Depth-Anything-V2-Small-hf"
        ).to(TORCH_DEVICE)
        depth_image_processor = AutoImageProcessor.from_pretrained(
            "depth-anything/Depth-Anything-V2-Small-hf"
        )
        print("Depth Anything model and processor loaded.")
    except Exception as e:
        print(f"Error loading Depth Anything model or processor: {e}")
        raise e
    
def get_depth_map(image, contrast_factor=1.5):
    """
    Generates a depth map using the Depth Anything model and enhances its contrast.

    Args:
        image (PIL.Image.Image): The input image.
        contrast_factor (float): The factor by which to enhance the contrast.

    Returns:
        PIL.Image.Image: The contrast-enhanced depth map in RGB format.
    """
    try:
        # Preprocess the image for depth estimation using Depth Anything
        inputs = depth_image_processor(images=image, return_tensors="pt")
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

        # **Convert to RGB**
        depth_image = depth_image.convert("RGB")
        print("Depth map converted to RGB format.")

        return depth_image
    except Exception as e:
        print(f"Error generating depth map with Depth Anything: {e}")
        return None


def load_pipeline(optimize_pipeline=True, load_depth_controlnet=False, load_canny_controlnet=False, load_normals_controlnet=False):
  
    controlnets = []
    
    try:
        load_depth_processor()
        # Load the depth ControlNet model
        depth_contorlnet = FluxControlNetModel.from_pretrained(
        "jasperai/Flux.1-dev-Controlnet-Depth",
            torch_dtype=torch.bfloat16
        )
        print("Depth ControlNet model loaded.")
        controlnets.append(depth_contorlnet)
    except Exception as e:
        print(f"Error loading ControlNet model: {e}")
        return None
        
    if(load_canny_controlnet):
        try:
            # Load the depth ControlNet model
            canny_contorlnet = FluxControlNetModel.from_pretrained(
            "InstantX/FLUX.1-dev-Controlnet-Canny",
                torch_dtype=torch.bfloat16
            )
            print("Canny ControlNet model loaded.")
            controlnets.append(canny_contorlnet)
        except Exception as e:
            print(f"Error loading ControlNet model: {e}")
            return None
        
    if(load_normals_controlnet):
        try:
            # Load the depth ControlNet model
            depth_contorlnet = FluxControlNetModel.from_pretrained(
            "jasperai/Flux.1-dev-Controlnet-Surface-Normals",
                torch_dtype=torch.bfloat16
            )
            print("Normals ControlNet model loaded.")
            controlnets.append(depth_contorlnet)
        except Exception as e:
            print(f"Error loading ControlNet model: {e}")
            return None

    # Load the VAE
    """
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
    """
    try:
        if controlnets:
            pipe = FluxControlNetPipeline.from_pretrained(
                MODEL_LOCATION,
                controlnet=depth_contorlnet,
                torch_dtype=torch.bfloat16,
                use_safetensors=True
            ).to(TORCH_DEVICE)
            print("Pipeline with ControlNets loaded and moved to device.")
        else:
            pipe = FluxImg2ImgPipeline.from_pretrained(
                MODEL_LOCATION,
                torch_dtype=torch.bfloat16,
                use_safetensors=True
            ).to(TORCH_DEVICE)
            print("Pipeline without ControlNets loaded and moved to device.")
    except Exception as e:
        print(f"Error loading pipeline: {e}")
        return None
    
     #helper = DeepCacheSDHelper(pipe=pipe)
        #helper.set_params(cache_interval=3, cache_branch_id=0)
        #helper.enable()

    if(optimize_pipeline):
        pipe = apply_optimizations()

    return pipe


def apply_optimizations():
    # #########################
    # Enabling xformers and Triton (Optional)
    # #########################
    global pipeline
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
    # Compiling the Pipeline (Optional)
    # #########################

    try:
        pipeline = compile(pipeline, config)
        print("Pipeline compiled with optimizations.")
    except ImportError:
        print("Compiler module 'sfast' not found. Skipping pipeline compilation.")
    except Exception as e:
        print(f"Error during pipeline compilation: {e}")

    


@app.on_event("startup")
async def startup_event():
    global pipeline

    print("FastAPI application is starting... Loading models into GPU.")
    

    try:
        pipeline = load_pipeline(load_depth_controlnet=True)
        if pipeline is None:
            print("Failed to load pipeline.")
            raise RuntimeError("Pipeline loading failed.")
        else:
            print("Pipeline loaded successfully.")
    except Exception as e:
        print(f"Exception during pipeline loading: {e}")
        raise e
    print("Models loaded successfully.")


MAX_CONCURRENT_TASKS = 10
semaphore = asyncio.Semaphore(MAX_CONCURRENT_TASKS)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket connection accepted.")
    try:
        while True:
            data = await websocket.receive_bytes()
            print("Received data from client.")
            await semaphore.acquire()
            asyncio.create_task(
                process_and_send_frame(data, websocket)
            ).add_done_callback(lambda t: semaphore.release())
    except WebSocketDisconnect:
        print("Client disconnected.")
    except Exception as e:
        print(f"Error: {e}")
    finally:
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
        # Attempt to process the frame with a 200 ms timeout
        result = process_frame(data)
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
  


def process_frame(data):
    """
    Process the received frame and generate the output image.
    This function runs in a separate thread.
    """
    try:
        # Read current settings
        prompt = DEFAULT_PROMPT
        seed = RANDOM_SEED
        inference_steps = INFERENCE_STEPS
        noise_strength = DEFAULT_NOISE_STRENGTH
        conditioning_scale = CONDITIONING_SCALE
        guidance_scale = GUIDANCE_SCALE

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
        pipeline_output = pipeline(
            control_image=control_image,
            generator=generator,
            prompt=prompt,
            num_inference_steps=inference_steps,
            controlnet_conditioning_scale=noise_strength,
            # strength=noise_strength,
            # conditioning_scale=conditioning_scale,
            width=WIDTH,
            height=HEIGHT,
            guidance_scale=guidance_scale,
        )

        # **Extract the generated image from the pipeline output**
        if hasattr(pipeline_output, 'images') and len(pipeline_output.images) > 0:
            gen_image = pipeline_output.images[0]
            print("Image successfully generated by pipeline.")
        else:
            print("Pipeline output does not contain any images.")
            return None

        # **Ensure the generated image is in RGB format**
        if gen_image.mode != "RGB":
            gen_image = gen_image.convert("RGB")
            print("Generated image converted to RGB format.")

        # **Convert the generated PIL Image to a NumPy array**
        result_image = np.array(gen_image)

        # **Ensure that the NumPy array has the correct shape and dtype**
        if result_image.ndim != 3 or result_image.shape[2] != 3:
            print(f"Unexpected image shape: {result_image.shape}")
            return None

        # **Convert from RGB to BGR for OpenCV**
        result_image_bgr = cv.cvtColor(result_image, cv.COLOR_RGB2BGR)
        print("Converted generated image from RGB to BGR format.")

        # **Encode the image as JPEG**
        success, buffer = cv.imencode('.jpg', result_image_bgr)
        if not success:
            print("Failed to encode generated image to JPEG format.")
            return None
        print("Generated image encoded to JPEG format.")

        return buffer.tobytes()

    except Exception as e:
        print(f"Error in process_frame: {e}")
        return None






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


@app.on_event("shutdown")
async def shutdown_event():
    global pipeline
    if pipeline:
        del pipeline
        torch.cuda.empty_cache()
        print("Pipeline and GPU resources have been released.")