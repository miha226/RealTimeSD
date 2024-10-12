# app.py
import cv2 as cv
import numpy as np
from PIL import Image, ImageEnhance
from diffusers import (
    StableDiffusionXLControlNetImg2ImgPipeline,
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    AutoencoderTiny,
    LCMScheduler,
    DPMSolverMultistepScheduler,
    DiffusionPipeline,
    UNet2DConditionModel,
    UniPCMultistepScheduler
)
import torch
from fastapi import FastAPI, File, UploadFile, WebSocket, HTTPException, WebSocketDisconnect, Response
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

DEFAULT_PROMPT = "Architecture, city, sunshine, day, urban landscape, skyscrapers, scenery, white clouds, buildings, bridges, sky, city lights, blue sky, east_ Asia_ Architecture, mountains, rivers, pagodas, outdoor, trees, tokyo_\\ (City )<lora:20_a:0.2>"
DEFAULT_NEGATIVE_PROMPT = "water, lake water.,2 faces, cropped image, out of frame, draft, deformed hands, signatures, twisted fingers, double image, long neck, malformed hands, multiple heads, extra limb, poorly drawn hands, missing limb, disfigured, cut-off, low-res, deformed, blurry, bad anatomy, mutation, mutated, floating limbs, disconnected limbs, long body, disgusting, poorly drawn, mutilated, mangled, extra fingers, duplicate artifacts, morbid, gross proportions, missing arms, mutated hands, mutilated hands, malformed, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, extra limbs, disfigured, deformed, body out of frame, bad anatomy, watermark, signature, cut off, low contrast, underexposed, overexposed, bad art, beginner, amateur, distorted face, blurry, draft, grainy."
MODEL = "sdxlturbo"  # "lcm" or "sdxlturbo"
SDXLTURBO_MODEL_LOCATION = 'models/sdxl-turbo'
TORCH_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VARIANT = "fp16"
TORCH_DTYPE = torch.float16
GUIDANCE_SCALE = 2  # Higher guidance scale for better prompt adherence
INFERENCE_STEPS = 6  # 4 for lcm (high quality), 2 for turbo
DEFAULT_NOISE_STRENGTH = 0.0  # 0.5 or 0.7 works well too
CONDITIONING_SCALE = 0.8  # 0.5 or 0.7 works well too
GUIDANCE_START = 0.0
GUIDANCE_END = 1.0
RANDOM_SEED = 21
HEIGHT = 768
WIDTH = 768

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
            "lllyasviel/control_v11f1p_sd15_depth",
            use_safetensors=True,
            torch_dtype=torch.float16,
        )
        print("Depth ControlNet model loaded.")
    except Exception as e:
        print(f"Error loading ControlNet model: {e}")
        return None
   

    try:
        pipe = StableDiffusionControlNetPipeline.from_single_file(
            "models/epiCrealism/epiCRealism.safetensors",
            controlnet=controlnet,
            variant=VARIANT,
            use_safetensors=True,
            torch_dtype=TORCH_DTYPE
        ).to(TORCH_DEVICE)
        
        print("Pipeline loaded.")
        print("Lora weights loaded.")
        pipe.load_lora_weights("models/loras/LCM_LoRA_Weights_SD15.safetensors", use_safetensors=True)
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
 

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
    
    guidance = 6 * strength + 2
    latent = torch.zeros((1,4,96,96),dtype=torch.float16, device="cuda")
    try:
        gen_image = pipeline(
            prompt=prompt,
            negative_prompt=DEFAULT_NEGATIVE_PROMPT,
            num_inference_steps=num_inference_steps,
            guidance_scale=2,  # Keeping it as default
            width=WIDTH,
            height=HEIGHT,
            latents= latent,
            image = control_image,
            generator=generator,                # Control image (depth map)
            controlnet_conditioning_scale=float(conditioning_scale)
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
MAX_CONCURRENT_PROCESSES = 4  # Adjust based on your GPU capacity
processing_semaphore = asyncio.Semaphore(MAX_CONCURRENT_PROCESSES)

@app.post("/process")
async def process_image(file: UploadFile = File(...)):
    """
    Endpoint to process an uploaded image and return the processed image.
    """
    if file.content_type.split('/')[0] != 'image':
        raise HTTPException(status_code=400, detail="Invalid image file")

    try:
        # Read the image bytes
        data = await file.read()
        print(f"Received image: {file.filename} ({len(data)} bytes)")

        # Acquire the semaphore to respect concurrency limits
        try:
            await asyncio.wait_for(processing_semaphore.acquire(), timeout=10.0)
        except asyncio.TimeoutError:
            raise HTTPException(status_code=503, detail="Server is busy. Please try again later.")

        # Process the image in a separate thread to avoid blocking
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(process_frame, data),
                timeout=60.0  # Adjust timeout as needed
            )
        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail="Image processing timed out.")
        except Exception as e:
            print(f"Exception during image processing: {e}")
            raise HTTPException(status_code=500, detail="Internal server error during image processing.")
        finally:
            # Release the semaphore
            processing_semaphore.release()

        if result is None:
            raise HTTPException(status_code=500, detail="Failed to process the image.")

        # Return the processed image as a response with appropriate headers
        return Response(content=result, media_type="image/jpeg")

    except Exception as e:
        print(f"Error in /process endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal server error.")

"""
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
   
    Process a single frame and send the result back to the client.
    Implements a timeout of 200 ms for both acquiring the semaphore and processing the frame.
    If either step exceeds 200 ms, the frame is dropped.
 
    try:
        # Attempt to acquire the semaphore with a 200 ms timeout
        await asyncio.wait_for(processing_semaphore.acquire(), timeout=0.1)
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
"""

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
