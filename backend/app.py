# app.py

import os
import cv2 as cv
import numpy as np
import torch
import clip
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState
from fastapi.middleware.cors import CORSMiddleware
from roboflow import Roboflow
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

# Import FastSAM and related modules
from fastsam import FastSAM, FastSAMPrompt
import supervision as sv

# Initialize FastAPI
app = FastAPI()

# Configure CORS (adjust origins as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#################
################# CONSTANTS AND GLOBAL VARIABLES
#################

FAST_SAM_CHECKPOINT_PATH = ("weights/FastSAM.pt")
SAM_SAM_CHECKPOINT_PATH = ("sam_vit_h_4b8939.pth")

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"DEVICE = {DEVICE}")

# Initialize FastSAM model
fast_sam = FastSAM(FAST_SAM_CHECKPOINT_PATH)

# Text prompt for segmentation
TEXT_PROMPT = "plastic models"  # Example prompt; adjust as needed

# Concurrency limit
MAX_CONCURRENT_PROCESSES = 1  # Adjust based on your server's capacity
processing_semaphore = asyncio.Semaphore(MAX_CONCURRENT_PROCESSES)

#################
################# HELPER FUNCTIONS
#################

def masks_to_bool(masks):
    """
    Convert masks to boolean format.
    """
    if isinstance(masks, np.ndarray):
        return masks.astype(bool)
    return masks.cpu().numpy().astype(bool)

def annotate_image(image: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """
    Create an annotated mask image with a black background.
    The mask areas will be white.
    """
    # Create a black background
    mask_image = np.zeros_like(image, dtype=np.uint8)

    # Aggregate all masks into a single 2D mask
    if masks.ndim == 3:
        masks = masks.sum(axis=0)
        masks = masks > 0  # Convert to binary

    # Ensure masks are now 2D
    if masks.ndim != 2:
        print(f"Unexpected mask shape after aggregation: {masks.shape}")
        return mask_image  # Return black image if mask shape is unexpected

    # Apply mask: set mask areas to white
    mask_image[masks] = [255, 255, 255]  # White mask

    return mask_image


def generate_mask_image(image: np.ndarray, text_prompt: str) -> bytes:
    try:
        # Original image size
        original_height, original_width = image.shape[:2]
        print(f"Original image size: {original_width}x{original_height}")

        # Convert OpenCV BGR image to RGB
        image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        # Run FastSAM inference with imgsz set to 512
        results = fast_sam(
            source=image_rgb,
            device=DEVICE,
            retina_masks=True,
            imgsz=512,  # Changed to match input image size
            conf=0.5,
            iou=0.6
        )
        print("FastSAM inference completed.")

        # Initialize FastSAMPrompt with results
        prompt_process = FastSAMPrompt(image_rgb, results, device=DEVICE)
        print("FastSAMPrompt initialized.")

        # Generate masks using text prompt
        masks = prompt_process.text_prompt(text=text_prompt)
        print(f"Masks generated with shape: {masks.shape}")

        # Convert masks to boolean
        masks_bool = masks_to_bool(masks)
        print(f"Masks converted to boolean with shape: {masks_bool.shape}")

        # Annotate image to create mask with black background
        mask_image = annotate_image(image, masks_bool)
        print(f"Mask image annotated with shape: {mask_image.shape}")

        # Encode the mask image to JPEG format
        success, buffer = cv.imencode('.jpg', mask_image)
        if not success:
            print("Failed to encode mask image to JPEG.")
            return None

        return buffer.tobytes()

    except Exception as e:
        print(f"Error in generate_mask_image: {e}")
        return None



#################
################# WEBSOCKET ENDPOINT
#################

# Add an asyncio Lock for thread safety
processing_lock = asyncio.Lock()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket connection accepted.")
    tasks = []
    try:
        while True:
            data = await websocket.receive_bytes()
            print("Received data from client.")
            task = asyncio.create_task(process_and_send_frame(data, websocket))
            tasks.append(task)
    except WebSocketDisconnect:
        print("Client disconnected.")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        for task in tasks:
            task.cancel()
        if websocket.application_state != WebSocketState.DISCONNECTED:
            await websocket.close()
        print("WebSocket closed.")

async def process_and_send_frame(data, websocket):
    try:
        await asyncio.wait_for(processing_semaphore.acquire(), timeout=3)
    except asyncio.TimeoutError:
        print("Processing queue is full. Dropping frame.")
        return

    try:
        async with processing_lock:
            try:
                result = await asyncio.wait_for(
                    asyncio.to_thread(process_frame, data),
                    timeout=3
                )
            except asyncio.TimeoutError:
                print("Processing frame timed out. Dropping frame.")
                return
            except Exception as e:
                print(f"Exception in processing frame: {e}")
                return

            if result is not None:
                print("Sending processed mask back to the client.")
                try:
                    if websocket.application_state == WebSocketState.CONNECTED:
                        await websocket.send_bytes(result)
                        print("Processed mask sent to client.")
                    else:
                        print("WebSocket is closed. Cannot send data.")
                except Exception as e:
                    print(f"Failed to send data: {e}")
    finally:
        processing_semaphore.release()

def process_frame(data):
    """
    Process the received frame to generate a mask using FastSAM.
    This function runs in a separate thread.
    """
    try:
        # Decode the received bytes to a numpy array
        nparr = np.frombuffer(data, np.uint8)
        print("Converted received data to numpy array.")

        # Decode the numpy array to an image (OpenCV)
        frame = cv.imdecode(nparr, cv.IMREAD_COLOR)
        if frame is None:
            print("Failed to decode image from received data.")
            return None
        else:
            print("Image successfully decoded.")

        # Generate mask image using FastSAM
        mask_bytes = generate_mask_image(frame, TEXT_PROMPT)
        if mask_bytes is None:
            print("Mask generation failed.")
            return None
        else:
            print("Mask image generated successfully.")

        return mask_bytes

    except Exception as e:
        print(f"Error in process_frame: {e}")
        return None
