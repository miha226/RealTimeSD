# app.py

import os
import cv2 as cv
import numpy as np
import torch
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState
from fastapi.middleware.cors import CORSMiddleware

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

FAST_SAM_CHECKPOINT_PATH = "weights/FastSAM.pt"
SAM_SAM_CHECKPOINT_PATH = "sam_vit_h_4b8939.pth"

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"DEVICE = {DEVICE}")

# Initialize FastSAM model
fast_sam = FastSAM(FAST_SAM_CHECKPOINT_PATH)

# Text prompt for segmentation (no longer needed, but kept for reference)
TEXT_PROMPT = "plastic models"  # Can be removed if not used elsewhere

# Concurrency limit
MAX_CONCURRENT_PROCESSES = 2  # Adjust based on your server's capacity
FRAME_QUEUE_MAX_SIZE = 10      # Maximum number of frames in the queue

# Initialize semaphore and lock
processing_semaphore = asyncio.Semaphore(MAX_CONCURRENT_PROCESSES)
processing_lock = asyncio.Lock()

# Initialize frame queue
frame_queue = asyncio.Queue(maxsize=FRAME_QUEUE_MAX_SIZE)

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

# Optional: Using Supervision's MaskAnnotator for enhanced annotation
def annotate_image_with_supervision(image: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """
    Create an annotated mask image using Supervision's MaskAnnotator.
    """
    try:
        # Convert masks to XYXY format if needed
        xyxy = sv.mask_to_xyxy(masks=masks)
        detections = sv.Detections(xyxy=xyxy, mask=masks)

        # Initialize the MaskAnnotator
        mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)
        
        # Annotate the image
        annotated_image = mask_annotator.annotate(scene=image.copy(), detections=detections)
        
        return annotated_image

    except Exception as e:
        print(f"Error in annotate_image_with_supervision: {e}")
        # Return the original image or a black mask in case of failure
        return image

def generate_mask_image(image: np.ndarray) -> bytes:
    """
    Generate a mask image using FastSAM's everything prompt.
    Returns the mask image encoded in JPEG format.
    """
    try:
        # Original image size
        original_height, original_width = image.shape[:2]
        print(f"Original image size: {original_width}x{original_height}")

        # Convert OpenCV BGR image to RGB
        image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        # Run FastSAM inference with imgsz set to 512 (matches client image size)
        results = fast_sam(
            source=image_rgb,
            device=DEVICE,
            retina_masks=True,
            imgsz=512,  # Ensures mask size matches input image size
            conf=0.6,    # Increased from 0.5 for higher precision
            iou=0.8      # Increased from 0.6 for stricter mask overlap
        )
        print("FastSAM inference completed.")

        # Initialize FastSAMPrompt with results
        prompt_process = FastSAMPrompt(image_rgb, results, device=DEVICE)
        print("FastSAMPrompt initialized.")

        # Generate masks using everything prompt
        masks = prompt_process.everything_prompt()
        print(f"Masks generated with shape: {masks.shape}")

        # Convert masks to boolean
        masks_bool = masks_to_bool(masks)
        print(f"Masks converted to boolean with shape: {masks_bool.shape}")

        # Annotate image to create mask with black background
        mask_image = annotate_image(image, masks_bool)
        print(f"Mask image annotated with shape: {mask_image.shape}")

        # Alternatively, use supervision's MaskAnnotator for better visualization
        # mask_image = annotate_image_with_supervision(image, masks_bool)
        # print(f"Annotated image with supervision has shape: {mask_image.shape}")

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
                # Process the frame in a separate thread with a timeout
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
        mask_bytes = generate_mask_image(frame)
        if mask_bytes is None:
            print("Mask generation failed.")
            return None
        else:
            print("Mask image generated successfully.")

        return mask_bytes

    except Exception as e:
        print(f"Error in process_frame: {e}")
        return None

# Worker tasks to process frames from the queue
async def frame_worker():
    while True:
        data, websocket = await frame_queue.get()
        await process_and_send_frame(data, websocket)
        frame_queue.task_done()

# Start worker tasks
for _ in range(MAX_CONCURRENT_PROCESSES):
    asyncio.create_task(frame_worker())
