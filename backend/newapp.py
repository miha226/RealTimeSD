###
### PRE-ORDER BOOK on DIY AI! https://a.co/d/eDxJXJ0
###

import cv2 as cv
import numpy as np
from PIL import Image, ImageFilter
from diffusers import (AutoPipelineForImage2Image, StableDiffusionControlNetPipeline,
                       ControlNetModel)
import torch
import os
from fastapi import FastAPI, WebSocket
from fastapi.websockets import WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware


app  = FastAPI()


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

# DEFAULT_PROMPT                = "portrait of adult pikachu monster, in the style of pixar movie, pikachu face, pokemon" #van gogh in the style of van gogh"
# DEFAULT_PROMPT                = "pikachu, pokemon, wizard hat, style of pixar movie, Disney, 8k" #van gogh in the style of van gogh"
DEFAULT_PROMPT                = "portrait of a minion, wearing goggles, yellow skin, wearing a beanie, despicable me movie, in the style of pixar movie" #van gogh in the style of van gogh"
# DEFAULT_PROMPT                = "portrait of a indiana jones, harrison ford film"
# DEFAULT_PROMPT                =  "van gogh in the style of van gogh"
# DEFAULT_PROMPT                =  "beautiful and cute angry crying success kid wearing beanie"

MODEL                         = "sdxlturbo" #"lcm" # or "sdxlturbo"
SDXLTURBO_MODEL_LOCATION      = 'models/sdxl-turbo'
LCM_MODEL_LOCATION            = 'models/LCM_Dreamshaper_v7'
CONTROLNET_CANNY_LOCATION     = "models/control_v11p_sd15_canny" 
TORCH_DEVICE                  = "cuda"
TORCH_DTYPE                   = torch.float16
GUIDANCE_SCALE                = 3 # 0 for sdxl turbo (hardcoded already)
INFERENCE_STEPS               = 4 #4 for lcm (high quality) #2 for turbo
DEFAULT_NOISE_STRENGTH        = 0.7 # 0.5 works well too
CONDITIONING_SCALE            = .7 # .5 works well too
GUIDANCE_START                = 0.
GUIDANCE_END                  = 1.
RANDOM_SEED                   = 21
HEIGHT                        = 384 #512 #384 #512
WIDTH                         = 384 #512 #384 #512

def prepare_seed():
    generator = torch.manual_seed(RANDOM_SEED)
    return generator

def convert_numpy_image_to_pil_image(image):
    return Image.fromarray(cv.cvtColor(image, cv.COLOR_BGR2RGB))

def get_result_and_mask(frame, center_x, center_y, width, height):
    "just gets full frame and the mask for cutout"
    
    mask = np.zeros_like(frame)
    mask[center_y:center_y+height, center_x:center_x+width, :] = 255
    cutout = frame[center_y:center_y+height, center_x:center_x+width, :]

    return frame, cutout

def process_lcm(image, lower_threshold = 100, upper_threshold = 100, aperture=3): 
    image = np.array(image)
    image = cv.Canny(image, lower_threshold, upper_threshold,apertureSize=aperture)
    image = np.repeat(image[:, :, np.newaxis], 3, axis=2)
    return image

def process_sdxlturbo(image):
    return image

def prepare_lcm_controlnet_or_sdxlturbo_pipeline():

    if MODEL=="lcm":

        controlnet = ControlNetModel.from_pretrained(CONTROLNET_CANNY_LOCATION, torch_dtype=TORCH_DTYPE,
                                                use_safetensors=True)
    
        pipeline = StableDiffusionControlNetPipeline.from_pretrained(LCM_MODEL_LOCATION,\
                                                        controlnet=controlnet, 
                                                        # unet=unet,\
                                                        torch_dtype=TORCH_DTYPE, safety_checker=None).\
                                                    to(TORCH_DEVICE)

    elif MODEL=="sdxlturbo":

        pipeline = AutoPipelineForImage2Image.from_pretrained(
                    SDXLTURBO_MODEL_LOCATION, torch_dtype=TORCH_DTYPE,
                    safety_checker=None).to(TORCH_DEVICE)
        
    return pipeline

def run_lcm(pipeline, ref_image):

    generator = prepare_seed()
    gen_image = pipeline(prompt                        = DEFAULT_PROMPT,
                         num_inference_steps           = INFERENCE_STEPS, 
                         guidance_scale                = GUIDANCE_SCALE,
                         width                         = WIDTH, 
                         height                        = HEIGHT, 
                         generator                     = generator,
                         image                         = ref_image, 
                         controlnet_conditioning_scale = CONDITIONING_SCALE, 
                         control_guidance_start        = GUIDANCE_START, 
                         control_guidance_end          = GUIDANCE_END, 
                        ).images[0]

    return gen_image

def run_sdxlturbo(pipeline,ref_image):

    generator = prepare_seed()
    gen_image = pipeline(prompt                        = DEFAULT_PROMPT,
                         num_inference_steps           = INFERENCE_STEPS, 
                         guidance_scale                = 0.0 ,
                         width                         = WIDTH, 
                         height                        = HEIGHT, 
                         generator                     = generator,
                         image                         = ref_image, 
                         strength                      = DEFAULT_NOISE_STRENGTH, 
                        ).images[0]
                        
    return gen_image



def run_lcm_or_sdxl():

    ###
    ### PREPARE MODELS
    ###

    pipeline  = prepare_lcm_controlnet_or_sdxlturbo_pipeline()
    
    processor  = process_lcm if MODEL=="lcm" else process_sdxlturbo

    run_model  = run_lcm if MODEL=="lcm" else run_sdxlturbo

    ###
    ### PREPARE WEBCAM 
    ###

    # Open a connection to the webcam
    cap = cv.VideoCapture(0)

    CAP_WIDTH  = cap.get(cv.CAP_PROP_FRAME_WIDTH)  #320
    CAP_HEIGHT = cap.get(cv.CAP_PROP_FRAME_HEIGHT) #240

    cap.set(cv.CAP_PROP_FRAME_WIDTH, CAP_WIDTH/2) 
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, CAP_HEIGHT/2)

    ###
    ### RUN WEBCAM AND DIFFUSION
    ###

    while True:
        # Read a frame from the webcam
        ret, image = cap.read()

        # break if cap returns false
        if not ret:
            print("Error: Failed to capture frame.")
            break
    
        # Calculate the center position for the black and white filter
        center_x = (image.shape[1] - WIDTH) // 2
        center_y = (image.shape[0] - HEIGHT) // 2

        result_image, masked_image = get_result_and_mask(image, center_x, center_y, WIDTH, HEIGHT)

        numpy_image = processor(masked_image)
        pil_image   = convert_numpy_image_to_pil_image(numpy_image)
        pil_image   = run_model(pipeline, pil_image)

        result_image[center_y:center_y+HEIGHT, center_x:center_x+WIDTH] = cv.cvtColor(np.array(pil_image), cv.COLOR_RGB2BGR)

        # Display the resulting frame
        cv.imshow("output", result_image)

        # Break the loop when 'q' key is pressed
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close all windows
    cap.release()
    cv.destroyAllWindows()

###
### RUN SCRIPT
###
#@app.on_event("startup")
#async def startup_event():
#    print("FastAPI application has started.")
#    prepare_lcm_controlnet_or_sdxlturbo_pipeline()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Receive the frame as bytes from the client
            data = await websocket.receive_bytes()

            # Debugging: Check if any data is received
            print(f"Received {len(data)} bytes from the client.")

            # Convert the received bytes to a numpy array
            nparr = np.frombuffer(data, np.uint8)

            # Decode the numpy array to an image (OpenCV)
            frame = cv.imdecode(nparr, cv.IMREAD_COLOR)

            # Check if the frame is valid
            if frame is None:
                print("Received an invalid image frame")
                continue

            # Process the frame (e.g., convert to grayscale)
            gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            # Encode the processed frame back to JPEG format
            _, buffer = cv.imencode('.jpg', gray_frame)

            # Debugging: Check the size of the processed frame
            print(f"Sending {len(buffer)} bytes back to the client.")

            # Send the processed frame back to the client
            await websocket.send_bytes(buffer.tobytes())

    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        await websocket.close()




