import cv2
import numpy as np
from PIL import Image, ImageFilter

import torch
from diffusers import AutoencoderTiny, StableDiffusionPipeline

from streamdiffusion import StreamDiffusion
from streamdiffusion.image_utils import postprocess_image
# from streamdiffusion.acceleration.tensorrt import accelerate_with_tensorrt

PROMPT        = "pikachu, yellow skin, furry, wearing a beanie, style of pokemon, cartoon, 8k"
PROMPT        = "star wars porg, furry, cute, Disney" # GOOD
# PROMPT        = "Disney yellow minion, black goggles, plastic, style of pixar"
# PROMPT        = "Gordon, Half-Life 2, Valve, style of video game"
# PROMPT        = "Darth Vader, Star Wars, waering helmet, black mask" # so so 
# PROMPT        = "Goku, Dragon Ball Z, yellow hair"
# PROMPT        = "portrait of a man on fire with a forrest on fire, burning, smoke background"
# PROMPT        = "Ghost Rider, flaming skull" #GOOD
PROMPT        = "Ghost Rider, flaming skull, eye sockets, white skeleton" #GOOD

# PROMPT      = "santa claus, thick frame glasses"
# PROMPT      = "toy story, buzz light-year, render 3d, pixar"
WIDTH, HEIGHT = 512,512
TORCH_DEVICE  = 'mps' # "cuda" if you have a cuda GPU
TORCH_DTYPE   = torch.float16

def get_result_and_mask(frame, center_x, center_y, width, height):
    "just gets full frame and the mask for cutout"

    mask = np.zeros_like(frame)
    mask[center_y:center_y+height, center_x:center_x+width, :] = 255
    cutout = frame[center_y:center_y+height, center_x:center_x+width, :]

    return frame, cutout


model_loc = "models/LCM_Dreamshaper_v7"
pipe = StableDiffusionPipeline.from_pretrained(model_loc).to(
    device=torch.device(TORCH_DEVICE),
    dtype=TORCH_DTYPE,
)

stream = StreamDiffusion(
    pipe,
    t_index_list=[32, 45],
    torch_dtype=TORCH_DTYPE,
    do_add_noise=False,
)

# If the loaded model is not LCM, merge LCM
stream.load_lcm_lora()
stream.fuse_lora()

# Use Tiny VAE for further acceleration 
stream.vae = AutoencoderTiny.from_pretrained("madebyollin/taesd").to(device=pipe.device, dtype=pipe.dtype)

# Enable acceleration (FOR THOSE WITH CUDA GPU)
# pipe.enable_xformers_memory_efficient_attention()

# Prepare the stream
stream.prepare(prompt = PROMPT, 
        num_inference_steps=50,
        guidance_scale=0)

# optional
stream.enable_similar_image_filter(
    # similar_image_filter_threshold,
    # similar_image_filter_max_skip_frame
    )

# Open a connection to the webcam
cap = cv2.VideoCapture(0)

CAP_WIDTH  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  #320
CAP_HEIGHT = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) #240

cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_WIDTH/2) 
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_HEIGHT/2)

# Check if the webcam is opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Read a frame from the webcam (for warmup)
ret, image = cap.read()
center_x = (image.shape[1] - WIDTH) // 2
center_y = (image.shape[0] - HEIGHT) // 2
result_image, image_cutout = get_result_and_mask(image, center_x, center_y, WIDTH, HEIGHT)

# Warmup >= len(t_index_list) x frame_buffer_size
for _ in range(4):
    stream(image_cutout)

print("here.")

# Run the stream infinitely
while True:

    # Read frame (image) from the webcam
    ret, frame = cap.read()

    # Break the loop if reading the frame fails
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # get center
    center_x = (frame.shape[1] - WIDTH) // 2
    center_y = (frame.shape[0] - HEIGHT) // 2

    result_image, result_cutout = get_result_and_mask(frame, center_x, center_y, WIDTH, HEIGHT)
    result_cutout = Image.fromarray(cv2.cvtColor(result_cutout, cv2.COLOR_BGR2RGB)) 

    x_output = stream(result_cutout)
    rendered_image = postprocess_image(x_output, output_type="pil")[0]#.show()

    result_image[center_y:center_y+HEIGHT, center_x:center_x+WIDTH] = cv2.cvtColor(np.array(rendered_image), cv2.COLOR_RGB2BGR)

    # Display output
    cv2.imshow("output", result_image)

      # Break the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()


#################
#################
#################

