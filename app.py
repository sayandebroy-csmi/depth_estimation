# Import necessary libraries
import cv2
import torch
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
import uvicorn

# Load MiDaS model for depth estimation
model_type = "MiDaS_small"  # MiDaS v2.1 - Small (lowest accuracy, highest inference speed)
midas = torch.hub.load("intel-isl/MiDaS", model_type)

# Initialize FastAPI app
app = FastAPI()

# Serve static files from the "static" directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Set up templates directory for HTML responses
templates = Jinja2Templates(directory="templates")

# Function to estimate depth from a video frame
def estimate_depth(frame):
    # Convert frame to RGB and resize it to 384x384
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (384, 384))
    # Convert image to a tensor and normalize it
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    # Estimate depth with MiDaS model
    with torch.no_grad():
        prediction = midas(img)
        # Interpolate prediction to match original frame size
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=frame.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    # Convert depth map to numpy array and normalize it
    depth_map = prediction.cpu().numpy()
    depth_map = cv2.normalize(depth_map, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    # Return depth map as an 8-bit image
    return (depth_map * 255).astype(np.uint8)

# Function to process an input video and create an output video with depth maps
def process_video(file_path, output_path):
    video = cv2.VideoCapture(file_path)
    # Set up video writer with XVID codec
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = video.get(cv2.CAP_PROP_FPS)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), False)
    while True:
        ret, frame = video.read()
        if not ret:
            break
        # Estimate depth map for each frame
        depth_map = estimate_depth(frame)
        out.write(depth_map)
    # Release video capture and writer objects
    video.release()
    out.release()

# Main route for serving the home page
@app.get("/", response_class=HTMLResponse)
async def main(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Route to process uploaded video file and return the result
@app.post("/process_video/", response_class=HTMLResponse)
async def process_video_stream(request: Request, file: UploadFile = File(...)):
    input_file_path = f"static/{file.filename}"
    output_file_path = f"static/output_depth_video.avi"

    # Save uploaded file to the static directory
    with open(input_file_path, "wb") as f:
        f.write(file.file.read())
    # Process video to generate depth map video
    process_video(input_file_path, output_file_path)

    # Return result template with the path to the output video
    return templates.TemplateResponse("result.html", {"request": request, "output_video": output_file_path})

# Entry point for running the app with uvicorn
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
