import numpy as np
import os
import cv2
import insightface
from insightface.app import FaceAnalysis
from moviepy.editor import VideoFileClip, VideoClip
from functools import partial
import torch
import onnxruntime as ort

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize the face analysis application
app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

# Load face swapper model
swapper = insightface.model_zoo.get_model('inswapper_128.onnx', download=False, download_zip=False)

def process_frame(frame, target_face, app, swapper):
    # Face detection
    faces = app.get(frame)
    
    if len(faces) == 0:
        return frame
    
    # Use the first detected face
    source_face = faces[0]
    
    # Face swapping
    result = swapper.get(frame, source_face, target_face, paste_back=True)
    
    return result

def swap_faces_video(source_video_path, target_face_path, output_path, app, swapper):
    # Load the source video
    source_video = VideoFileClip(source_video_path)
    
    # Load the target face image
    target_face_img = cv2.imread(target_face_path)
    target_faces = app.get(target_face_img)
    
    if len(target_faces) == 0:
        raise ValueError("No face detected in the target image.")
    
    target_face = target_faces[0]
    
    # Create a function to process each frame
    process_frame_with_target = partial(process_frame, target_face=target_face, app=app, swapper=swapper)
    
    # Apply the face swap to each frame of the video
    output_video = source_video.fl_image(process_frame_with_target)
    
    # Write the output video
    output_video.write_videofile(output_path, codec="libx264")
    
    # Close the video clips
    source_video.close()
    output_video.close()

# Example usage
swap_faces_video("data_dst.mp4", "00001.png", "output_video.mp4", app, swapper)
