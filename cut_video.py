import cv2
from moviepy.editor import VideoFileClip
import os

def cut_video(input_path, output_path, duration=10):
    # Load the video
    video = VideoFileClip(input_path)
    
    # Cut the video to the specified duration
    cut_video = video.subclip(0, duration)
    
    # Write the cut video to the output path
    cut_video.write_videofile(output_path, codec="libx264")
    
    # Close the video objects
    video.close()
    cut_video.close()

def cut_video_into_frames(input_path, output_folder, frame_rate=1):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Load the video
    video = cv2.VideoCapture(input_path)
    
    # Get video properties
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps / frame_rate)
    
    count = 0
    frame_count = 0
    
    while True:
        ret, frame = video.read()
        if not ret:
            break
        
        if count % frame_interval == 0:
            frame_path = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")
            cv2.imwrite(frame_path, frame)
            frame_count += 1
        
        count += 1
    
    # Release the video object
    video.release()
cut_video("data_dst.mp4", "dst.mp4")
