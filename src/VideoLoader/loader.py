# Use opencv to load mp4 video with audio track separation support
import cv2
import os
import ffmpeg  # Importa a biblioteca ffmpeg-python
import numpy as np
import os # Para remover arquivos temporários




class VideoLoader:
    def __init__(self, path_directory):
        """
        Initialize the VideoLoader with a directory containing video files.
        Args:
            path_directory (str): Directory containing video files.
        """
        self.path_directory = path_directory
        self.video_paths = [f"{path_directory}/{file}" for file in os.listdir(path_directory) 
                           if file.endswith(('.mp4', '.mkv'))]
    
    def load_video(self, video_path):
        """
        Load a specific video file (audio, frames, t) using cv2
        Args:
            video_path (str): Path to the video file.
        Returns:
            dict
        """
        x = cv2.VideoCapture(video_path)
        if not x.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        return ... # TODO

    def load_heaviest_video(self):
        """
        Load the heaviest video file in the directory.
        Returns:
            cv2.VideoCapture: OpenCV VideoCapture object for the heaviest video.
        """
        if not self.video_paths:
            raise ValueError("No video files found in the directory.")
        heaviest_video = max(self.video_paths, key=lambda p: os.path.getsize(p))
        return self.load_video(heaviest_video)
    


