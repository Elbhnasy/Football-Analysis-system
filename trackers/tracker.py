from ultralytics import YOLO
import supervision as sv
import pickle
import cv2
import os
import numpy as np
import pandas as pd

class Tracker:
    def __init__(self, model_path):
        """
        Initializes the Tracker with a YOLO model and ByteTrack tracker.

        Args:
            model_path (str): Path to the YOLO model file.
        """
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def detect_frames(self, frames, batch_size=20, conf=0.1):
        """
        Detects objects in the given frames using the YOLO model.

        Args:
            frames (list): List of frames (numpy arrays) to detect objects in.
            batch_size (int, optional): Number of frames to process in a batch. Default is 20.
            conf (float, optional): Confidence threshold for detections. Default is 0.1.

        Returns:
            list: List of detections for each frame.
        """
        detections = []
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size], conf=conf)
            detections.extend(detections_batch)
        return detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        """
        Gets object tracks from the given frames, optionally reading from a stub file.

        Args:
            frames (list): List of frames (numpy arrays) to track objects in.
            read_from_stub (bool, optional): Whether to read tracks from a stub file. Default is False.
            stub_path (str, optional): Path to the stub file. Default is None.

        Returns:
            dict: Dictionary containing tracks for players, referees, and ball.
        """
        if read_from_stub and stub_path and os.path.exists(stub_path):
            with open(stub_path, "rb") as f:
                return pickle.load(f)

        detections = self.detect_frames(frames)
        tracks = {
            "players": [],
            "referees": [],
            "ball": []
        }

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            # Create an inverse mapping of class names to class IDs
            cls_names_inv = {v: k for k, v in cls_names.items()}

            # Convert detections to supervision format
            detection_supervision = sv.Detections.from_ultralytics(detection)

            # Convert goalkeepers to players
            for object_ind, class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[object_ind] = cls_names_inv["player"]

            # # Track objects using the ByteTrack tracker
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)
            
            # Initialize dictionaries for the current frame's tracks
            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})
            
            # Process tracked detections
            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                class_id = frame_detection[3]
                track_id = frame_detection[4]
                # Add player tracks
                if cls_names[class_id] == "player":
                    tracks["players"][frame_num][track_id] = {"bbox": bbox}
                # Add referee tracks
                elif cls_names[class_id] == "referee":
                    tracks["referees"][frame_num][track_id] = {"bbox": bbox}

            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                class_id = frame_detection[3]
                # Add ball tracks
                if cls_names[class_id] == "ball":
                    tracks["ball"][frame_num][1] = {"bbox": bbox}

        if stub_path:
            with open(stub_path, "wb") as f:
                pickle.dump(tracks, f)

        return tracks
