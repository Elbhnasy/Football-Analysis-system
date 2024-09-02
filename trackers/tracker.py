from ultralytics import YOLO
import supervision as sv
import pickle
import cv2
import os
import numpy as np
import pandas as pd
import sys
sys.path.append('../')
from utils.bbox_utils import get_center_of_bbox, get_bbox_width

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

    def draw_ellipse(self, frame, bbox, color, track_id=None):
        """
        Draws an ellipse on the given frame.

        Args:
            frame (numpy array): Frame to draw the ellipse on.
            bbox (tuple): Bounding box coordinates in the format (x1, y1, x2, y2).
            color (tuple): Color of the ellipse in the format (B, G, R).
            track_id (int, optional): ID of the track to draw. Default is None.

        Returns:
            numpy array: Frame with the ellipse drawn on it.
        """
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)
        y2 = int(bbox[3])
        cv2.ellipse(
            img = frame,
            center = (x_center, y2),
            axes = (int(width) , int(width*0.35)),
            angle = 0.0,
            startAngle = -45,
            endAngle = 235,
            color = color,
            thickness = 2,
            lineType = cv2.LINE_4
        )

        rectangle_width = 40
        rectangle_height=20
        x1_rect = x_center - rectangle_width//2
        x2_rect = x_center + rectangle_width//2
        y1_rect = (y2- rectangle_height//2) +15
        y2_rect = (y2+ rectangle_height//2) +15

        if track_id is not None:
            cv2.rectangle(
                frame,
                (int(x1_rect), int(y1_rect)),
                (int(x2_rect), int(y2_rect)),
                color,
                cv2.FILLED
            )

            x1_text = x1_rect+12
            if track_id > 99:
                x1_text -=10

            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text),int(y1_rect+15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,0,0),
                2
            )
        return frame

    def draw_traingle(self, frame, bbox, color):
        """
        Draws a triangle on the given frame.

        Args:
            frame (numpy array): Frame to draw the triangle on.
            bbox (tuple): Bounding box coordinates in the format (x1, y1, x2, y2).
            color (tuple): Color of the triangle in the format (B, G, R).

        Returns:
            numpy array: Frame with the triangle drawn on it.
        """
        y = int(bbox[1])
        x, _ = get_center_of_bbox(bbox)

        triangle_points = np.array([
            [x, y],
            [x-10, y-20],
            [x+10, y-20]
        ], np.int32)
        cv2.drawContours([frame], [triangle_points], 0, color, cv2.FILLED)
        cv2.drawContours([frame], [triangle_points], 0, (0,0,0), 2)
        
        return frame

    
    def draw_annotations(self,video_frames, tracks,team_ball_contro):
        """
        Draws annotations on the given video frames.

        Args:
            video_frames (list): List of frames (numpy arrays) to draw annotations on.
            tracks (dict): Dictionary containing tracks for players, referees, and ball.
            team_ball_contro (str): The team that has control of the ball.

        Returns:
            list: List of frames with annotations drawn on them.
        """
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()
            player_dict = tracks["players"][frame_num]
            referee_dict = tracks["referees"][frame_num]
            ball_dict = tracks["ball"][frame_num]

            # Draw player tracks
            for track_id, player in player_dict.items():
                color = player.get("team_color",(0,0,255))
                frame = self.draw_ellipse(frame, player["bbox"],color, track_id)

                if player.get('has_ball',False):
                    frame = self.draw_traingle(frame, player["bbox"],(0,0,255))

            # Draw referee tracks
            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"],(0,255,255))          

            # Draw ball tracks
            for track_id, ball in ball_dict.items():
                frame = self.draw_traingle(frame, ball["bbox"],(0,255,0))

            # Draw team ball control
            frame = self.draw_team_ball_control(frame, frame_num, team_ball_contro)

            output_video_frames.append(frame)

        return output_video_frames
