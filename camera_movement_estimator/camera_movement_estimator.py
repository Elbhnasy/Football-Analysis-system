import cv2
import os
import sys
import pickle
import numpy as np
sys.path.append("../")
from utils import measure_distance , measure_xy_distance


class CameraMovementEstimator():
    def __init__(self,frame):
        # define the minimum distance to be considered as movement
        self.mimimum_distance = 5

        # define lk_params for lucas kanade optical flow
        self.lk_params = dict(
                            winSize  = (15,15),
                            maxLevel = 2,
                            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)  
                            )
        # define the mask for goodFeaturesToTrack
        first_frame_graysacle = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask_features = np.zeros_like(first_frame_graysacle)
        mask_features[: , 0 : 20] = 1
        mask_features[: , 900 :1050] = 1

        # define the feature_params for goodFeaturesToTrack
        self.feature_params = dict(
                                maxCorners = 100,
                                qualityLevel = 0.3,
                                minDistance = 3,
                                blockSize = 7,
                                mask = mask_features
                                )
    def add_adjust_positions_to_tracks(self, tracks, camera_movement_per_frame):
        """
        Adjust the position of the tracks based on the camera movement

        Parameters:
        tracks: list
            The list of tracks to adjust the position
        camera_movement_per_frame: list
            The camera movement per frame

        Returns:
        adjusted_tracks: list
            The list of adjusted tracks
        """
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    # get the position of the track
                    position = track_info['position']
                    # get the camera movement for the frame
                    camera_movement = camera_movement_per_frame[frame_num]
                    # adjust the position based on the camera movement
                    position_adjusted = (position[0]-camera_movement[0],position[1]-camera_movement[1])
                    # add the adjusted position to the track
                    tracks[object][frame_num][track_id]['position_adjusted'] = position_adjusted
                    


    def get_camera_movement(self, frames, read_from_stub= False, stub_path = None):
        """
        Get the camera movement from the previous frame to the current frame

        Parameters:
        frame: np.array
            The current frame to calculate the camera movement from the previous frame
        read_from_stub: bool
            If True, the function will read the previous frame from the stub_path
        stub_path: str  
            The path to the previous frame

        Returns:
        camera_movement: tuple
            The camera movement in the x and y direction
        """

        # read the stub frame if read_from_stub is True
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, "rb") as f:
                return pickle.load(f)
        
        camera_movement = [[0,0]]*len(frames)

        # get the first frame and the features
        old_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        old_features = cv2.goodFeaturesToTrack(old_gray, **self.feature_params)

        for frame_num in range(1, len(frames)):
            frame_gray = cv2.cvtColor(frames[frame_num], cv2.COLOR_BGR2GRAY)
            
            # calculate the optical flow
            new_features,_ , _ = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, old_features, None, **self.lk_params)

            max_distance, camera_movement_X, camera_movement_Y = 0, 0, 0

            for i, (new, old) in enumerate(zip(new_features, old_features)):
                # get the x and y coordinates of the new and old features
                new_features_point = new.ravel()
                old_features_point = old.ravel()

                # calculate the distance between the new and old features
                distance = measure_distance(new_features_point, old_features_point)

                if distance > max_distance:
                    max_distance = distance
                    # calculate the movement in the x and y direction
                    camera_movement_X, camera_movement_Y = measure_xy_distance(old_features_point, new_features_point)

            # if the distance is greater than the minimum distance, then consider it as movement
            if max_distance > self.mimimum_distance:
                camera_movement[frame_num] = [camera_movement_X, camera_movement_Y]
                old_features = cv2.goodFeaturesToTrack(frame_gray, **self.feature_params)

            old_gray = frame_gray.copy()

        if stub_path is not None:
            with open(stub_path, "wb") as f:
                pickle.dump(camera_movement, f)
        
        return camera_movement

    def draw_camera_movement(self, frames, camera_movement_per_frame):
        """
        Draw the camera movement on the frames

        Parameters:
        frames: list
            The list of frames to draw the camera movement on
        camera_movement_per_frame: list
            The camera movement per frame

        Returns:
        frames_with_movement: list
            The list of frames with the camera movement drawn on them
        """

        output_frames = []

        for frame_num, frame in enumerate(frames):
            frame = frame.copy()
            overlay = frame.copy()
            # draw a white rectangle on the top of the frame
            cv2.rectangle(overlay,(0,0),(385,80),(255,255,255),-1)
            alpha =0.6
            # apply the overlay
            cv2.addWeighted(overlay,alpha,frame,1-alpha,0,frame)
            # get the x and y movement from the camera_movement_per_frame
            X_movement, Y_movement = camera_movement_per_frame[frame_num]
            # draw the movement on the frame
            frame = cv2.putText(frame,f"Movement (x) : {X_movement:.2f}",(10,30), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3)
            frame = cv2.putText(frame,f"Movement (y) : {Y_movement:.2f}",(10,60), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3)

            output_frames.append(frame) 

        return output_frames
