import cv2
import numpy as np
from utils.videos_utils import read_video, save_video
from team_assigner.team_assigner import TeamAssigner
from trackers.tracker import Tracker
from speed_and_distance.speed_and_distance_estimator import SpeedAndDistanceEstimator
from view_transformer.view_transformer import ViewTransformer
from player_ball_assigner.player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator.camera_movement_estimator import CameraMovementEstimator


class FootballAnalysisSystem:
    def __init__(self, input_video_path, output_video_path, model_path, track_stub_path, camera_stub_path):
        """
        Initializes the Football Analysis System with paths to required resources.
        """
        self.input_video_path = input_video_path
        self.output_video_path = output_video_path
        self.model_path = model_path
        self.track_stub_path = track_stub_path
        self.camera_stub_path = camera_stub_path

        # Load video frames
        self.input_video_frames = read_video(input_video_path)

        # Initialize components
        self.tracker = Tracker(model_path)
        self.camera_movement_estimator = CameraMovementEstimator(self.input_video_frames[0])
        self.view_transformer = ViewTransformer()
        self.speed_and_distance_estimator = SpeedAndDistanceEstimator()
        self.team_assigner = TeamAssigner()
        self.player_assigner = PlayerBallAssigner()

        self.tracks = {}
        self.camera_movement_per_frame = []
        self.team_ball_control = []

    def process_video(self):
        """
        Processes the input video: tracks objects, estimates camera movement, transforms views,
        assigns teams, calculates speed and distance, and assigns ball control.
        """
        self._track_objects()
        self._estimate_camera_movement()
        self._adjust_object_positions()
        self._transform_object_views()
        self._interpolate_ball_positions()
        self._estimate_speed_and_distance()
        self._assign_teams()
        self._assign_ball_possession()
        self._draw_annotations()

        # Save the processed video
        save_video(self.input_video_frames, self.output_video_path)

    def _track_objects(self):
        """
        Tracks objects using the provided tracker and reads from the stub if available.
        """
        try:
            self.tracks = self.tracker.get_object_tracks(self.input_video_frames, read_from_stub=True, stub_path=self.track_stub_path)
            self.tracker.add_posstion_to_tracks(self.tracks)
        except Exception as e:
            raise RuntimeError(f"Error in object tracking: {e}")

    def _estimate_camera_movement(self):
        """
        Estimates camera movement per frame using the stub if available.
        """
        try:
            self.camera_movement_per_frame = self.camera_movement_estimator.get_camera_movement(
                self.input_video_frames, read_from_stub=True, stub_path=self.camera_stub_path
            )
        except Exception as e:
            raise RuntimeError(f"Error in camera movement estimation: {e}")

    def _adjust_object_positions(self):
        """
        Adjusts the positions of tracked objects based on camera movement.
        """
        self.camera_movement_estimator.add_adjust_positions_to_tracks(self.tracks, self.camera_movement_per_frame)

    def _transform_object_views(self):
        """
        Transforms the view of tracked objects to a top-down perspective.
        """
        self.view_transformer.add_transformed_position_to_tracks(self.tracks)

    def _interpolate_ball_positions(self):
        """
        Interpolates missing ball positions.
        """
        self.tracks["ball"] = self.tracker.interpolate_ball_positions(self.tracks["ball"])

    def _estimate_speed_and_distance(self):
        """
        Estimates speed and distance of objects and adds it to the tracks.
        """
        self.speed_and_distance_estimator.add_speed_and_distance_to_tracks(self.tracks)

    def _assign_teams(self):
        """
        Assigns teams to the players based on their appearance and track data.
        """
        try:
            self.team_assigner.assign_team_colors(self.input_video_frames[0], self.tracks['players'][0])
            for frame_num, player_track in enumerate(self.tracks['players']):
                for player_id, track in player_track.items():
                    team = self.team_assigner.get_player_team(self.input_video_frames[frame_num], track['bbox'], player_id)
                    self.tracks['players'][frame_num][player_id]['team'] = team
                    self.tracks['players'][frame_num][player_id]['team_color'] = self.team_assigner.team_colors[team]
        except Exception as e:
            raise RuntimeError(f"Error in team assignment: {e}")

    def _assign_ball_possession(self):
        """
        Assigns ball possession to players and tracks the team controlling the ball.
        """
        self.team_ball_control = []
        try:
            for frame_num, player_track in enumerate(self.tracks['players']):
                ball_bbox = self.tracks['ball'][frame_num][1]['bbox']
                assigned_player = self.player_assigner.assign_ball_to_players(player_track, ball_bbox)
                if assigned_player != -1:
                    self.tracks['players'][frame_num][assigned_player]['has_ball'] = True
                    self.team_ball_control.append(self.tracks['players'][frame_num][assigned_player]['team'])
                else:
                    self.team_ball_control.append(self.team_ball_control[-1] if self.team_ball_control else None)
            self.team_ball_control = np.array(self.team_ball_control)
        except Exception as e:
            raise RuntimeError(f"Error in ball possession assignment: {e}")

    def _draw_annotations(self):
        """
        Draws annotations for object tracks, camera movement, and speed/distance, and overlays them on video frames.
        """
        try:
            self.input_video_frames = self.tracker.draw_annotations(self.input_video_frames, self.tracks, self.team_ball_control)
            self.input_video_frames = self.camera_movement_estimator.draw_camera_movement(self.input_video_frames, self.camera_movement_per_frame)
            self.speed_and_distance_estimator.draw_speed_and_distance(self.input_video_frames, self.tracks)
        except Exception as e:
            raise RuntimeError(f"Error in drawing annotations: {e}")


if __name__ == "__main__":
    input_video_path = "/home/fox/Desktop/Football-Analysis-system/videos/input_videos/08fd33_4.mp4"
    output_video_path = "/home/fox/Desktop/Football-Analysis-system/videos/output_videos/output.avi"
    model_path = "/home/fox/Desktop/Football-Analysis-system/models/v5.pt"
    track_stub_path = "/home/fox/Desktop/Football-Analysis-system/stubs/track_stub.pkl"
    camera_stub_path = "/home/fox/Desktop/Football-Analysis-system/stubs/camera_movement_stub.pkl"

    # Run the Football Analysis System
    system = FootballAnalysisSystem(input_video_path, output_video_path, model_path, track_stub_path, camera_stub_path)
    system.process_video()
