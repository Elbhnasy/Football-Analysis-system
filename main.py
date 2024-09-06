import cv2
import numpy as np
from utils.videos_utils import read_video, save_video
from team_assigner.team_assigner import TeamAssigner
from trackers.tracker import Tracker
from view_transformer.view_transformer import ViewTransformer
from player_ball_assigner.player_ball_assigner import PlayerBallAssigner  
from camera_movement_estimator.camera_movement_estimator import CameraMovementEstimator


def main():
    input_video_path = "/home/fox/Desktop/Football-Analysis-system/videos/input_videos/08fd33_4.mp4"
    output_video_path = "/home/fox/Desktop/Football-Analysis-system/videos/output_videos/output.avi"
    model_path = "/home/fox/Desktop/Football-Analysis-system/models/v5.pt"
    track_stub_path = "/home/fox/Desktop/Football-Analysis-system/stubs/track_stub.pkl"
    camera_stub_path = "/home/fox/Desktop/Football-Analysis-system/stubs/camera_movement_stub.pkl"  
    # Read the input video
    input_video_frames = read_video(input_video_path)
    
    # Initialize the tracker
    tracker = Tracker(model_path)

    # Get the object tracks
    tracks = tracker.get_object_tracks(input_video_frames,
                                    read_from_stub=True,
                                    stub_path=track_stub_path)
    # Get object positions 
    tracker.add_posstion_to_tracks(tracks)

    # Estimate Camera Movement
    camera_movement_estimator = CameraMovementEstimator(input_video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(input_video_frames,
                                                                            read_from_stub=True,
                                                                            stub_path = camera_stub_path)

    # Add Adjust Position to Track
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)
    
    # View Trasnformer
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)

    # Interpolate Ball Positions
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    # # Speed and distance estimator
    # speed_and_distance_estimator = SpeedAndDistance_Estimator()
    # speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)

    # Assign Player Teams
    team_assigner = TeamAssigner()
    team_assigner.assign_team_colors(input_video_frames[0], 
                                    tracks['players'][0])
    
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(input_video_frames[frame_num],   
                                                track['bbox'],
                                                player_id)
            tracks['players'][frame_num][player_id]['team'] = team 
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]
    # Assign Ball Aquisition
    player_assigner =PlayerBallAssigner()
    team_ball_control= []
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_players(player_track, ball_bbox)

        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            team_ball_control.append(team_ball_control[-1])
    team_ball_control= np.array(team_ball_control)

    # Draw output 
    ## Draw object Tracks
    output_video_frames = tracker.draw_annotations(input_video_frames, tracks, team_ball_control)

    ## Draw Camera Movement
    output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames, camera_movement_per_frame)
    # ## Draw Speed and Distance
    # speed_and_distance_estimator.draw_speed_and_distance(output_video_frames,tracks)

    # Save the output video
    save_video(output_video_frames, output_video_path)
    
if __name__ == "__main__":
    main()

