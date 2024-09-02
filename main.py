import cv2
import numpy as np
from utils.videos_utils import read_video, save_video
from team_assigner.team_assigner import TeamAssigner
from trackers.tracker import Tracker

def main():
    input_video_path = "/home/fox/Desktop/Football-Analysis-system/videos/input_videos/08fd33_4.mp4"
    output_video_path = "/home/fox/Desktop/Football-Analysis-system/videos/output_videos/output.avi"
    model_path = "/home/fox/Desktop/Football-Analysis-system/models/v5.pt"
    stub_path = "/home/fox/Desktop/Football-Analysis-system/stubs/track_stub.pkl"
                
    # Read the input video
    input_video_frames = read_video(input_video_path)
    
    # Initialize the tracker
    tracker = Tracker(model_path)

    # Get the object tracks
    tracks = tracker.get_object_tracks(input_video_frames,
                                    read_from_stub=True,
                                    stub_path=stub_path)
    # Interpolate Ball Positions
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])


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

    # Draw output 
    ## Draw object Tracks
    output_video_frames = tracker.draw_annotations(input_video_frames, tracks,team_ball_control)

    # Save the output video
    save_video(output_video_frames, output_video_path)
if __name__ == "__main__":
    main()

