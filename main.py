import cv2
import numpy as np
from utils.videos_utils import read_video, save_video
from team_assigner.team_assigner import TeamAssigner
from trackers.tracker import Tracker
from player_ball_assigner.player_ball_assigner import PlayerBallAssigner  

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


    # Save the output video
    save_video(output_video_frames, output_video_path)
    
if __name__ == "__main__":
    main()

