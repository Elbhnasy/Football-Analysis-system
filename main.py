from utils.videos_utils import read_video, save_video
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
if __name__ == "__main__":
    main()

