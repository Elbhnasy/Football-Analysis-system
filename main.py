from utils.videos_utils import read_video, save_video

def main():
    input_video_path = "/home/fox/Desktop/Football-Analysis-system/videos/input_videos/08fd33_4.mp4"
    output_video_path = "/home/fox/Desktop/Football-Analysis-system/videos/output_videos/output.avi"

    input_video_frames = read_video(input_video_path)
    save_video(input_video_frames, output_video_path)

if __name__ == "__main__":
    main()

