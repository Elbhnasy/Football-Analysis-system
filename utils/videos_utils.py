import cv2

def read_video(video_path):
    """
    Reads a video from the specified path and returns its frames.

    Args:
        video_path (str): Path to the input video file.

    Returns:
        list: A list of frames (numpy arrays) extracted from the video.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()
    return frames



def save_video(output_video_frames, output_video_path, fps=24, codec='XVID'):
    """
    Saves a list of frames as a video to the specified path.

    Args:
        output_video_frames (list): List of frames (numpy arrays) to be saved as a video.
        output_video_path (str): Path to the output video file.
        fps (int, optional): Frames per second for the output video. Default is 24.
        codec (str, optional): Codec to be used for the output video. Default is 'XVID'.

    Returns:
        None
    """
    if not output_video_frames:
        raise ValueError("The list of output video frames is empty.")

    # Ensure all frames are valid
    valid_frames = [frame for frame in output_video_frames if frame is not None]
    if not valid_frames:
        raise ValueError("The list of output video frames contains no valid frames.")

    frame_height, frame_width = valid_frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    for frame in valid_frames:
        out.write(frame)

    out.release()