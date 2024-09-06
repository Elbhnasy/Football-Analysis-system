import numpy as np
import cv2

class ViewTransformer:
    def __init__(self, court_width=68, court_length=23.32):
        """
        Initializes the ViewTransformer with the court dimensions and calculates the perspective transformation matrix.
        
        Args:
            court_width (float): The width of the court in real-world units.
            court_length (float): The length of the court in real-world units.
        """
        self.pixel_vertices =  np.array([
                                        [110, 1035], 
                                        [265, 275],
                                        [910, 260], 
                                        [1640, 915]
                                    ], dtype=np.float32)
                                    
        self.real_world_vertices = np.array([
                                            [0, court_width],
                                            [0, 0],
                                            [court_length, 0],
                                            [court_length, court_width]
                                        ], dtype=np.float32)

        self.transformation_matrix = cv2.getPerspectiveTransform(self.pixel_vertices, self.real_world_vertices)

    def transform_point(self, point):
        """
        Transforms a point from pixel coordinates to real-world coordinates.
        
        Args:
            point (list or tuple): The point to transform in the format [x, y].
        
        Returns:
            list: The transformed point in real-world coordinates, or None if the point is outside the court.
        """
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ValueError("Point must be a list or tuple with two elements [x, y].")
        
        point = (int(point[0]), int(point[1]))
        
        # Check if the point is inside the court
        if cv2.pointPolygonTest(self.pixel_vertices, point, False) < 0:
            return None

        reshaped_point = np.array(point, dtype=np.float32).reshape(-1, 1, 2)
        transformed_point = cv2.perspectiveTransform(reshaped_point, self.transformation_matrix)
        return transformed_point.reshape(-1, 2).tolist()

    def add_transformed_position_to_tracks(self, tracks):
        """
        Adds the transformed positions of players and the ball to the tracks.
        
        Args:
            tracks (dict): The object tracks containing positions to be transformed.
        """
        if not isinstance(tracks, dict):
            raise ValueError("Tracks must be a dictionary.")
        
        for object_type, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    position = track_info.get('position_adjusted')
                    if position is None:
                        continue
                    
                    transformed_position = self.transform_point(position)
                    if transformed_position is not None:
                        tracks[object_type][frame_num][track_id]['position_transformed'] = transformed_position
