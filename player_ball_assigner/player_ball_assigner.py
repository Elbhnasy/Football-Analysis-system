import sys
sys.path.append('../')
from utils.bbox_utils import get_center_of_bbox, measure_distance

class PlayerBallAssigner:
    def __init__(self):
        self.max_playr_ball_distance = 70
    
    def assign_ball_to_players(self, players, ball_bbox):
        """
        Assigns the ball to the closest player.

        Args:
            players (dict): Dictionary containing player IDs and their bounding boxes.
            ball_bbox (Tuple[int, int, int, int]): Bounding box of the ball in the format (x1, y1, x2, y2).

        Returns:
            dict: Dictionary containing player IDs and their assigned ball.
        """
        ball_position = get_center_of_bbox(ball_bbox)

        min_distance = float('inf')
        assigned_player = -1

        for player_id, player in players.items():
            player_bbox = player['bbox']

            # Calculate the distance between the player and the ball
            distance_left = measure_distance((player_bbox[0],player_bbox[-1]), ball_position)
            distance_right = measure_distance((player_bbox[2],player_bbox[-1]), ball_position)
            distance = min(distance_left, distance_right)

            if distance < self.max_playr_ball_distance:
                if distance < min_distance:
                    min_distance = distance
                    assigned_player = player_id
        return assigned_player

