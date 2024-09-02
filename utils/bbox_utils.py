def get_center_of_bbox(bbox):
    """
    Calculate the center of a bounding box.

    Parameters:
    bbox (Tuple[int, int, int, int]): A tuple representing the bounding box in the format (x1, y1, x2, y2).

    Returns:
    Tuple[int, int]: A tuple representing the center (x, y) of the bounding box.
    
    Raises:
    ValueError: If the bounding box coordinates are not valid.
    """
    if len(bbox) != 4:
        raise ValueError("Bounding box must contain exactly 4 elements (x1, y1, x2, y2).")
    
    x1, y1, x2, y2 = bbox
    
    if x1 >= x2 or y1 >= y2:
        raise ValueError("Invalid bounding box coordinates: (x1, y1) should be top-left and (x2, y2) should be bottom-right.")
    
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    
    return center_x, center_y

def get_bbox_width(bbox) :
    """
    Calculate the width of a bounding box.

    Parameters:
    bbox (Tuple[int, int, int, int]): A tuple representing the bounding box in the format (x1, y1, x2, y2).

    Returns:
    int: The width of the bounding box.
    
    Raises:
    ValueError: If the bounding box coordinates are not valid.
    """
    if len(bbox) != 4:
        raise ValueError("Bounding box must contain exactly 4 elements (x1, y1, x2, y2).")
    
    x1, _, x2, _ = bbox
    
    if x1 >= x2:
        raise ValueError("Invalid bounding box coordinates: x1 should be less than x2.")
    
    return x2 - x1

def measure_distance(p1, p2):
    """
    Calculate the Euclidean distance between two points.

    Parameters:
    p1 (Tuple[int, int]): A tuple representing the first point in the format (x, y).
    p2 (Tuple[int, int]): A tuple representing the second point in the format (x, y).

    Returns:
    float: The Euclidean distance between the two points.
    """
    x1, y1 = p1
    x2, y2 = p2
    
    return ((x2 - x1)**2 + (y2 - y1)**2)**0.5

def measure_xy_distance(p1, p2):
    """
    Calculate the Euclidean distance between two points in the x-y plane.

    Parameters:
    p1 (Tuple[int, int]): A tuple representing the first point in the format (x, y).
    p2 (Tuple[int, int]): A tuple representing the second point in the format (x, y).

    Returns:
    float: The Euclidean distance between the two points in the x-y plane.
    """
    x1, y1 = p1
    x2, y2 = p2
    
    return (x2 - x1) , (y2 - y1)

def get_foot_position(bbox):
    """
    Calculate the foot position of a player.

    Parameters:
    bbox (Tuple[int, int, int, int]): A tuple representing the bounding box in the format (x1, y1, x2, y2).

    Returns:
    Tuple[int, int]: A tuple representing the foot position (x, y) of the player.
    
    Raises:
    ValueError: If the bounding box coordinates are not valid.
    """
    if len(bbox) != 4:
        raise ValueError("Bounding box must contain exactly 4 elements (x1, y1, x2, y2).")
    
    x1, y1, x2, y2 = bbox
    
    if x1 >= x2 or y1 >= y2:
        raise ValueError("Invalid bounding box coordinates: (x1, y1) should be top-left and (x2, y2) should be bottom-right.")
    
    foot_x = (x1 + x2) // 2
    foot_y = int(y2)
    
    return foot_x, foot_y
