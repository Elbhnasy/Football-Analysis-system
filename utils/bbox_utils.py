def get_center_of_bbox(bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
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

def get_bbox_width(bbox: Tuple[int, int, int, int]) -> int:
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
