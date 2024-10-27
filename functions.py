import cv2 as cv
import numpy as np
from typing import List, Tuple

LINE_COLOR = (0, 140, 255)
WHITE_COLOR = (255, 255, 255)

def position_data(lmlist: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """
    Extracts and returns key fingertip and hand positions from the landmark list.

    Args:
        lmlist (list): List of tuples containing (x, y) coordinates of keypoints.

    Returns:
        list: Coordinates of wrist, thumb tip, index mcp, index tip,
              middle mcp, middle tip, ring tip, and pinky tip.
    """
    if len(lmlist) < 21:
        raise ValueError("Landmark list must contain at least 21 points.")

    keys = [0, 4, 5, 8, 9, 12, 16, 20]  # Indices of relevant landmarks
    return [lmlist[i] for i in keys]

def calculate_distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
    """
    Calculates the Euclidean distance between two points.

    Args:
        p1 (tuple): First point (x1, y1).
        p2 (tuple): Second point (x2, y2).

    Returns:
        float: Euclidean distance between the two points.
    """
    # Use numpy for faster computation
    return np.linalg.norm(np.array(p1) - np.array(p2))

def draw_line(
    frame: np.ndarray,
    p1: Tuple[int, int],
    p2: Tuple[int, int],
    color: Tuple[int, int, int] = LINE_COLOR,
    thickness: int = 5
) -> np.ndarray:
    """
    Draws a line between two points on the given frame.

    Args:
        frame (ndarray): The image/frame to draw the line on.
        p1 (tuple): First point (x1, y1).
        p2 (tuple): Second point (x2, y2).
        color (tuple): Color of the outer line.
        thickness (int): Thickness of the outer line.

    Returns:
        ndarray: Modified frame with the drawn line.
    """
    cv.line(frame, p1, p2, color, thickness)
    cv.line(frame, p1, p2, WHITE_COLOR, max(1, thickness // 2))
    return frame

def overlay_image(
    target_img: np.ndarray,
    frame: np.ndarray,
    x: int, y: int,
    size: Tuple[int, int] = None
) -> np.ndarray:
    """
    Overlays a target image onto a frame at the given position.

    Args:
        target_img (ndarray): Image to be overlaid with an alpha channel.
        frame (ndarray): Frame on which the target image is to be overlaid.
        x (int): X-coordinate for the top-left corner of the overlay.
        y (int): Y-coordinate for the top-left corner of the overlay.
        size (tuple, optional): Size to resize the target image. Defaults to None.

    Returns:
        ndarray: Modified frame with the overlaid image.
    """
    if size:
        try:
            target_img = cv.resize(target_img, size)
        except cv.error as e:
            raise ValueError(f"Error resizing the target image: {e}")

    if target_img.shape[-1] != 4:
        raise ValueError("Target image must have 4 channels (RGBA).")

    # Split the RGBA channels
    b, g, r, a = cv.split(target_img)
    overlay_color = cv.merge((b, g, r))
    mask = cv.medianBlur(a, 5)

    h, w, _ = overlay_color.shape
    if y + h > frame.shape[0] or x + w > frame.shape[1]:
        raise ValueError("Overlay exceeds frame boundaries.")

    roi = frame[y:y + h, x:x + w]

    # Create background and foreground masks
    img1_bg = cv.bitwise_and(roi, roi, mask=cv.bitwise_not(mask))
    img2_fg = cv.bitwise_and(overlay_color, overlay_color, mask=mask)

    # Overlay the foreground onto the background
    frame[y:y + h, x:x + w] = cv.add(img1_bg, img2_fg)

    return frame
