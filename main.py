import cv2 as cv
import mediapipe as mp
import json
from functions import position_data, calculate_distance, draw_line, overlay_image

def load_config(path: str = "config.json") -> dict:
    """Loads the configuration from a JSON file."""
    with open(path, "r") as file:
        return json.load(file)

def limit_value(val: int, min_val: int, max_val: int) -> int:
    """Clamps a value between a given min and max."""
    return max(min(val, max_val), min_val)

def initialize_camera(config: dict) -> cv.VideoCapture:
    """Initializes the webcam with given width, height, and device ID."""
    cap = cv.VideoCapture(config["camera"]["device_id"])
    cap.set(cv.CAP_PROP_FRAME_WIDTH, config["camera"]["width"])
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, config["camera"]["height"])
    if not cap.isOpened():
        raise RuntimeError("Failed to open the webcam.")
    return cap

def load_images(config: dict) -> tuple:
    """Loads overlay images and raises an error if any are missing."""
    inner_circle = cv.imread(config["overlay"]["inner_circle_path"], -1)
    outer_circle = cv.imread(config["overlay"]["outer_circle_path"], -1)
    if inner_circle is None or outer_circle is None:
        raise FileNotFoundError("Failed to load one or more overlay images.")
    return inner_circle, outer_circle

def process_frame(frame, hands, config, inner_circle, outer_circle, deg):
    """Processes the frame, applies overlays, and returns the updated frame."""
    h, w, _ = frame.shape
    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            lm_list = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks.landmark]

            (wrist, thumb_tip, index_mcp, index_tip,
             middle_mcp, middle_tip, ring_tip, pinky_tip) = position_data(lm_list)

            index_wrist_distance = calculate_distance(wrist, index_mcp)
            index_pinky_distance = calculate_distance(index_tip, pinky_tip)
            ratio = index_pinky_distance / index_wrist_distance

            if 0.5 < ratio < 1.3:
                fingers = [thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip]
                for finger in fingers:
                    frame = draw_line(frame, wrist, finger,
                                      color=tuple(config["line_settings"]["color"]),
                                      thickness=config["line_settings"]["thickness"])
                for i in range(len(fingers) - 1):
                    frame = draw_line(frame, fingers[i], fingers[i + 1],
                                      color=tuple(config["line_settings"]["color"]),
                                      thickness=config["line_settings"]["thickness"])

            elif ratio >= 1.3:
                center_x, center_y = middle_mcp
                diameter = round(index_wrist_distance * config["overlay"]["shield_size_multiplier"])

                x1 = limit_value(center_x - diameter // 2, 0, w)
                y1 = limit_value(center_y - diameter // 2, 0, h)
                diameter = min(diameter, w - x1, h - y1)

                deg = (deg + config["overlay"]["rotation_degree_increment"]) % 360
                M1 = cv.getRotationMatrix2D((outer_circle.shape[1] // 2, outer_circle.shape[0] // 2), deg, 1.0)
                M2 = cv.getRotationMatrix2D((inner_circle.shape[1] // 2, inner_circle.shape[0] // 2), -deg, 1.0)

                rotated_outer = cv.warpAffine(outer_circle, M1, (outer_circle.shape[1], outer_circle.shape[0]))
                rotated_inner = cv.warpAffine(inner_circle, M2, (inner_circle.shape[1], inner_circle.shape[0]))

                frame = overlay_image(rotated_outer, frame, x1, y1, (diameter, diameter))
                frame = overlay_image(rotated_inner, frame, x1, y1, (diameter, diameter))

    return frame, deg

def main():
    """Main function to run the application."""
    config = load_config()
    cap = initialize_camera(config)
    inner_circle, outer_circle = load_images(config)
    hands = mp.solutions.hands.Hands()
    deg = 0

    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Failed to capture frame.")
                break

            frame = cv.flip(frame, 1)
            frame, deg = process_frame(frame, hands, config, inner_circle, outer_circle, deg)

            cv.imshow("Image", frame)
            if cv.waitKey(1) == ord(config["keybindings"]["quit_key"]):
                break
    finally:
        cap.release()
        cv.destroyAllWindows()

if __name__ == "__main__":
    main()
