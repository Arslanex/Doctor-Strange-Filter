import cv2 as cv
import mediapipe as mp
from functions import position_data, calculate_distance, draw_line, asd

INNER_CIRCLE = "Models/inner_circles/orange.png"
OUTER_CIRCLE = "Models/outer_circles/orange.png"

# Camera Setup
cap = cv.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# Mediapipe setup for handlandmarks
mpDraw = mp.solutions.drawing_utils
mpHands = mp.solutions.hands
hands = mpHands.Hands()

# Initials
inner_circle = cv.imread(INNER_CIRCLE, -1)
outer_circle = cv.imread(OUTER_CIRCLE, -1)

LINE_COLOR = (0, 140, 255)
deg = 0

# Main Loop
while cap.isOpened():
    _, frame = cap.read()
    frame = cv.flip(frame, 1)
    rgbFrame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    results = hands.process(rgbFrame)
    if results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks:
            lmLists = []
            for id, lm in enumerate(hand.landmark):
                h,w,_ = frame.shape
                lmLists.append([int(lm.x * w), int(lm.y * h)])

            coordinates = position_data(lmLists)
            wrist, thumb_tip, index_mcp, index_tip, midle_mcp, midle_tip, ring_tip, pinky_tip = coordinates[0],coordinates[1], coordinates[2],coordinates[3], coordinates[4],coordinates[5], coordinates[6],coordinates[7] 
            index_wrist_distance = calculate_distance(wrist, index_mcp)
            index_pinks_distance = calculate_distance(index_tip, pinky_tip)
            ratio = index_pinks_distance/index_wrist_distance

            # Not enough
            if (1.3 > ratio > 0.5):
                frame=draw_line(frame, wrist, thumb_tip)
                frame=draw_line(frame, wrist, index_tip)
                frame=draw_line(frame, wrist, midle_tip)
                frame=draw_line(frame, wrist, ring_tip)
                frame=draw_line(frame, wrist, pinky_tip)
                frame=draw_line(frame, thumb_tip, index_tip)
                frame=draw_line(frame, thumb_tip, midle_tip)
                frame=draw_line(frame, thumb_tip, ring_tip)
                frame=draw_line(frame, thumb_tip, pinky_tip)
            
            elif (ratio > 1.3):
                centerx = midle_mcp[0]
                centery = midle_mcp[1]
                shield_size = 3.0
                diameter = round(index_wrist_distance * shield_size)
                x1 = round(centerx - (diameter / 2))
                y1 = round(centery - (diameter / 2))
                h, w, c = frame.shape
                if x1 < 0:
                    x1 = 0
                elif x1 > w:
                    x1 = w
                if y1 < 0:
                    y1 = 0
                elif y1 > h:
                    y1 = h
                if x1 + diameter > w:
                    diameter = w - x1
                if y1 + diameter > h:
                    diameter = h - y1
                shield_size = diameter, diameter
                ang_vel = 2.0
                deg = deg + ang_vel
                if deg > 360:
                    deg = 0
                hei, wid, col = outer_circle.shape
                cen = (wid // 2, hei // 2)
                M1 = cv.getRotationMatrix2D(cen, round(deg), 1.0)
                M2 = cv.getRotationMatrix2D(cen, round(360 - deg), 1.0)
                rotated1 = cv.warpAffine(outer_circle, M1, (wid, hei))
                rotated2 = cv.warpAffine(inner_circle, M2, (wid, hei))
                if (diameter != 0):
                    frame = asd(rotated1, frame, x1, y1, shield_size)
                    frame = asd(rotated2, frame, x1, y1, shield_size)

    cv.imshow("Image", frame)
    if cv.waitKey(1) == ord("q"):
        break

cap.release()
cv.destroyAllWindows()
