import cv2 as cv
import mediapipe as mp

mpDraw = mp.solutions.drawing_utils
mpHands = mp.solutions.hands
hands = mpHands.Hands()

cap = cv.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

img_1 = cv.imread('magic_circles/magic_circle_5.png', -1)
img_2 = cv.imread('magic_circles/magic_circle_4.png', -1)

orangeColor = (0, 140, 255)
deg = 0

def position_data(lmlist):
    global wrist, thumb_tip, index_mcp, index_tip, midle_mcp, midle_tip, \
        ring_tip, pinky_tip
    wrist = (lmlist[0][0], lmlist[0][1])
    thumb_tip = (lmlist[4][0], lmlist[4][1])
    index_mcp = (lmlist[5][0], lmlist[5][1])
    index_tip = (lmlist[8][0], lmlist[8][1])
    midle_mcp = (lmlist[9][0], lmlist[9][1])
    midle_tip = (lmlist[12][0], lmlist[12][1])
    ring_tip  = (lmlist[16][0], lmlist[16][1])
    pinky_tip = (lmlist[20][0], lmlist[20][1])

def draw_line(p1, p2, color=orangeColor, size=5):
    cv.line(frame, p1, p2, color, size)
    cv.line(frame, p1, p2, (255, 255, 255), round(size / 2))

def calculate_distance(p1,p2):
    x1, y1, x2, y2 = p1[0], p1[1], p2[0], p2[1]
    lenght = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** (1.0 / 2)
    return lenght

def asd(targetImg, x, y, size=None):
    if size is not None:
        targetImg = cv.resize(targetImg, size)

    newFrame = frame.copy()
    b, g, r, a = cv.split(targetImg)
    overlay_color = cv.merge((b, g, r))
    mask = cv.medianBlur(a, 1)
    h, w, _ = overlay_color.shape
    roi = newFrame[y:y + h, x:x + w]

    img1_bg = cv.bitwise_and(roi.copy(), roi.copy(), mask=cv.bitwise_not(mask))
    img2_fg = cv.bitwise_and(overlay_color, overlay_color, mask=mask)
    newFrame[y:y + h, x:x + w] = cv.add(img1_bg, img2_fg)

    return newFrame

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

            position_data(lmLists)
            palm = calculate_distance(wrist, index_mcp)
            distance = calculate_distance(index_tip, pinky_tip)
            ratio = distance / palm

            if (1.3 > ratio > 0.5):
                draw_line(wrist, thumb_tip)
                draw_line(wrist, index_tip)
                draw_line(wrist, midle_tip)
                draw_line(wrist, ring_tip)
                draw_line(wrist, pinky_tip)
                draw_line(thumb_tip, index_tip)
                draw_line(thumb_tip, midle_tip)
                draw_line(thumb_tip, ring_tip)
                draw_line(thumb_tip, pinky_tip)

            if (ratio > 1.3):
                    centerx = midle_mcp[0]
                    centery = midle_mcp[1]
                    shield_size = 3.0
                    diameter = round(palm * shield_size)
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
                    hei, wid, col = img_1.shape
                    cen = (wid // 2, hei // 2)
                    M1 = cv.getRotationMatrix2D(cen, round(deg), 1.0)
                    M2 = cv.getRotationMatrix2D(cen, round(360 - deg), 1.0)
                    rotated1 = cv.warpAffine(img_1, M1, (wid, hei))
                    rotated2 = cv.warpAffine(img_2, M2, (wid, hei))
                    if (diameter != 0):
                        frame = asd(rotated1, x1, y1, shield_size)
                        frame = asd(rotated2, x1, y1, shield_size)

    cv.imshow("Image", frame)
    if cv.waitKey(1) == ord("q"):
        break

cap.release()
cv.destroyAllWindows()
