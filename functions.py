import cv2 as cv

LINE_COLOR = (0, 140, 255)

def position_data(lmlist):
    """
    Calculates cordinate of tip 
    """
    wrist = (lmlist[0][0], lmlist[0][1])
    thumb_tip = (lmlist[4][0], lmlist[4][1])
    index_mcp = (lmlist[5][0], lmlist[5][1])
    index_tip = (lmlist[8][0], lmlist[8][1])
    midle_mcp = (lmlist[9][0], lmlist[9][1])
    midle_tip = (lmlist[12][0], lmlist[12][1])
    ring_tip  = (lmlist[16][0], lmlist[16][1])
    pinky_tip = (lmlist[20][0], lmlist[20][1])

    return [wrist, thumb_tip, index_mcp, index_tip, midle_mcp, midle_tip, ring_tip, pinky_tip]

def calculate_distance(p1, p2):
    x1, y1, x2, y2 = p1[0], p1[1], p2[0], p2[1]
    lenght = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** (1.0 / 2)
    return lenght

def draw_line(frame, p1, p2, color=LINE_COLOR, size=5):
    cv.line(frame, p1, p2, color, size)
    cv.line(frame, p1, p2, (255, 255, 255), round(size / 2))
    return frame

def asd(targetImg, frame, x, y, size=None):
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
