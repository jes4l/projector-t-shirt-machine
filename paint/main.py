from handTracker import *
import cv2
import numpy as np
import random
import mediapipe as mp
import time
import math


class ColorRect:
    def __init__(self, x, y, w, h, color, text="", alpha=0.5):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.color = color
        self.text = text
        self.alpha = alpha

    def drawRect(
        self,
        img,
        text_color=(255, 255, 255),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.8,
        thickness=2,
    ):
        alpha = self.alpha
        bg_rec = img[self.y : self.y + self.h, self.x : self.x + self.w]
        white_rect = np.ones(bg_rec.shape, dtype=np.uint8)
        white_rect[:] = self.color
        res = cv2.addWeighted(bg_rec, alpha, white_rect, 1 - alpha, 1.0)
        img[self.y : self.y + self.h, self.x : self.x + self.w] = res
        text_size = cv2.getTextSize(self.text, fontFace, fontScale, thickness)
        text_pos = (
            int(self.x + self.w / 2 - text_size[0][0] / 2),
            int(self.y + self.h / 2 + text_size[0][1] / 2),
        )
        cv2.putText(
            img, self.text, text_pos, fontFace, fontScale, text_color, thickness
        )

    def isOver(self, x, y):
        return (self.x + self.w > x > self.x) and (self.y + self.h > y > self.y)


class PoseOverlay:
    def __init__(self, overlay_img_path="board_drawing.png"):
        # Initialize Mediapipe Pose
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose()
        self.overlay_img = cv2.imread(overlay_img_path, cv2.IMREAD_UNCHANGED)

    def find_chest_area(self, img):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(imgRGB)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            h, w, _ = img.shape
            left_shoulder = (int(landmarks[11].x * w), int(landmarks[11].y * h))
            right_shoulder = (int(landmarks[12].x * w), int(landmarks[12].y * h))

            chest_center = (
                (left_shoulder[0] + right_shoulder[0]) // 2,
                (left_shoulder[1] + right_shoulder[1]) // 2,
            )
            chest_width = int(
                math.hypot(
                    right_shoulder[0] - left_shoulder[0],
                    right_shoulder[1] - left_shoulder[1],
                )
            )
            return chest_center, chest_width

        return None, None

    def overlay_on_chest(self, img, chest_center, chest_width):
        # Check if the overlay image has an alpha channel
        has_alpha = (
            self.overlay_img.shape[2] == 4 if self.overlay_img is not None else False
        )

        # Resize the overlay image to fit the chest width
        aspect_ratio = self.overlay_img.shape[1] / self.overlay_img.shape[0]
        overlay_height = int(chest_width / aspect_ratio)
        resized_overlay = cv2.resize(self.overlay_img, (chest_width, overlay_height))

        # Calculate top-left corner for placing the overlay on the chest
        x_offset = chest_center[0] - chest_width // 2
        # Adjust y_offset to place it lower for a T-shirt design
        y_offset = chest_center[1] + int(
            overlay_height * 0.3
        )  # Move down further (30% of overlay height)

        # Ensure the overlay does not go out of bounds
        if (
            (y_offset < 0)
            or (y_offset + overlay_height > img.shape[0])
            or (x_offset < 0)
            or (x_offset + chest_width > img.shape[1])
        ):
            return  # Skip overlaying if it goes out of bounds

        # Blend the overlay image onto the frame
        if has_alpha:
            for c in range(3):  # Loop over BGR channels only
                img[
                    y_offset : y_offset + overlay_height,
                    x_offset : x_offset + chest_width,
                    c,
                ] = resized_overlay[:, :, c] * (resized_overlay[:, :, 3] / 255.0) + img[
                    y_offset : y_offset + overlay_height,
                    x_offset : x_offset + chest_width,
                    c,
                ] * (
                    1.0 - resized_overlay[:, :, 3] / 255.0
                )
        else:
            img[
                y_offset : y_offset + overlay_height, x_offset : x_offset + chest_width
            ] = resized_overlay

    def run_pose_detection(self):
        cap = cv2.VideoCapture(0)

        while True:
            success, img = cap.read()
            if not success:
                break

            chest_center, chest_width = self.find_chest_area(img)
            if chest_center and chest_width:
                self.overlay_on_chest(img, chest_center, chest_width)

            cv2.imshow("Pose Detection with Overlay", img)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()


detector = HandTracker(detectionCon=0.5)

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

canvas = np.zeros((720, 1280, 3), np.uint8)
px, py = 0, 0
color = (255, 0, 0)
brushSize = 5
eraserSize = 20
colorsBtn = ColorRect(200, 0, 100, 100, (120, 255, 0), "Colors")

# Define color options and buttons
colors = [
    ColorRect(
        300,
        0,
        100,
        100,
        (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)),
    ),
    ColorRect(400, 0, 100, 100, (0, 0, 255)),
    ColorRect(500, 0, 100, 100, (255, 0, 0)),
    ColorRect(600, 0, 100, 100, (0, 255, 0)),
    ColorRect(700, 0, 100, 100, (0, 255, 255)),
    ColorRect(800, 0, 100, 100, (0, 0, 0), "Eraser"),
]
clear = ColorRect(900, 0, 100, 100, (100, 100, 100), "Clear")
pens = [
    ColorRect(1100, 50 + 100 * i, 100, 100, (50, 50, 50), str(size))
    for i, size in enumerate(range(5, 25, 5))
]

penBtn = ColorRect(1100, 0, 100, 50, color, "Pen")
boardBtn = ColorRect(50, 0, 100, 100, (255, 255, 0), "Board")
whiteBoard = ColorRect(50, 120, 1020, 580, (255, 255, 255), alpha=0.6)
saveBtn = ColorRect(1100, 600, 100, 70, (70, 200, 70), "Save")

coolingCounter = 20
hideBoard = True
hideColors = True
hidePenSizes = True

while True:
    if coolingCounter:
        coolingCounter -= 1

    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (1280, 720))
    frame = cv2.flip(frame, 1)

    detector.findHands(frame)
    positions = detector.getPostion(frame, draw=False)
    upFingers = detector.getUpFingers(frame)

    if upFingers:
        x, y = positions[8][0], positions[8][1]
        if upFingers[1] and not whiteBoard.isOver(x, y):
            px, py = 0, 0

            if not hidePenSizes:
                for pen in pens:
                    if pen.isOver(x, y):
                        brushSize = int(pen.text)
                        pen.alpha = 0
                    else:
                        pen.alpha = 0.5

            if not hideColors:
                for cb in colors:
                    if cb.isOver(x, y):
                        color = cb.color
                        cb.alpha = 0
                    else:
                        cb.alpha = 0.5

                if clear.isOver(x, y):
                    clear.alpha = 0
                    canvas = np.zeros((720, 1280, 3), np.uint8)
                else:
                    clear.alpha = 0.5

            if colorsBtn.isOver(x, y) and not coolingCounter:
                coolingCounter = 10
                colorsBtn.alpha = 0
                hideColors = not hideColors
                colorsBtn.text = "Colours" if hideColors else "Hide"
            else:
                colorsBtn.alpha = 0.5

            if penBtn.isOver(x, y) and not coolingCounter:
                coolingCounter = 10
                penBtn.alpha = 0
                hidePenSizes = not hidePenSizes
                penBtn.text = "Pen" if hidePenSizes else "Hide"
            else:
                penBtn.alpha = 0.5

            if boardBtn.isOver(x, y) and not coolingCounter:
                coolingCounter = 10
                boardBtn.alpha = 0
                hideBoard = not hideBoard
                boardBtn.text = "Board" if hideBoard else "Hide"
            else:
                boardBtn.alpha = 0.5

            if saveBtn.isOver(x, y) and not coolingCounter:
                coolingCounter = 10
                saveBtn.alpha = 0
                cv2.imwrite("board_drawing.png", canvas)

                # Close the current OpenCV windows and release the camera
                cap.release()
                cv2.destroyAllWindows()

                # Open pose detection with chest overlay
                pose_overlay = PoseOverlay("board_drawing.png")
                pose_overlay.run_pose_detection()
                break
            else:
                saveBtn.alpha = 0.5

        elif upFingers[1] and not upFingers[2]:
            if whiteBoard.isOver(x, y) and not hideBoard:
                cv2.circle(frame, positions[8], brushSize, color, -1)
                if px == 0 and py == 0:
                    px, py = positions[8]
                if color == (0, 0, 0):
                    cv2.line(canvas, (px, py), positions[8], color, eraserSize)
                else:
                    cv2.line(canvas, (px, py), positions[8], color, brushSize)
                px, py = positions[8]

        else:
            px, py = 0, 0

    colorsBtn.drawRect(frame)
    boardBtn.drawRect(frame)
    penBtn.color = color
    penBtn.drawRect(frame)
    saveBtn.drawRect(frame)

    if not hideBoard:
        whiteBoard.drawRect(frame)
        canvasGray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, imgInv = cv2.threshold(canvasGray, 20, 255, cv2.THRESH_BINARY_INV)
        imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
        frame = cv2.bitwise_and(frame, imgInv)
        frame = cv2.bitwise_or(frame, canvas)

    if not hideColors:
        for c in colors:
            c.drawRect(frame)
        clear.drawRect(frame)

    if not hidePenSizes:
        for pen in pens:
            pen.drawRect(frame)

    cv2.imshow("video", frame)
    k = cv2.waitKey(1)
    if k == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
