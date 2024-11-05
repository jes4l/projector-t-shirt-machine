import cv2
import numpy as np
import random
import mediapipe as mp
import math
from handTracker import HandTracker
from datetime import datetime
import os


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
        white_rect = np.ones(bg_rec.shape, dtype=np.uint8) * 255

        if len(self.color) == 4:  # If color has an alpha channel
            white_rect = np.dstack(
                (white_rect, np.ones(bg_rec.shape[:2], dtype=np.uint8) * 255)
            )
            white_rect[:, :, :3] = self.color[:3]
            white_rect[:, :, 3] = self.color[3]
            res = cv2.addWeighted(bg_rec, alpha, white_rect[:, :, :3], 1 - alpha, 1.0)
        else:
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
    def __init__(self, overlay_img_path="tshirtsdesigns/board_drawing.png"):
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose()
        self.overlay_img = cv2.imread(overlay_img_path, cv2.IMREAD_UNCHANGED)
        self.startBtn = ColorRect(10, 10, 100, 50, (70, 70, 200), "Home")
        self.clicked = False
        self.overlay_img_path = overlay_img_path

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
        has_alpha = (
            self.overlay_img.shape[2] == 4 if self.overlay_img is not None else False
        )
        aspect_ratio = self.overlay_img.shape[1] / self.overlay_img.shape[0]
        overlay_height = int(chest_width / aspect_ratio)
        resized_overlay = cv2.resize(self.overlay_img, (chest_width, overlay_height))

        x_offset = chest_center[0] - chest_width // 2
        y_offset = chest_center[1] + int(overlay_height * 0.3)

        if (
            (y_offset < 0)
            or (y_offset + overlay_height > img.shape[0])
            or (x_offset < 0)
            or (x_offset + chest_width > img.shape[1])
        ):
            return

        if has_alpha:
            for c in range(3):
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

    def display_overlay_only(
        self,
        chest_center,
        chest_width,
        primary_window_size,
        overlay_movement_factor=0.5,
    ):
        primary_width, primary_height = primary_window_size
        aspect_ratio = self.overlay_img.shape[1] / self.overlay_img.shape[0]
        max_projection_size = int(0.6 * primary_width)
        min_projection_size = int(0.15 * primary_width)

        if chest_width <= 0.15 * primary_width:
            projection_width = max_projection_size
        elif chest_width >= 0.3 * primary_width:
            projection_width = min_projection_size
        else:
            scale_factor = (0.3 * primary_width - chest_width) / (0.15 * primary_width)
            projection_width = min_projection_size + int(
                scale_factor * (max_projection_size - min_projection_size)
            )

        projection_height = int(projection_width / aspect_ratio)
        resized_overlay = cv2.resize(
            self.overlay_img, (projection_width, projection_height)
        )
        overlay_window = np.zeros(
            (projection_height, projection_width, 3), dtype=np.uint8
        )
        has_alpha = self.overlay_img.shape[2] == 4
        if has_alpha:
            for c in range(3):
                overlay_window[:, :, c] = resized_overlay[:, :, c] * (
                    resized_overlay[:, :, 3] / 255.0
                )
        else:
            overlay_window = resized_overlay

        x_center, y_center = chest_center
        projection_x = int(x_center * overlay_movement_factor)
        projection_y = int(y_center * overlay_movement_factor)

        projection_x = min(max(projection_x, 0), primary_width - projection_width)
        projection_y = min(max(projection_y, 0), primary_height - projection_height)

        cv2.imshow("Overlay Projection", overlay_window)

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.startBtn.isOver(x, y):
                self.clicked = True

    def run_pose_detection(self):
        cap = cv2.VideoCapture(0)
        cv2.namedWindow("Pose Detection with Overlay")
        cv2.setMouseCallback("Pose Detection with Overlay", self.on_mouse)

        while True:
            success, img = cap.read()
            if not success:
                break

            chest_center, chest_width = self.find_chest_area(img)
            if chest_center and chest_width:
                self.overlay_on_chest(img, chest_center, chest_width)
                primary_window_size = (img.shape[1], img.shape[0])  # width, height
                self.display_overlay_only(
                    chest_center, chest_width, primary_window_size
                )

            self.startBtn.drawRect(img)

            cv2.imshow("Pose Detection with Overlay", img)

            if cv2.waitKey(1) & 0xFF == ord("q") or self.clicked:
                break

        cap.release()
        cv2.destroyAllWindows()

    def add_transparency(self, image_path):
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if img.shape[2] == 3:
            tmp = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, alpha = cv2.threshold(tmp, 0, 255, cv2.THRESH_BINARY)
            b, g, r = cv2.split(img)
            rgba = [b, g, r, alpha]
            img = cv2.merge(rgba, 4)
        cv2.imwrite(image_path, img)


detector = HandTracker(detectionCon=0.5)

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

canvas = np.zeros((720, 1280, 4), np.uint8)
canvas[:, :, 3] = 0  # Set alpha channel to 0 (fully transparent)

px, py = 0, 0
color = (255, 0, 0, 255)
brushSize = 5
eraserSize = 20
colorsBtn = ColorRect(200, 0, 100, 100, (120, 255, 0, 255), "Colours")

colors = [
    ColorRect(
        300,
        0,
        100,
        100,
        (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), 255),
    ),
    ColorRect(400, 0, 100, 100, (0, 0, 255, 255)),
    ColorRect(500, 0, 100, 100, (255, 0, 0, 255)),
    ColorRect(600, 0, 100, 100, (0, 255, 0, 255)),
    ColorRect(700, 0, 100, 100, (0, 255, 255, 255)),
    ColorRect(800, 0, 100, 100, (0, 0, 0, 255), "Eraser"),
]
clear = ColorRect(900, 0, 100, 100, (100, 100, 100, 255), "Clear")
pens = [
    ColorRect(1100, 50 + 100 * i, 100, 100, (50, 50, 50, 255), str(size))
    for i, size in enumerate(range(5, 25, 5))
]

penBtn = ColorRect(1100, 0, 100, 50, color, "Pen")
boardBtn = ColorRect(50, 0, 100, 100, (255, 255, 0, 255), "Board")
whiteBoard = ColorRect(
    50, 120, 1020, 580, (255, 255, 255, 153)
)  # Semi-transparent white
saveBtn = ColorRect(1100, 600, 100, 70, (70, 200, 70, 255), "Save")

coolingCounter = 20
hideBoard = True
hideColors = True
hidePenSizes = True

os.makedirs("tshirtsdesigns/gallery", exist_ok=True)

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
                    canvas = np.zeros((720, 1280, 4), np.uint8)
                    canvas[:, :, 3] = 0  # Reset alpha channel to 0 (fully transparent)
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

                # Save main board drawing image
                cv2.imwrite("tshirtsdesigns/board_drawing.png", canvas)

                # Save timestamped copy to gallery
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                gallery_path = f"tshirtsdesigns/gallery/board_gallery_{timestamp}.png"
                cv2.imwrite(gallery_path, canvas)

                # Close OpenCV windows and release camera
                cap.release()
                cv2.destroyAllWindows()

                # Add transparency to saved image
                pose_overlay = PoseOverlay("tshirtsdesigns/board_drawing.png")
                pose_overlay.add_transparency("tshirtsdesigns/board_drawing.png")

                # Open pose detection with chest overlay
                pose_overlay.run_pose_detection()
                break
            else:
                saveBtn.alpha = 0.5

        elif upFingers[1] and not upFingers[2]:
            if whiteBoard.isOver(x, y) and not hideBoard:
                cv2.circle(frame, positions[8], brushSize, color, -1)
                if px == 0 and py == 0:
                    px, py = positions[8]
                if color == (0, 0, 0, 255):
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
        canvasGray = cv2.cvtColor(canvas[:, :, :3], cv2.COLOR_BGR2GRAY)
        _, imgInv = cv2.threshold(canvasGray, 20, 255, cv2.THRESH_BINARY_INV)
        imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
        frame = cv2.bitwise_and(frame, imgInv)
        frame = cv2.bitwise_or(frame, canvas[:, :, :3])

    if not hideColors:
        for c in colors:
            c.drawRect(frame)
        clear.drawRect(frame)

    if not hidePenSizes:
        for pen in pens:
            pen.drawRect(frame)

    cv2.imshow("Virtual Canvas and Overlay", frame)
    k = cv2.waitKey(1)
    if k == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
