import sys
import cv2
import mediapipe as mp
import platform
from copy import deepcopy
import time
from pathlib import Path
import json
from normalize import normalize_altogether
from mediapipe.framework.formats.landmark_pb2 import NormalizedLandmarkList
from typing import NamedTuple
import recognition
import pyautogui

class Record:
    def __init__(self, model_folder: str, max_hands: int = 2, min_detection_confidence: float = 0.5, min_tracking_confidence:float = 0.5):
        self._init_model(max_hands, min_detection_confidence, min_tracking_confidence)
        self._init_camera()
        self.detector = recognition.Recognition(model_folder)
        self.screen_w, self.screen_h = pyautogui.size()
        self.smooth_x = 0
        self.smooth_y = 0
        self.alpha = 0.6 # cursor smoothing coefficient
        self.margin = 0.2 # shrinks active camera/sensitivity lower = larger active area
        self.safe_margin = 5 # pixel buffor 

    def _init_model(self, max_hands: int, min_detection_confidence: float, min_tracking_confidence: float):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        
        print("Loading mediapipe model...")
        
        try:
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            print("Successfully loaded model")
        
        except Exception as e:
            print(f"Error while loading model {e}")
            sys.exit(1)

    def _init_camera(self):
        print("Loading camera...")
        current_os = platform.system()
        if current_os == "Darwin":
            # MAC OS
            camera_backend = cv2.CAP_AVFOUNDATION
        else:
            # WINDOWS
            camera_backend = cv2.CAP_DSHOW
        
        self.cap = cv2.VideoCapture(0, camera_backend)
        if not self.cap.isOpened():
            print("Error: Could not open camera.")
            sys.exit(1)

    def get_left_hand(self, curr: NamedTuple) -> NormalizedLandmarkList | None:
        if not curr.multi_hand_landmarks:
            return None
        
        for i, landmarks in enumerate(curr.multi_hand_landmarks):
            label = curr.multi_handedness[i].classification[0].label
            if label == "Right":
                return landmarks
        
        return None

    def get_right_hand(self, curr: NamedTuple) -> NormalizedLandmarkList | None:
        if not curr.multi_hand_landmarks:
            return None
        
        for i, landmarks in enumerate(curr.multi_hand_landmarks):
            label = curr.multi_handedness[i].classification[0].label
            if label == "Left":
                return landmarks
        
        return None
    
    def scale_point(self, raw_x: float, raw_y: float):
        scaled_x = (raw_x - self.margin) / (1 - self.margin * 2)
        scaled_y = (raw_y - self.margin) / (1 - self.margin * 2)
        
        scaled_x = max(0, min(1, scaled_x))
        scaled_y = max(0, min(1, scaled_y))

        target_x = self.screen_w - scaled_x * self.screen_w
        target_y = scaled_y * self.screen_h
        target_x = max(self.safe_margin, min(self.screen_w - self.safe_margin, target_x))
        target_y = max(self.safe_margin, min(self.screen_h - self.safe_margin, target_y))

        self.smooth_x = (self.smooth_x * self.alpha) + (target_x * (1 - self.alpha))
        self.smooth_y = (self.smooth_y * self.alpha) + (target_y * (1 - self.alpha))

    def run(self):
        is_pressed = False
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            results = self.hands.process(rgb_frame)
            if results.multi_hand_landmarks:    
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame, 
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS
                    )
            left_hand = self.get_left_hand(results)
            right_hand = self.get_right_hand(results)

            if right_hand:
                prediction = self.detector.predict(right_hand, frame)
                names = ["Normal", "Clenched", "Pinch"]
                for i, p in enumerate(prediction):
                    cv2.putText(frame, f"{names[i]}: {p:.2f}", (50, 50 + i * 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                raw_x, raw_y = [right_hand.landmark[5].x, right_hand.landmark[5].y]
                self.scale_point(raw_x, raw_y)
                if prediction[2] > 0.9 and prediction[2] > prediction[0] and prediction[2] > prediction[1]:
                    if not is_pressed:
                        is_pressed = True
                        pyautogui.mouseDown()
                    else:
                        pyautogui.moveTo(self.smooth_x, self.smooth_y, _pause=False)
                else:
                    if is_pressed:
                        pyautogui.mouseUp()
                        is_pressed = False
                    
                    if prediction[1] > 0.9 and prediction[1] > prediction[2] and prediction[1] > prediction[0]:
                        pyautogui.rightClick()
                    else:
                        pyautogui.moveTo(self.smooth_x, self.smooth_y, _pause=False)

            key = cv2.waitKey(1)
            cv2.imshow('Hand Tracking - GesturesAI', frame)
            
            if key & 0xFF == ord('q'):
                break   

        self.cap.release()
        cv2.destroyAllWindows()
        self.hands.close()
