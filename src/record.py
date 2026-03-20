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

class Record:
    def __init__(self, model_folder: str, sampling_rate: float = 0.1, max_hands: int = 2, min_detection_confidence: float = 0.5, min_tracking_confidence:float = 0.5):
        self.sampling_rate = sampling_rate # 0.1 - 10 photos per second
        self._init_model(max_hands, min_detection_confidence, min_tracking_confidence)
        self._init_camera()
        self.detector = recognition.Recognition(model_folder)

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
    
    def run(self):
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
                cv2.putText(frame, str(prediction), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            key = cv2.waitKey(1)
            cv2.imshow('Hand Tracking - GesturesAI', frame)
            
            if key & 0xFF == ord('q'):
                break   

