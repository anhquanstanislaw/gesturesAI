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

class TakeData:
    def __init__(self, sampling_rate: float = 0.1, max_hands: int = 2, min_detection_confidence: float = 0.5, min_tracking_confidence:float = 0.5):
        self.sampling_rate = sampling_rate # 0.1 - 10 photos per second
        self._init_model(max_hands, min_detection_confidence, min_tracking_confidence)
        self._init_camera()

        # Const Paths
        self.data_path = Path("stored_data")
        self.clenched_fist_path = self.data_path / "clenched_fist.jsonl"
        self.normal_hand_path = self.data_path / "normal_hand.jsonl"
        self.middle_pinch_path = self.data_path / "middle_pinch.jsonl"

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
        print("\n" * 5)
        print("Type 1 to start recording normal hand")
        print("Type 2 to start recording clenched fist")
        print("Type 3 to start recording middle pinch")
        print("Type 4 to stop recording and save results")
        print("If you've done eny error type 5 to stop recording and abandon records")
        print("Type 5 while you're not recording to abandon any current records")

        recording = 0
        current_records = []
        self.saved_records_clenched_fist = []
        self.saved_records_normal_hand = []
        self.saved_records_middle_pinch = []
        last_captured_time = 0
        
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

            cv2.imshow('Hand Tracking - GesturesAI', frame)
            left_hand = self.get_left_hand(results)
            right_hand = self.get_right_hand(results)

            if recording and right_hand:
                current_time = time.time()
                if current_time - last_captured_time > self.sampling_rate:
                    normalized = normalize_altogether(right_hand, frame)
                    #print(normalized)
                    current_records.append(normalized)
                    last_captured_time = current_time
                    
            
            key = cv2.waitKey(1)

            if key & 0xFF == ord('1') and not recording:
                recording = 1
                last_captured_time = 0

                print("Started recording normal hand")    
            
            
            if key & 0xFF == ord('2') and not recording:
                recording = 2
                last_captured_time = 0

                print("Started recording clenched fist")

            if key & 0xFF == ord('3') and not recording:
                recording = 3
                last_captured_time = 0

                print("Started recording middle pinch")



            if key & 0xFF == ord('4') and recording:
                print("Saving changes")
                if recording == 1:
                    self.saved_records_normal_hand.extend(deepcopy(current_records))
                if recording == 2:
                    self.saved_records_clenched_fist.extend(deepcopy(current_records))
                if recording == 3:
                    self.saved_records_middle_pinch.extend(deepcopy(current_records))

                current_records = []
                recording = 0
            
            if key & 0xFF == ord('5'):
                if recording:
                    print("Abandoning current recording")
                    current_records = []
                    recording = 0
                else:
                    print("Abandoning all recordings from this session")

                    self.saved_records_clenched_fist = []
                    self.saved_records_normal_hand = []
                    self.saved_records_middle_pinch = []

            if key & 0xFF == ord('q'):
                break   

    
    def save_records_to_file(self):
        if self.saved_records_clenched_fist:
            with open(self.clenched_fist_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(self.saved_records_clenched_fist) + "\n")

        if self.saved_records_normal_hand:
            with open(self.normal_hand_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(self.saved_records_normal_hand) + "\n")

        if self.saved_records_middle_pinch:
            with open(self.middle_pinch_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(self.saved_records_middle_pinch) + "\n")

    def cleanup(self):
        self.cap.release()
        cv2.destroyAllWindows()
        self.hands.close()
        self.save_records_to_file()

if __name__ == "__main__":
    TD = TakeData()
    TD.run()
    TD.cleanup()