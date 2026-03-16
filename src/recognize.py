import sys
import cv2
import mediapipe as mp
import platform
import numpy as np
from pathlib import Path
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
import pickle
import json

class GestureRecognizer:
    def __init__(self, model_path='gesture_model.h5'):
        self.model_path = Path(model_path)
        # Derive scaler path from model path
        model_stem = self.model_path.stem
        self.scaler_path = self.model_path.parent / f'{model_stem}_scaler.pkl'
        
        self._init_model()
        self._init_camera()
        self._load_ml_model()
        
    def _init_model(self):
        """Initialize MediaPipe hand detection"""
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        
        print("Loading MediaPipe hands model...")
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
    
    def _init_camera(self):
        """Initialize camera"""
        print("Loading camera...")
        current_os = platform.system()
        if current_os == "Darwin":
            camera_backend = cv2.CAP_AVFOUNDATION
        else:
            camera_backend = cv2.CAP_DSHOW
        
        self.cap = cv2.VideoCapture(0, camera_backend)
        if not self.cap.isOpened():
            print("Error: Could not open camera.")
            sys.exit(1)
    
    def _load_ml_model(self):
        """Load trained gesture recognition model and scaler"""
        if not self.model_path.exists():
            print(f"Error: Model file '{self.model_path}' not found.")
            print("Please train a model first using train_model.py")
            sys.exit(1)
        
        print(f"Loading ML model from {self.model_path}...")
        self.model = keras.models.load_model(self.model_path)
        
        # Try to load scaler
        if self.scaler_path.exists():
            print(f"Loading scaler from {self.scaler_path}...")
            with open(self.scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
        else:
            print("Warning: Scaler not found, using no scaling")
            self.scaler = None
    
    def preprocess_frame(self, landmarks):
        """Convert hand landmarks to model input
        
        Input:  [[x1,y1], [x2,y2], ..., [x21,y21]]  (21 hand landmarks)
        Output: [x1, y1, x2, y2, ..., x21, y21]     (42 features)
        
        Feature indices:
        [0,1]=landmark0(wrist)  [2,3]=landmark1(thumb)  ...  [40,41]=landmark20(pinky)
        """
        flattened = np.array(landmarks).flatten()
        
        # Scale if scaler is available
        if self.scaler is not None:
            flattened = self.scaler.transform([flattened])[0]
        else:
            flattened = flattened.reshape(1, -1)[0]
        
        return flattened
    
    def predict_gesture(self, landmarks):
        """Predict gesture from hand landmarks"""
        processed = self.preprocess_frame(landmarks)
        prediction = self.model.predict(processed.reshape(1, -1), verbose=0)[0][0]
        confidence = prediction if prediction > 0.5 else 1 - prediction
        gesture = "Clenched Fist" if prediction > 0.5 else "Open Hand"
        
        return gesture, confidence
    
    def run(self):
        """Run real-time gesture recognition"""
        print("\nGesture Recognition - Live Prediction")
        print("Press 'q' to quit\n")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            h, w, _ = frame.shape
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw landmarks
                    self.mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS
                    )
                    
                    # Convert to normalized coordinates
                    landmarks_normalized = []
                    for lm in hand_landmarks.landmark:
                        landmarks_normalized.append([lm.x, lm.y])
                    
                    # Predict
                    gesture, confidence = self.predict_gesture(landmarks_normalized)
                    
                    # Display prediction
                    color = (0, 255, 0) if gesture == "Clenched Fist" else (255, 0, 0)
                    text = f"{gesture}: {confidence:.2%}"
                    
                    cv2.putText(
                        frame,
                        text,
                        (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2,
                        color,
                        2
                    )
            else:
                cv2.putText(
                    frame,
                    "No hand detected",
                    (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (128, 128, 128),
                    2
                )
            
            cv2.imshow('Gesture Recognition', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    def cleanup(self):
        """Cleanup resources"""
        self.cap.release()
        cv2.destroyAllWindows()
        self.hands.close()

if __name__ == "__main__":
    gr = GestureRecognizer()
    gr.run()
    gr.cleanup()
