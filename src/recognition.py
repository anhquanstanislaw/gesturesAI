import json
import sys
import numpy as np
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras  # type: ignore
from tensorflow.keras import layers  # type: ignore
import matplotlib.pyplot as plt
import normalize
from mediapipe.framework.formats.landmark_pb2 import NormalizedLandmarkList

class Recognition:
    def __init__(self, model_folder: str="model_defaulted"):
        path_to_curr_model = Path("trained_models") / model_folder
        if not path_to_curr_model.exists():
           print("file does not exitst")
           sys.exit(1)
        self.path_to_model = path_to_curr_model / "gesture_model.h5"
        self.path_to_scaler = path_to_curr_model / "gesture_model_scaler.pkl"
        self.load_models()
    
    def load_models(self):
        self.model = keras.models.load_model(self.path_to_model)
        with open(self.path_to_scaler, 'rb') as f:
            self.scaler = pickle.load(f)


    def predict(self, thishand: NormalizedLandmarkList, frame: np.ndarray):
        normalized = normalize.normalize_altogether(thishand, frame)
        X = np.array(normalized).flatten().reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        prediction = self.model.predict(X_scaled, verbose=0)[0]
        return prediction
        
