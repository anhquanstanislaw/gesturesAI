import json
import sys
import numpy as np
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras 
from tensorflow.keras import layers  
import matplotlib.pyplot as plt

class GestureModel:
    def __init__(self, path_to_curr_model: str):
        self.data_path = Path("stored_data")
        self.clenched_fist_path = self.data_path / "clenched_fist.jsonl"
        self.normal_hand_path = self.data_path / "normal_hand.jsonl"
        self.model = None
        self.path_to_curr_model = Path("trained_models") / path_to_curr_model
        if not self.path_to_curr_model.exists():
            self.path_to_curr_model.mkdir()
        self.scaler = StandardScaler()
    
    def flatten_frame(self, frame):
        return np.array(frame).flatten()
    
    def load_data(self):
        X = []
        y = []
        
        # Load clenched fist data (label=1)
        print("Loading clenched fist data...")
        try:
            with open(self.clenched_fist_path, 'r') as f:
                for line in f:
                    if line.strip():
                        gesture_frames = json.loads(line)
                        for frame in gesture_frames:
                            X.append(self.flatten_frame(frame))
                            y.append(1)
        except FileNotFoundError:
            print(f"Warning: {self.clenched_fist_path} not found")
            sys.exit(1)
        
        print("Loading normal hand data...")
        try:
            with open(self.normal_hand_path, 'r') as f:
                for line in f:
                    if line.strip():
                        gesture_frames = json.loads(line)
                        for frame in gesture_frames:
                            X.append(self.flatten_frame(frame))
                            y.append(0)
        except FileNotFoundError:
            print(f"Warning: {self.normal_hand_path} not found")
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"Total samples: {len(X)}")
        print(f"Feature dimension: {X.shape[1]}")
        print(f"Class distribution - Clenched: {np.sum(y)}, Normal: {len(y) - np.sum(y)}")
        return X, y
    
    def build_model(self, input_dim):

        self.model = keras.Sequential([
            layers.Input(shape=(input_dim,)),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(32, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            layers.Dense(16, activation='relu'),
            layers.Dense(1, activation='sigmoid')  # Binary classification
        ])
        
        self.model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        print(self.model.summary())
    
    def train(self, X, y, epochs=50, batch_size=16):
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"\nTraining set size: {len(X_train)}")
        print(f"Test set size: {len(X_test)}")
        
        # Train with early stopping
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        history = self.model.fit(
            X_train_scaled, y_train,
            validation_data=(X_test_scaled, y_test),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop],
            verbose=1
        )
        
        # Evaluate
        test_loss, test_accuracy = self.model.evaluate(X_test_scaled, y_test, verbose=0)
        print(f"\nTest Accuracy: {test_accuracy:.4f}")
        print(f"Test Loss: {test_loss:.4f}")
        
        return history, X_test_scaled, y_test
    
    def save_model(self, model_name='gesture_model'):
        self.model.save(f"{self.path_to_curr_model}/{model_name}.h5")
        print(f"Model saved as '{model_name}.h5'")
        
        # Save scaler for use in prediction
        with open(f"{self.path_to_curr_model}/{model_name}_scaler.pkl", 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"Scaler saved as '{model_name}_scaler.pkl'")
    
    def run_full_pipeline(self):
        """Run complete training pipeline"""
        X, y = self.load_data()
        
        if len(X) == 0:
            print("No training data found!")
            return
        
        self.build_model(input_dim=X.shape[1])
        
        history, X_test, y_test = self.train(X, y, epochs=50, batch_size=16)
        
        self.save_model('gesture_model')

if __name__ == "__main__":
    print("give path to the model you want to train, or if not, it will be saved to default model ")
    path_to_model = input().strip()
    if not path_to_model:
        path_to_model = "model_defaulted"
    gm = GestureModel(path_to_model)
    gm.run_full_pipeline()
