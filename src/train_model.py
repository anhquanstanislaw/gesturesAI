import json
import numpy as np
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

class GestureModel:
    def __init__(self):
        self.data_path = Path("stored_data")
        self.clenched_fist_path = self.data_path / "clenched_fist.jsonl"
        self.normal_hand_path = self.data_path / "normal_hand.jsonl"
        self.model = None
        self.scaler = StandardScaler()
    
    def flatten_frame(self, frame):
        """Convert frame to flat features
        
        Input frame: [[x1,y1], [x2,y2], ..., [x21,y21]]
        21 hand landmarks (0: wrist, 1-4: thumb, 5-8: index, 9-12: middle, 13-16: ring, 17-20: pinky)
        
        Output: [x1, y1, x2, y2, ..., x21, y21]
        42 features total (each landmark = 2 coords)
        """
        return np.array(frame).flatten()
        
    def load_data(self):
        """Load JSONL data and prepare features and labels
        
        Each sample becomes 42 features:
        [0,1]=wrist  [2,3]=thumb [4,5]=thumb [6,7]=thumb [8,9]=thumb
        [10,11]=index [12,13]=index ... [40,41]=pinky
        """
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
        
        # Load normal hand data (label=0)
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
        """Build neural network
        
        input_dim = 42 (21 landmarks × 2 coordinates each)
        """
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
        """Train the model"""
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
    
    def plot_training_history(self, history):
        """Plot training and validation metrics"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss
        ax1.plot(history.history['loss'], label='Training Loss')
        ax1.plot(history.history['val_loss'], label='Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Model Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy
        ax2.plot(history.history['accuracy'], label='Training Accuracy')
        ax2.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Model Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        print("\nTraining history saved as 'training_history.png'")
        plt.close()
    
    def save_model(self, model_name='gesture_model'):
        """Save the trained model and scaler"""
        self.model.save(f'{model_name}.h5')
        print(f"Model saved as '{model_name}.h5'")
        
        # Save scaler for use in prediction
        with open(f'{model_name}_scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"Scaler saved as '{model_name}_scaler.pkl'")
    
    def predict_frame(self, frame):
        """Predict on a single frame"""
        flattened = self.flatten_frame(frame).reshape(1, -1)
        flattened_scaled = self.scaler.transform(flattened)
        prediction = self.model.predict(flattened_scaled, verbose=0)[0][0]
        return prediction
    
    def run_full_pipeline(self):
        """Run complete training pipeline"""
        # Load data
        X, y = self.load_data()
        
        if len(X) == 0:
            print("No training data found!")
            return
        
        # Build model
        self.build_model(input_dim=X.shape[1])
        
        # Train
        history, X_test, y_test = self.train(X, y, epochs=50, batch_size=16)
        
        # Plot
        self.plot_training_history(history)
        
        # Save
        self.save_model('gesture_model')
        
        print("\n✓ Training complete!")

if __name__ == "__main__":
    gm = GestureModel()
    gm.run_full_pipeline()
