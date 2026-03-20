
# gesturesAI

gesturesAI is a gesture recognition system using computer vision and machine learning. It allows you to collect hand gesture data, train a model, and use your webcam to recognize gestures in real time.


## Main Functionality
- **Data Collection**: Use your webcam to record hand gesture data and save it for training. Recommended: Add your own gesture samples for best results.
- **Model Training**: Train a neural network model from the data(stored_data/)
- **Gesture Recognition**: Run the trained model(trained_models/) to recognize gestures live and control actions (e.g., mouse cursor) based on detected gestures.

### Supported Gestures
- **Clenched Fist**: cursor right click.
- **Normal Hand**: Default state, accounts for cursor steering
- **Middle Pinch**: cursor left click.

> It is recommended to collect and add your own gesture data before training the model for improved accuracy and personalization.

## How to Use
1. **Setup Environment**
	 - Install dependencies:
		 ```bash
		 conda env create -f environment.yml
		 conda activate gesture-controller
		 ```

2. **Run the Application**
	 - Start the main menu:
		 ```bash
		 python src/run.py
		 ```
	 - Choose from the menu:
		 - `1`: Train a gesture recognition model (optionally specify model name).
		 - `2`: Record and recognize gestures using your webcam (optionally specify model name).
		 - `3`: Collect new gesture data for training.
		 - `4`: Quit.

3. **Data and Models**
	 - Data is stored in the `stored_data/` folder.
	 - Trained models are saved in `trained_models/`.

## Requirements
- Python 3
- Conda environment (see `environment.yml`)
- Webcam
