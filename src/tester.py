import sys

def run_diagnostics():
    print("=== Starting Environment Diagnostics ===\n")

    # 1. Test OpenCV Installation
    try:
        import cv2
        print(f"✅ OpenCV installed successfully (Version: {cv2.__version__})")
    except ImportError:
        print("❌ OpenCV is NOT installed. Run: pip install opencv-python")
        sys.exit(1)

    # 2. Test MediaPipe Installation
    try:
        import mediapipe as mp
        print(f"✅ MediaPipe installed successfully (Version: {mp.__version__})")
    except ImportError:
        print("❌ MediaPipe is NOT installed. Run: pip install mediapipe")
        sys.exit(1)

    # 3. Test MediaPipe Model Initialization
    print("\n--- Testing MediaPipe Modules ---")
    try:
        # Initialize the Hands module briefly to ensure models download/load properly
        hands = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=1)
        print("✅ MediaPipe Hands module initialized successfully!")
        hands.close()
    except Exception as e:
        print(f"❌ Failed to initialize MediaPipe Hands: {e}")

    # 4. Test Webcam Access
    print("\n--- Testing Webcam Access ---")
    # 0 is usually the built-in or default USB webcam
    cap = cv2.VideoCapture(0) 
    
    if not cap.isOpened():
        print("❌ Could not access the webcam. Please check if it is plugged in, or if another app (like Zoom/Discord) is using it.")
    else:
        # Try to read a single frame
        success, frame = cap.read()
        if success:
            print(f"✅ Webcam accessed and captured a frame successfully! (Resolution: {frame.shape[1]}x{frame.shape[0]})")
        else:
            print("❌ Webcam was opened, but failed to read a frame.")
        
        # Release the camera so other apps can use it
        cap.release()

    print("\n======================================")
    print("🎉 Diagnostics Complete! If you saw all green checks, your environment is 100% ready!")

if __name__ == "__main__":
    run_diagnostics()