import cv2
import mediapipe as mp
import sys
import platform
from normalize import normalize_to_wrist, normalized_distance_to_wrist, normalize_altogether
def start_hand_tracking():
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    
    current_os = platform.system()
    if current_os == "Darwin":
        # MAC OS
        camera_backend = cv2.CAP_AVFOUNDATION
    else:
        # WINDOWS
        camera_backend = cv2.CAP_DSHOW
        
    cap = cv2.VideoCapture(0, camera_backend)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    print("--- TRACKING STARTED ---")
    print("Press 'q' to exit.")

    frame_count = 0
    print_every = 30  # Print every 30 frames (~1 second at 30fps)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results = hands.process(rgb_frame)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:

                pixel_coords = normalize_to_wrist(hand_landmarks, frame)
                norm_distances = normalize_altogether(hand_landmarks, frame)
                # Print wrist and index fingertip coordinates (throttled)
                if frame_count % print_every == 0:
                    wrist = hand_landmarks.landmark[0]
                    index_tip = hand_landmarks.landmark[8]
                    print(f"index: {pixel_coords[8]}")
                    print(f"wrist: {pixel_coords[0]}")
                    print(f"distance: {norm_distances[8]}")
                    print("-" * 40)
                
                mp_drawing.draw_landmarks(
                    frame, 
                    hand_landmarks, 
                    mp_hands.HAND_CONNECTIONS
                )
        cv2.imshow('Hand Tracking - GesturesAI', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    hands.close()

if __name__ == "__main__":
    start_hand_tracking()