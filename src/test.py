import cv2
import mediapipe as mp
import pyautogui
import sys

# Wyłączenie bezpiecznika pyautogui (opcjonalnie, ale ułatwia testy)
pyautogui.FAILSAFE = False

# Dynamiczne pobieranie rozdzielczości ekranu
screen_w, screen_h = pyautogui.size()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Błąd: Nie można otworzyć kamery.")
    sys.exit()

print(f"Rozdzielczość ekranu: {screen_w}x{screen_h}")
print("Naciśnij 'q' w oknie wideo, aby zamknąć.")

while cap.isOpened():
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    h, w, c = img.shape
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Rysowanie kropek na dłoni (jeśli je widzisz, MediaPipe działa)
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Pobieranie punktu nr 8 (palec wskazujący)
            index_finger = hand_landmarks.landmark[8]
            
            # Przeliczanie na piksele
            target_x = int(index_finger.x * screen_w)
            target_y = int(index_finger.y * screen_h)

            # Diagnostyka: wypisz pozycję w terminalu
            print(f"Ruch do: {target_x}, {target_y}", end='\r')

            # Próba ruchu
            try:
                pyautogui.moveTo(target_x, target_y, _pause=False)
            except Exception as e:
                print(f"Błąd pyautogui: {e}")

    cv2.imshow("Test Sterowania", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()