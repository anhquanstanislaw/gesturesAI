from mediapipe.framework.formats.landmark_pb2 import NormalizedLandmarkList
import numpy as np
import math
def normalize_to_wrist(thishand: NormalizedLandmarkList, frame: np.ndarray) -> list[tuple[int, int]]:
    h, w, _ = frame.shape
    newhand = []
    for i in range(21):  # 0-20 for all 21 landmarks
        newhand.append((int(thishand.landmark[i].x * w), (int(thishand.landmark[i].y * h))))

    x_norm = newhand[0][0]
    y_norm = newhand[0][1]
    for i in range(21):
        newhand[i] = (newhand[i][0] - x_norm, newhand[i][1] - y_norm)
    return newhand

def normalized_distance_to_wrist(hand_points: list[tuple[int, int]]) -> list[tuple[int, int]]: #deviding every distance by the distance from wrist to middle finger base (point 9)
    dist = []
    dist_wrist_to_9 = math.sqrt((hand_points[9][0] ** 2) + (hand_points[9][1] ** 2))
    if dist_wrist_to_9 < 1e-8: # Avoid division by zero
        return [(0.0, 0.0)] * 21
    for i in range(21):
        x = hand_points[i][0]
        y = hand_points[i][1]
        dist.append((x/ dist_wrist_to_9, y/ dist_wrist_to_9))
    return dist

def normalize_altogether(thishand: NormalizedLandmarkList, frame: np.ndarray) -> list[tuple[float, float]]:
    h, w, _ = frame.shape
    newhand = []
    for i in range(21):
        newhand.append(((thishand.landmark[i].x * w), (thishand.landmark[i].y * h)))

    x_norm = newhand[0][0]
    y_norm = newhand[0][1]

    for i in range(21):
        newhand[i] = (newhand[i][0] - x_norm, newhand[i][1] - y_norm)
    dist_wrist_to_9 = math.sqrt((newhand[9][0] ** 2) + (newhand[9][1] ** 2))
    if dist_wrist_to_9 < 1e-8: # Avoid division by zero
        return [(0.0, 0.0)] * 21
    for i in range(21):
        x = newhand[i][0]
        y = newhand[i][1]
        newhand[i] = (x / dist_wrist_to_9, y / dist_wrist_to_9)

    return newhand