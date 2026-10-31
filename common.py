"""Shared helpers: hand detection and turning a hand into numbers."""
import cv2
import numpy as np
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


def make_detector():
    # One hand is enough for static signs (letters, numbers, simple words).
    return mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )


def find_hand(detector, frame_bgr):
    """Return MediaPipe landmarks for the first hand, or None."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    result = detector.process(rgb)
    if result.multi_hand_landmarks:
        return result.multi_hand_landmarks[0]
    return None


def landmarks_to_features(hand):
    """Convert 21 (x, y) points into 42 numbers that don't depend on
    where the hand is in the picture or how close it is to the camera.

    1. Subtract the wrist position -> hand position no longer matters.
    2. Divide by the biggest value -> hand size no longer matters.
    """
    pts = np.array([[lm.x, lm.y] for lm in hand.landmark], dtype=np.float32)
    pts -= pts[0]                       # wrist becomes (0, 0)
    scale = np.abs(pts).max()
    if scale > 0:
        pts /= scale
    return pts.flatten()                # shape (42,)
