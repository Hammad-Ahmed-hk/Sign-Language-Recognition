"""Step 3: recognize signs live from your webcam.

Usage: python recognize.py      (press q to quit)
"""
from collections import Counter, deque
import cv2
import joblib
import numpy as np
from common import make_detector, find_hand, landmarks_to_features, mp_hands, mp_draw

model = joblib.load("model.pkl")
detector = make_detector()
cap = cv2.VideoCapture(0)

# Keep the last 10 predictions and show the most common one.
# This stops the text from flickering between signs.
recent = deque(maxlen=10)

while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)
    hand = find_hand(detector, frame)

    text = "No hand"
    if hand:
        mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
        features = landmarks_to_features(hand).reshape(1, -1)
        probs = model.predict_proba(features)[0]
        best = int(np.argmax(probs))
        # Only accept confident guesses
        if probs[best] > 0.6:
            recent.append(model.classes_[best])
        smoothed = Counter(recent).most_common(1)[0][0] if recent else "?"
        text = f"{smoothed} ({probs[best] * 100:.0f}%)"
    else:
        recent.clear()

    cv2.putText(frame, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                1.4, (0, 255, 0), 3)
    cv2.imshow("Sign language recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
