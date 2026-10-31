"""Step 1: record examples of each sign.

Usage:   python collect_data.py A
Then hold up the sign 'A' and:
    SPACE  -> start / stop recording
    q      -> quit
Repeat for every sign (e.g. A, B, C, HELLO ...). Aim for 200+ samples each,
moving your hand a little (closer, farther, slightly rotated) while recording.
"""
import csv
import os
import sys
import cv2
from common import make_detector, find_hand, landmarks_to_features, mp_hands, mp_draw

DATA_FILE = "data.csv"

if len(sys.argv) < 2:
    sys.exit("Usage: python collect_data.py <LABEL>   e.g. python collect_data.py A")
label = sys.argv[1]

detector = make_detector()
cap = cv2.VideoCapture(0)
recording = False
count = 0

new_file = not os.path.exists(DATA_FILE)
f = open(DATA_FILE, "a", newline="")
writer = csv.writer(f)
if new_file:
    writer.writerow(["label"] + [f"f{i}" for i in range(42)])

while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)  # mirror view feels natural
    hand = find_hand(detector, frame)

    if hand:
        mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
        if recording:
            writer.writerow([label] + list(landmarks_to_features(hand)))
            count += 1

    status = "RECORDING" if recording else "paused"
    cv2.putText(frame, f"Label: {label}  [{status}]  saved: {count}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow("Collect data", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord(" "):
        recording = not recording
    elif key == ord("q"):
        break

f.close()
cap.release()
cv2.destroyAllWindows()
print(f"Saved {count} samples for '{label}' to {DATA_FILE}")
