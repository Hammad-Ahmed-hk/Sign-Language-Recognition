# Simple Sign Language Recognition

Recognizes hand signs from your webcam in real time.

**How it works**
1. **MediaPipe** finds 21 points on your hand in each video frame.
2. Those points are turned into 42 numbers (position and size removed) -> `common.py`
3. A **Random Forest** (scikit-learn) learns which numbers belong to which sign.
4. Live, each frame is classified and the result is shown on screen.

## Setup (Python 3.9 - 3.12)
```
pip install -r requirements.txt
```
`mediapipe` is pinned below 0.10.30 because the code uses the legacy
`mp.solutions.hands` API, which newer releases removed.

## Use it
```
# 1. Record each sign (SPACE = start/stop recording, q = quit)
python collect_data.py A
python collect_data.py B
python collect_data.py C

# 2. Train
python train.py

# 3. Recognize live
python recognize.py
```

## Tips for good accuracy
- 200+ samples per sign; move your hand slightly while recording.
- Record in the lighting where you'll use it.
- Pick signs that look clearly different at first (e.g. A, B, L, V).
- Re-run `collect_data.py` for a sign that gets confused, then retrain.

## Files
| File | Purpose |
|---|---|
| `collect_data.py` | Record landmark samples for one sign into `data.csv` |
| `train.py` | Train the Random Forest and save `model.pkl` |
| `recognize.py` | Live webcam recognition |
| `common.py` | Hand detection and landmark-to-feature conversion |

`data.csv` and `model.pkl` are created by you and are git-ignored.

## Limits (good to understand)
- The accuracy printed by `train.py` comes from a random 80/20 split of
  frames. Frames from the same recording are almost identical, so that number
  is optimistic. Test the real thing live, or record a separate session for
  testing.
- Works for **static** signs (a held hand shape). Moving signs like J, Z or
  most words need a sequence model (e.g. LSTM over many frames).
- One hand only. Real sign languages also use face and body.
- Trained on you; other people may need to add samples.

## Ideas to extend
- Add a second hand (`max_num_hands=2`, 84 features).
- Collect frame sequences and train an LSTM for motion signs.
- Build a sentence by appending a sign when held for 1 second.
