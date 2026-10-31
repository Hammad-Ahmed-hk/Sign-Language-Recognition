"""Step 2: teach the computer using the data you collected.

Usage: python train.py
"""
import csv
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# 1. Load the CSV made by collect_data.py
labels, rows = [], []
with open("data.csv") as f:
    reader = csv.reader(f)
    next(reader)  # skip header
    for r in reader:
        labels.append(r[0])
        rows.append([float(x) for x in r[1:]])

X = np.array(rows)
y = np.array(labels)
print(f"Loaded {len(X)} samples, signs: {sorted(set(y))}")

# 2. Keep 20% aside to check the model on examples it has never seen
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Train. A random forest = many small decision trees that vote.
model = RandomForestClassifier(n_estimators=150, random_state=42)
model.fit(X_train, y_train)

# 4. Check how well it learned
acc = accuracy_score(y_test, model.predict(X_test))
print(f"Accuracy on unseen samples: {acc * 100:.1f}%")

# 5. Save for the live recognizer
joblib.dump(model, "model.pkl")
print("Saved model.pkl")
