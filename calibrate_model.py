
from pathlib import Path
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, accuracy_score


# 1. Load dataset

DATA_PATH = (
    Path(__file__).resolve().parent
    / "data"
    / "tickets_dataset.csv"
)

df = pd.read_csv(DATA_PATH)

X = df["ticket_text"]
y = df["category"]


# 2. Split dataset

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("--- DATA SPLIT ---")
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")


# 3. Create base ML pipeline

base_model = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2)
    )),
    ("classifier", LinearSVC(
        max_iter=5000
    ))
])


# 4. Apply probability calibration

calibrated_model = CalibratedClassifierCV(
    estimator=base_model,
    method="sigmoid",
    cv=5
)

calibrated_model.fit(X_train, y_train)

print("\n[SUCCESS] Calibrated model training completed!")


# 5. Make predictions and get probabilities

predictions = calibrated_model.predict(X_test)

probabilities = calibrated_model.predict_proba(X_test)

confidence_scores = probabilities.max(axis=1)


# 6. Define confidence threshold

CONFIDENCE_THRESHOLD = 0.60


# 7. Create prediction results

results = pd.DataFrame({
    "ticket_text": X_test.values,
    "actual_category": y_test.values,
    "predicted_category": predictions,
    "confidence": confidence_scores.round(3),
    "routing_decision": [
        "human_review"
        if score < CONFIDENCE_THRESHOLD
        else "auto_candidate"
        for score in confidence_scores
    ]
})


print("\n--- CONFIDENCE ANALYSIS ---")
print(results.to_string(index=False))


# 8. Evaluate model

accuracy = accuracy_score(y_test, predictions)

print("\n--- MODEL EVALUATION ---")
print(f"Test Accuracy: {accuracy:.3f}")

print("\n--- CLASSIFICATION REPORT ---")

class_names = sorted(y.unique())

print(classification_report(
    y_test,
    predictions,
    labels=class_names,
    zero_division=0
))


# 9. Save calibrated model

MODEL_DIR = (
    Path(__file__).resolve().parent
    / "models"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MODEL_PATH = MODEL_DIR / "calibrated_linear_svm.joblib"

joblib.dump(calibrated_model, MODEL_PATH)

print(f"\n[SUCCESS] Model saved: {MODEL_PATH}")

print("\n[SUCCESS] Confidence calibration completed!")
