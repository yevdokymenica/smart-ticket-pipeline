
from pathlib import Path
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix


# 1. Load dataset
DATA_PATH = Path(__file__).resolve().parent / "data" / "tickets_dataset.csv"

df = pd.read_csv(DATA_PATH)

X = df["ticket_text"]
y = df["category"]


# 2. Split data into training and testing sets
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


# 3. Create Machine Learning pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2)
    )),
    ("classifier", LogisticRegression(
        max_iter=1000
    ))
])

# 4. Cross-Validation

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

cv_scores = cross_val_score(
    model,
    X_train,
    y_train,
    cv=cv,
    scoring="f1_macro"
)

print("\n--- CROSS-VALIDATION RESULTS ---")

for fold, score in enumerate(cv_scores, start=1):
    print(f"Fold {fold}: Macro F1 = {score:.3f}")

print(f"Mean Macro F1: {cv_scores.mean():.3f}")
print(f"Standard Deviation: {cv_scores.std():.3f}")

# 5. Train the model
model.fit(X_train, y_train)

print("\n[SUCCESS] Model training completed!")


# 6. Make predictions
y_pred = model.predict(X_test)

# 7. Detailed prediction analysis

results = pd.DataFrame({
    "ticket_text": X_test.values,
    "actual_category": y_test.values,
    "predicted_category": y_pred
})

results["correct"] = (
    results["actual_category"] == results["predicted_category"]
)

print("\n--- PREDICTION DETAILS ---")
print(results.to_string(index=False))

errors = results[~results["correct"]]

print("\n--- MISCLASSIFIED TICKETS ---")

if errors.empty:
    print("No classification errors.")
else:
    print(errors.to_string(index=False))

# 8. Evaluate model performance
print("\n--- CLASSIFICATION REPORT ---")

class_names = sorted(y.unique())

print(classification_report(
    y_test,
    y_pred,
    labels=class_names,
    zero_division=0
))


# 9. Display confusion matrix
cm = confusion_matrix(
    y_test,
    y_pred,
    labels=class_names
)

cm_df = pd.DataFrame(
    cm,
    index=[f"Actual: {c}" for c in class_names],
    columns=[f"Predicted: {c}" for c in class_names]
)

print("\n--- CONFUSION MATRIX ---")
print(cm_df.to_string())

print("\n[SUCCESS] Model evaluation completed!")