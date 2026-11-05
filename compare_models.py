
from pathlib import Path
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# 1. Load dataset
DATA_PATH = Path(__file__).resolve().parent / "data" / "tickets_dataset.csv"

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


# 3. Define cross-validation
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# 4. Define models
models = {
    "Logistic Regression": Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 2)
        )),
        ("classifier", LogisticRegression(
            max_iter=1000
        ))
    ]),

    "Linear SVM": Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 2)
        )),
        ("classifier", LinearSVC(
            max_iter=5000
        ))
    ])
}


# 5. Compare models using Cross-Validation
comparison_results = []

for name, model in models.items():

    scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring="f1_macro"
    )

    mean_score = scores.mean()
    std_score = scores.std()

    comparison_results.append({
        "Model": name,
        "Mean Macro F1": mean_score,
        "Standard Deviation": std_score
    })

    print(f"\n--- {name} ---")

    for fold, score in enumerate(scores, start=1):
        print(f"Fold {fold}: Macro F1 = {score:.3f}")

    print(f"Mean Macro F1: {mean_score:.3f}")
    print(f"Standard Deviation: {std_score:.3f}")


# 6. Display comparison
comparison_df = pd.DataFrame(comparison_results)

comparison_df = comparison_df.sort_values(
    by="Mean Macro F1",
    ascending=False
)

print("\n--- MODEL COMPARISON ---")
print(comparison_df.to_string(index=False))

print("\n[SUCCESS] Model comparison completed!")


# 7. Final evaluation of the selected model

best_model = models["Linear SVM"]

# Train selected model on the complete training set
best_model.fit(X_train, y_train)


# Save trained model
MODEL_DIR = Path(__file__).resolve().parent / "models"

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MODEL_PATH = MODEL_DIR / "linear_svm_pipeline.joblib"

joblib.dump(best_model, MODEL_PATH)

print(f"\nModel saved successfully: {MODEL_PATH}")

# Predict unseen holdout tickets
y_pred = best_model.predict(X_test)

print("\n--- FINAL LINEAR SVM EVALUATION ---")

accuracy = accuracy_score(y_test, y_pred)

print(f"Test Accuracy: {accuracy:.3f}")

print("\n--- CLASSIFICATION REPORT ---")

class_names = sorted(y.unique())

print(classification_report(
    y_test,
    y_pred,
    labels=class_names,
    zero_division=0
))

# Confusion Matrix
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

print("\n[SUCCESS] Linear SVM final evaluation completed!")
