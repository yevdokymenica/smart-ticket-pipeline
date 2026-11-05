
from pathlib import Path
import joblib

# 1. Define model path
MODEL_PATH = (
    Path(__file__).resolve().parent
    / "models"
    / "linear_svm_pipeline.joblib"
)

# 2. Check model availability
if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Trained model not found: {MODEL_PATH}"
    )

# 3. Load trained model
model = joblib.load(MODEL_PATH)

print("[SUCCESS] ML model loaded successfully!")

# 4. Get a new customer ticket
ticket_text = input("\nEnter customer ticket: ").strip()

if not ticket_text:
    raise ValueError("Ticket text cannot be empty.")

# 5. Predict category
prediction = model.predict([ticket_text])[0]

# 6. Display result
print("\n--- ML PREDICTION ---")
print(f"Original ticket: {ticket_text}")
print(f"Predicted category: {prediction}")
