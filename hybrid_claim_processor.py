import json
import os
import time
from pathlib import Path

import joblib
import pandas as pd
import requests

from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Get API key securely
API_KEY = os.getenv("GROQ_API_KEY")

if not API_KEY or API_KEY == "YOUR_API_KEY":
    raise RuntimeError("Please configure GROQ_API_KEY in your .env file")
API_URL = "https://api.groq.com/openai/v1/chat/completions"

# Load trained Machine Learning model

MODEL_PATH = (
    Path(__file__).resolve().parent
    / "models"
    / "calibrated_linear_svm.joblib"
)

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"ML model not found: {MODEL_PATH}"
    )

ml_model = joblib.load(MODEL_PATH)

print("[SUCCESS] ML classification model loaded!")


# Predict ticket category using Linear SVM


CONFIDENCE_THRESHOLD = 0.60

def predict_category(text):

    # Get calibrated probabilities
    probabilities = ml_model.predict_proba([text])[0]

    # Find the most likely category
    best_index = probabilities.argmax()

    category = ml_model.classes_[best_index]
    confidence = float(probabilities[best_index])

    return category, confidence


def analyze_ticket(text):

    # 1. Predict category using Machine Learning
    category, confidence = predict_category(text)

    print(
        f"ML predicted category: {category} "
        f"(confidence: {confidence:.3f})"
    )
    
    # Decide whether human review is required
    routing_decision = (
        "human_review"
        if confidence < CONFIDENCE_THRESHOLD
        else "auto_candidate"
    )

    print(f"Routing decision: {routing_decision}")

    # Stop automated processing when confidence is low
    if routing_decision == "human_review":
        print("LLM skipped: manual review required.")

        return {
            "kategorie": category,
            "confidence": confidence,
            "routing_decision": "human_review",
            "dringlichkeit": None,
            "zusammenfassung": None,
            "processing_status": "human_review"
        }

    # 2. Prepare API headers
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    # 3. Prepare LLM prompt
    prompt = f"""
    Analysiere die folgende Kundenanfrage.

    Bestimme ausschließlich:
    - dringlichkeit (hoch, mittel, niedrig)
    - zusammenfassung (kurzer Satz auf Deutsch)

    Gib ein JSON-Objekt mit genau diesen beiden Feldern zurück.

    Die Kategorie wird bereits durch ein separates
    Machine-Learning-Modell bestimmt.
    Bestimme deshalb keine Kategorie.

    Kundenanfrage:
    "{text}"
    """

    # 4. Prepare API payload
    payload = {
        "model": "openai/gpt-oss-120b",
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {"type": "json_object"},
        "temperature": 0.1,
    }

    # 5. Retry configuration
    max_attempts = 3

    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.post(
                API_URL,
                headers=headers,
                json=payload,
                timeout=30
            )

            response.raise_for_status()

            data = response.json()
            content = data["choices"][0]["message"]["content"]

            result = json.loads(content)

            # 6. Validate LLM response
            if not isinstance(result, dict):
                print("Invalid response: expected JSON object.")
                return {}

            allowed_urgencies = {
                "hoch",
                "mittel",
                "niedrig"
            }

            urgency = result.get("dringlichkeit")
            summary = result.get("zusammenfassung")

            if not isinstance(urgency, str) or urgency not in allowed_urgencies:
                print("Invalid urgency.")
                return {}

            if not isinstance(summary, str) or not summary.strip():
                print("Invalid summary.")
                return {}

            # 7. Combine ML and LLM results
            combined_result = {
                "kategorie": category,
                "confidence": confidence,
                "routing_decision": "auto_candidate",
                "dringlichkeit": urgency,
                "zusammenfassung": summary,
                "processing_status": "success"
            }

            return combined_result

        except requests.exceptions.Timeout:
            print(f"Timeout! Attempt {attempt}/{max_attempts}")

        except requests.exceptions.HTTPError as error:
            status = error.response.status_code

            if status == 429 or status >= 500:
                print(f"Temporary API error: {status}")
            else:
                print(f"HTTP error: {status}")
                return {}

        except requests.exceptions.RequestException as error:
            print(f"Connection error: {error}")

        except (KeyError, ValueError) as error:
            print(f"Invalid API response: {error}")
            return {}

        if attempt < max_attempts:
            wait_seconds = 2 ** attempt
            print(f"Retrying in {wait_seconds} seconds...")
            time.sleep(wait_seconds)

    print(f"Ticket failed after {max_attempts} attempts.")
    return {}


tickets = [
    {
        "id": "T-1",
        "text": "Ich warte schon seit 3 Wochen auf die Erstattung meiner Rechnung.",
    },
    {
        "id": "T-2",
        "text": "Bitte meine neue Adresse eintragen: Hauptstr. 10, Berlin.",
    },
    {
        "id": "T-3",
        "text": "Ich brauche Informationen zu den Leistungen während der Schwangerschaft.",
    },
]

if __name__ == "__main__":
    out = []
    for item in tickets:
        print(f"Checke Ticket {item['id']}...")
        res = analyze_ticket(item["text"])

        out.append(
            {
                "id": item["id"],
                "original": item["text"],
                "category": res.get("kategorie"),
                "confidence": res.get("confidence"),
                "routing_decision": res.get("routing_decision"),
                "urgency": res.get("dringlichkeit"),
                "summary": res.get("zusammenfassung"),
                "processing_status": res.get("processing_status", "failed"),
            }
        )

    
    df = pd.DataFrame(out)

    # Processing statistics
    success_count = (df["processing_status"] == "success").sum()
    failed_count = (df["processing_status"] == "failed").sum()
    human_review_count = (
        df["processing_status"] == "human_review"
    ).sum()

    print("\n--- PROCESSING SUMMARY ---")
    print(f"Total tickets: {len(df)}")
    print(f"Successfully processed: {success_count}")
    print(f"Failed: {failed_count}")
    print(f"Human review required: {human_review_count}")

    df.to_excel("hybrid_processed_tickets.xlsx", index=False)
    df.to_csv("hybrid_processed_tickets.csv", index=False, encoding="utf-8-sig")

    print("\n[SUCCESS] Excel- und CSV-Dateien wurden erfolgreich erstellt!")