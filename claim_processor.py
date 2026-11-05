
import json
import os
import time

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


def analyze_ticket(text):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    prompt = f"""
    Analysiere diesen Text und gib mir ein JSON-Objekt zurück mit genau diesen Feldern:
    - kategorie (Abrechnung, Stammdaten, Leistung, Beschwerde, Sonstiges)
    - dringlichkeit (hoch, mittel, niedrig)
    - zusammenfassung (kurzer Satz auf Deutsch)

    Text: "{text}"
    """

    payload = {
        "model": "openai/gpt-oss-120b",
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {"type": "json_object"},
        "temperature": 0.1,
    }

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

            allowed_categories = {
                "Abrechnung",
                "Stammdaten",
                "Leistung",
                "Beschwerde",
                "Sonstiges"
            }

            allowed_urgencies = {
                "hoch",
                "mittel",
                "niedrig"
            }

            if not isinstance(result, dict):
                print("Invalid response: expected JSON object.")
                return {}

            category = result.get("kategorie")
            urgency = result.get("dringlichkeit")
            summary = result.get("zusammenfassung")

            if not isinstance(category, str) or category not in allowed_categories:
                print("Invalid category.")
                return {}

            if not isinstance(urgency, str) or urgency not in allowed_urgencies:
                print("Invalid urgency.")
                return {}

            if not isinstance(summary, str) or not summary.strip():
                print("Invalid summary.")
                return {}

            return result

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
                "urgency": res.get("dringlichkeit"),
                "summary": res.get("zusammenfassung"),
                "processing_status": "success" if res else "failed",
            }
        )

    df = pd.DataFrame(out)

    success_count = (df["processing_status"] == "success").sum()
    failed_count = (df["processing_status"] == "failed").sum()

    print("\n--- PROCESSING SUMMARY ---")
    print(f"Total tickets: {len(df)}")
    print(f"Successfully processed: {success_count}")
    print(f"Failed: {failed_count}")

    df.to_excel("processed_tickets.xlsx", index=False)
    df.to_csv("processed_tickets.csv", index=False, encoding="utf-8-sig")

    print("\n[SUCCESS] Excel- und CSV-Dateien wurden erfolgreich erstellt!")
