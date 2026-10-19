import json
import pandas as pd
import requests

API_KEY = "YOUR_API_KEY"
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

    response = requests.post(API_URL, headers=headers, json=payload)

    if response.status_code == 200:
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        return json.loads(content)
    else:
        print("Fehler:", response.status_code, response.text)
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
            }
        )

    df = pd.DataFrame(out)
    df.to_excel("processed_tickets.xlsx", index=False)
    df.to_csv("processed_tickets.csv", index=False, encoding="utf-8-sig")

    print("\n[SUCCESS] Excel- und CSV-Dateien wurden erfolgreich erstellt!")
