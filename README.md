# Smart Ticket & Claims Automation Pipeline

Ein kleiner Python-Prototyp zur automatisierten Analyse und Strukturierung von Kundenanfragen. Das Skript nutzt eine REST-API, um un Mails oder Tickets per LLM klassifizieren zu lassen und speichert die Ergebnisse direkt als Excel/CSV ab.

## Was das Skript macht:
1. Liest unbearbeitete Kundenanfragen ein.
2. Schickt die Texte über eine REST-API an eine LLM-Schnittstelle (mit erzwungener JSON-Antwort).
3. Filtert automatisch:
   - **Kategorie** (z. B. Abrechnung, Stammdaten, Leistung)
   - **Dringlichkeit** (hoch, mittel, niedrig)
   - **Zusammenfassung** (kurzer Satz auf Deutsch)
4. Exportiert das Ganze sauber in eine Excel- und CSV-Datei für die weitere Analyse (z. B. in Power BI).

## Tech Stack
- Python 3
- Requests (REST API Integration)
- Pandas / openpyxl (Data Processing & Excel Export)

## Setup & Start
1. Repository klonen und Abhängigkeiten installieren:
   ```bash
   pip install requests pandas openpyxl