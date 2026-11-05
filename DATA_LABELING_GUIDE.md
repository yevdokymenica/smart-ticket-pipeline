
# Data Labeling Guidelines

## 1. Abrechnung
Invoices, insurance contributions, payments,
incorrect charges, billing questions and refunds.

## 2. Stammdaten
Changes to personal information, addresses,
phone numbers, email addresses and bank details.

## 3. Leistung
Questions about insurance benefits, medical
treatments, therapies, coverage and approvals.

## 4. Beschwerde
Complaints about customer service, communication,
delays and unsatisfactory handling of requests.

## 5. Sonstiges
General membership questions, application usage,
office opening hours and technical support.

## Classification Rules

- Classify according to the primary subject of the ticket.
- An incorrect invoice belongs to Abrechnung,
  even if the customer is unhappy.
- A complaint about customer service belongs to Beschwerde.
- Questions about medical coverage belong to Leistung.
- Personal information changes belong to Stammdaten.
- Use Sonstiges only when no other category applies.
