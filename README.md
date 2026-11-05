# Smart Ticket & Claims Automation Pipeline

A Python-based prototype for automated customer support ticket classification, intelligent routing and knowledge-grounded question answering.

The project combines classical Machine Learning, Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG) in one workflow.

> **Project scope:** Educational prototype using synthetic German-language ticket data and fictional insurance knowledge documents. It is not a production system.

## 1. Project Overview

The pipeline explores three approaches to processing customer support requests:

- **LLM-based classification:** Uses the Groq API to classify incoming tickets and generate summaries.
- **Classical Machine Learning:** Uses TF-IDF and LinearSVC to classify ticket categories.
- **Hybrid ML + LLM:** Combines confidence-based ML classification with LLM-generated summaries and urgency assessment.

Additionally, a RAG component retrieves relevant information from a local knowledge base and generates answers grounded in the retrieved documents.

## 2. Key Features

### Ticket Classification
- German-language customer ticket processing.
- Five categories: Abrechnung, Stammdaten, Leistung, Beschwerde and Sonstiges.
- Urgency classification: hoch, mittel and niedrig.
- Automated summaries.
- JSON response validation and error handling.

### Machine Learning
- TF-IDF text vectorization.
- Logistic Regression and LinearSVC model comparison.
- Stratified train/test split.
- 5-fold cross-validation.
- Probability calibration using CalibratedClassifierCV.
- Confidence-based human review routing.

### Hybrid ML + LLM Pipeline
- ML predicts the ticket category.
- Low-confidence predictions are routed to human review.
- The LLM generates summaries and assesses urgency.
- Automated CSV and Excel exports.

### Retrieval-Augmented Generation (RAG)
- Local vector database using ChromaDB.
- Multilingual sentence embeddings.
- Semantic and keyword-based hybrid retrieval.
- Context-grounded answer generation using Groq.
- Source validation and insufficient-information handling.

### Automated Testing
- Pytest-based tests for retrieval functionality.
- Tests for refund and personal-data retrieval.
- Tests for empty retrieval results.
- Validation of LLM source references.

## 3. Dataset

The project uses a synthetic German-language dataset containing:

- 100 customer support tickets.
- 5 classification categories.
- 20 examples per category.

The dataset is intended for educational experimentation and does not contain real customer information.

See `DATA_LABELING_GUIDE.md` for labeling guidelines.

## 4. Model Evaluation

The following results were obtained during model experimentation:

| Model | Development Holdout Accuracy | Macro F1 |
|---|---:|---:|
| Logistic Regression | 0.70 | 0.68 |
| LinearSVC | 0.75 | 0.73 |

LinearSVC achieved slightly better results in this experiment.

**Important limitations:**
- The dataset is small and synthetic.
- The development holdout has been used during experimentation.
- The results should not be interpreted as independent production benchmarks.
- The confidence threshold of 0.60 is a demonstration setting, not a production-ready decision threshold.

## 5. RAG Knowledge Base

The knowledge base contains five fictional Markdown documents covering:

- Billing and refunds.
- Personal data.
- Benefits and treatments.
- Complaints and customer service.
- General information.

The documents are divided into 22 chunks and indexed using multilingual sentence embeddings in ChromaDB.

The generated answers are restricted to the retrieved context. If the documentation does not contain sufficient information, the system can indicate this instead of inventing an answer.

All knowledge base policies are fictional and intended exclusively for demonstration.

## 6. Tech Stack

- Python 3
- Pandas
- Scikit-learn
- TF-IDF
- LinearSVC
- Logistic Regression
- CalibratedClassifierCV
- Groq API
- REST API integration
- ChromaDB
- Sentence Transformers
- Requests
- Pytest
- Excel / CSV processing

## 7. Project Structure

```text
smart-ticket-pipeline/
│
├── data/
│   └── tickets_dataset.csv
│
├── models/
│   ├── linear_svm_pipeline.joblib
│   └── calibrated_linear_svm.joblib
│
├── knowledge_base/
│   ├── billing_and_refunds.md
│   ├── personal_data.md
│   ├── benefits_and_treatments.md
│   ├── complaints_and_service.md
│   └── general_information.md
│
├── claim_processor.py
├── train_model.py
├── compare_models.py
├── calibrate_model.py
├── predict_ticket.py
├── hybrid_claim_processor.py
├── build_knowledge_base.py
├── retrieve_knowledge.py
├── rag_answer.py
├── test_rag.py
├── check_dataset.py
├── DATA_LABELING_GUIDE.md
├── .gitignore
└── README.md
```

The `chroma_db/` directory is generated locally by the knowledge base builder and is excluded from version control.

## 8. Installation

Clone the repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd smart-ticket-pipeline
```

Install the required Python packages:

```bash
python -m pip install requests pandas openpyxl python-dotenv scikit-learn joblib chromadb sentence-transformers pytest
```

## 9. Configuration

Create a `.env` file in the project root:

```text
GROQ_API_KEY=your_groq_api_key
```

Replace the placeholder with your own Groq API key.

Never commit `.env` or expose API credentials publicly.

## 10. Running the Project

Train and compare ML models:

```bash
python train_model.py
python compare_models.py
```

Calibrate the model:

```bash
python calibrate_model.py
```

Run the hybrid pipeline:

```bash
python hybrid_claim_processor.py
```

Build the knowledge base:

```bash
python build_knowledge_base.py
```

Run the RAG application:

```bash
python rag_answer.py
```

Run automated tests:

```bash
python -m pytest test_rag.py -v
```

## 11. Testing

The current RAG test suite contains six automated tests.

All six tests passed in the latest local test run.

The tests cover document retrieval, specific knowledge base sections, empty retrieval results and validation of unknown source references.

## 12. Future Improvements

Potential next steps include:

- Expanding the labeled dataset.
- Evaluating models on an independent test dataset.
- Improving class-specific performance.
- Measuring calibration quality.
- Adding a dedicated human review interface.
- Evaluating RAG retrieval quality with a larger test set.
- Adding more comprehensive automated tests.

## 13. Author's Learning Objectives

This project demonstrates practical experience with:

- Text preprocessing and feature extraction.
- Supervised Machine Learning.
- Model evaluation and comparison.
- Probability calibration.
- LLM API integration.
- Hybrid AI architectures.
- Retrieval-Augmented Generation.
- Vector databases.
- Automated testing and reproducible workflows.

---

**Project status:** Educational prototype with working ML, LLM and RAG components.