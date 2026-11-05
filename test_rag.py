import json
import pytest
import rag_answer

from rag_answer import retrieve_context, generate_answer


def test_retrieve_context_returns_documents():
    question = "Was soll ich tun, wenn mein Beitrag doppelt abgebucht wurde?"

    documents = retrieve_context(question)

    assert len(documents) > 0
    assert any(
        document["source"] == "billing_and_refunds.md"
        for document in documents
    )


def test_retrieve_context_finds_duplicate_payment_section():
    question = "Was soll ich tun, wenn mein Beitrag doppelt abgebucht wurde?"

    documents = retrieve_context(question)

    assert any(
        "Duplicate Payments" in document["section"]
        for document in documents
    )

def test_retrieve_context_finds_refund_information():
    question = "Wie kann ich eine Rückerstattung beantragen?"

    documents = retrieve_context(question)

    assert any(
        document["section"] == "2. Refund Requests"
        for document in documents
    )

def test_retrieve_context_finds_personal_data():
    question = "Wie kann ich meine Adresse ändern?"

    documents = retrieve_context(question)

    assert any(
        document["source"] == "personal_data.md"
        for document in documents
    )

def test_generate_answer_handles_empty_context():
    question = "Wie funktioniert eine Rückerstattung?"

    result = generate_answer(question, [])

    assert result["answer"] == "Keine relevanten Informationen gefunden."
    assert result["sources"] == []
    assert result["insufficient_information"] is True


def test_generate_answer_rejects_unknown_source(monkeypatch):
    fake_response = {
        "choices": [
            {
                "message": {
                    "content": json.dumps({
                        "answer": "Ihre Rückerstattung wurde bearbeitet.",
                        "sources": ["unknown_document.md"],
                        "insufficient_information": False
                    })
                }
            }
        ]
    }

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return fake_response

    monkeypatch.setattr(
        rag_answer.requests,
        "post",
        lambda *args, **kwargs: FakeResponse()
    )

    documents = [{
        "source": "billing_and_refunds.md",
        "section": "2. Refund Requests",
        "content": "Information about refunds."
    }]

    with pytest.raises(ValueError, match="unknown source"):
        generate_answer(
            "Wie funktioniert eine Rückerstattung?",
            documents
        )