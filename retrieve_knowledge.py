
from pathlib import Path
import re

import chromadb
from sentence_transformers import SentenceTransformer


# 1. Define paths and configuration

BASE_DIR = Path(__file__).resolve().parent

VECTOR_DB_DIR = BASE_DIR / "chroma_db"

COLLECTION_NAME = "insurance_knowledge"

EMBEDDING_MODEL_NAME = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)

TOP_K = 10


# 2. Load embedding model

print("\n--- LOADING EMBEDDING MODEL ---")

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL_NAME
)


# 3. Connect to ChromaDB

client = chromadb.PersistentClient(
    path=str(VECTOR_DB_DIR)
)

collection = client.get_collection(
    name=COLLECTION_NAME
)

if collection.count() == 0:
    raise ValueError("Knowledge base is empty.")

print(f"[SUCCESS] Connected to ChromaDB!")
print(f"Available chunks: {collection.count()}")


# 4. Define semantic search function

def search_knowledge(query, top_k=TOP_K):

    if not query.strip():
        raise ValueError("Search query cannot be empty.")

    # Convert query into an embedding
    query_embedding = embedding_model.encode(
        query,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    # Retrieve relevant documents
    results = collection.query(
        query_embeddings=[
            query_embedding.tolist()
        ],
        n_results=min(top_k, collection.count()),
        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )

    # Display search results
    print("\n--- RETRIEVED KNOWLEDGE ---")

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for rank, (document, metadata, distance) in enumerate(
        zip(documents, metadatas, distances),
        start=1
    ):

        print(f"\nResult {rank}")

        print(f"Source: {metadata['source']}")
        print(f"Category: {metadata['category']}")
        print(f"Section: {metadata['section']}")

        print(f"Cosine distance: {distance:.4f}")

        print("\nContent:")
        print(document)

        print("-" * 50)

# 5. Simple keyword search

def search_by_keyword(query, top_k=TOP_K):

    query_words = set(re.findall(r"\w+", query.lower()))

    stop_words = {
        "was", "soll", "wenn", "mein",
        "wurde", "ich", "tun", "der",
        "die", "das", "und", "ein"
    }

    keywords = query_words - stop_words

    results = collection.get(
        include=["documents", "metadatas"]
    )

    matches = []

    for document, metadata in zip(
        results["documents"],
        results["metadatas"]
    ):

        text = document.lower()

        score = sum(
            1 for word in keywords
            if word in text
        )

        if score > 0:
            matches.append((score, document, metadata))

    matches.sort(
        key=lambda item: item[0],
        reverse=True
    )

    print("\n--- KEYWORD SEARCH RESULTS ---")

    for score, document, metadata in matches[:top_k]:

        print(f"\nSource: {metadata['source']}")
        print(f"Section: {metadata['section']}")
        print(f"Keyword matches: {score}")
        print(document[:500])
        print("-" * 50)

def search_hybrid(query, top_k=TOP_K):

    semantic_results = collection.query(
        query_embeddings=[
            embedding_model.encode(
                query,
                normalize_embeddings=True
            ).tolist()
        ],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    query_words = set(re.findall(r"\w+", query.lower()))

    stop_words = {
        "was", "soll", "wenn", "mein",
        "wurde", "ich", "tun", "der",
        "die", "das", "und", "ein"
    }

    keywords = query_words - stop_words

    combined = {}

    for document, metadata, distance in zip(
        semantic_results["documents"][0],
        semantic_results["metadatas"][0],
        semantic_results["distances"][0]
    ):
        key = (metadata["source"], metadata["section"])
        combined[key] = {
            "document": document,
            "metadata": metadata,
            "score": 1 - distance
        }

    all_documents = collection.get(
        include=["documents", "metadatas"]
    )

    for document, metadata in zip(
        all_documents["documents"],
        all_documents["metadatas"]
    ):
        text = document.lower()

        keyword_score = sum(
            1 for word in keywords if word in text
        )

        key = (metadata["source"], metadata["section"])

        if keyword_score > 0:
            if key not in combined:
                combined[key] = {
                    "document": document,
                    "metadata": metadata,
                    "score": 0
                }

            combined[key]["score"] += keyword_score

    ranked = sorted(
        combined.values(),
        key=lambda item: item["score"],
        reverse=True
    )

    print("\n--- HYBRID SEARCH RESULTS ---")

    for result in ranked[:top_k]:
        print(f"\nSource: {result['metadata']['source']}")
        print(f"Section: {result['metadata']['section']}")
        print(f"Combined score: {result['score']:.4f}")
        print(result["document"][:500])
        print("-" * 50)


# 6. Run interactive search

if __name__ == "__main__":

    query = input(
        "\nEnter your question in German: "
    ).strip()

    search_hybrid(query)
