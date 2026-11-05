
from pathlib import Path
import re

import chromadb
from sentence_transformers import SentenceTransformer


# 1. Define project paths

BASE_DIR = Path(__file__).resolve().parent

KNOWLEDGE_BASE_DIR = BASE_DIR / "knowledge_base"
VECTOR_DB_DIR = BASE_DIR / "chroma_db"

COLLECTION_NAME = "insurance_knowledge"

# Multilingual embedding model
EMBEDDING_MODEL_NAME = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)


# 2. Extract metadata from Markdown documents

def extract_metadata(text, field, default="Unknown"):

    for line in text.splitlines():

        if line.startswith(f"{field}:"):
            return line.split(":", 1)[1].strip()

    return default


# 3. Split documents into meaningful sections

def split_into_chunks(markdown_text):

    lines = markdown_text.splitlines()

    title = next(
        (
            line.replace("# ", "", 1).strip()
            for line in lines
            if line.startswith("# ")
        ),
        "Untitled Document"
    )

    sections = []
    preamble_lines = []

    current_heading = None
    current_content = []

    for line in lines:

        if line.startswith("## "):

            if current_heading is not None:
                sections.append((
                    current_heading,
                    "\n".join(current_content).strip()
                ))

            current_heading = line.replace("## ", "", 1).strip()
            current_content = []

        elif current_heading is None:
            preamble_lines.append(line)

        else:
            current_content.append(line)

    if current_heading is not None:
        sections.append((
            current_heading,
            "\n".join(current_content).strip()
        ))

    # Preserve document context in each chunk
    preamble = "\n".join(
        line for line in preamble_lines
        if line.strip() and not line.startswith("# ")
    ).strip()

    chunks = []

    for heading, content in sections:

        chunk = "\n".join(
            part for part in [
                title,
                preamble,
                f"Section: {heading}",
                content
            ]
            if part
        )

        if chunk.strip():
            chunks.append((heading, chunk))

    return chunks


# 4. Load all knowledge documents

def load_knowledge_documents():

    if not KNOWLEDGE_BASE_DIR.exists():
        raise FileNotFoundError(
            f"Knowledge base not found: {KNOWLEDGE_BASE_DIR}"
        )

    files = sorted(KNOWLEDGE_BASE_DIR.glob("*.md"))

    if not files:
        raise ValueError("No Markdown documents found.")

    documents = []
    metadatas = []
    ids = []

    for file_path in files:

        text = file_path.read_text(encoding="utf-8")

        document_id = extract_metadata(text, "Document ID")
        category = extract_metadata(text, "Category")

        chunks = split_into_chunks(text)

        for index, (heading, chunk) in enumerate(chunks):

            ids.append(
                f"{file_path.stem}-{index + 1:03d}"
            )

            documents.append(chunk)

            metadatas.append({
                "source": file_path.name,
                "document_id": document_id,
                "category": category,
                "section": heading,
                "chunk_index": index
            })

    if not documents:
        raise ValueError("No document chunks were created.")

    return documents, metadatas, ids


# 5. Generate embeddings and store documents

def build_vector_database():

    print("\n--- LOADING KNOWLEDGE DOCUMENTS ---")

    documents, metadatas, ids = load_knowledge_documents()

    print(f"Documents loaded: {len(set(m['source'] for m in metadatas))}")
    print(f"Chunks created: {len(documents)}")

    print("\n--- LOADING EMBEDDING MODEL ---")

    embedding_model = SentenceTransformer(
        EMBEDDING_MODEL_NAME
    )

    print("\n--- GENERATING EMBEDDINGS ---")

    embeddings = embedding_model.encode(
        documents,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    print(f"Embedding dimensions: {embeddings.shape[1]}")

    print("\n--- CONNECTING TO CHROMADB ---")

    client = chromadb.PersistentClient(
        path=str(VECTOR_DB_DIR)
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    # Remove previous records from this collection
    # to avoid duplicate chunks when rebuilding.
    existing_ids = collection.get()["ids"]

    if existing_ids:
        collection.delete(ids=existing_ids)

    # Store documents, metadata and embeddings
    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings.tolist()
    )

    print("\n--- VECTOR DATABASE SUMMARY ---")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Stored chunks: {collection.count()}")
    print(f"Database location: {VECTOR_DB_DIR}")

    print("\n[SUCCESS] Knowledge base built successfully!")


# 6. Run pipeline

if __name__ == "__main__":
    build_vector_database()
