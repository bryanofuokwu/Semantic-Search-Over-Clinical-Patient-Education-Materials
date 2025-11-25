import os
import argparse
from typing import List, Dict

import pandas as pd


# -----------------------------
# Default paths
# -----------------------------

DEFAULT_RAW_PATH = os.path.join("data", "raw", "patient_education.parquet")
DEFAULT_DOC_OUTPUT_PATH = os.path.join("data", "processed", "patient_education_full.parquet")
DEFAULT_CHUNK_OUTPUT_PATH = os.path.join("data", "processed", "patient_education_chunks.parquet")


# -----------------------------
# Helpers
# -----------------------------

def build_full_text_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine the section fields into a single 'full_text' string per document.
    This is the text that will later be chunked and embedded.
    """

    def combine(row) -> str:
        parts = [
            f"Title: {row['title']}",
            "",
            "Overview:",
            row["overview"],
            "",
            "Symptoms:",
            row["symptoms"],
            "",
            "Causes:",
            row["causes"],
            "",
            "Diagnosis:",
            row["diagnosis"],
            "",
            "Treatment Options:",
            row["treatment_options"],
            "",
            "Self-Care and Daily Management:",
            row["self_care"],
            "",
            "When to Seek Help:",
            row["when_to_seek_help"],
            "",
            "Frequently Asked Questions:",
            row["faq"],
        ]
        return "\n".join(parts)

    df = df.copy()
    df["full_text"] = df.apply(combine, axis=1)
    return df


def chunk_text(text: str, max_chars: int, overlap: int) -> List[str]:
    """
    Split a long string into overlapping chunks based on character count.

    - max_chars: maximum characters per chunk
    - overlap: how many characters overlap between consecutive chunks

    This is a simple, model-agnostic chunker; it works fine for
    sentence-transformer-style embedding pipelines.
    """
    text = text.strip()
    if not text:
        return []

    chunks: List[str] = []
    n = len(text)
    start = 0

    while start < n:
        end = min(start + max_chars, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= n:
            break

        # Move start forward but keep some overlap
        start = max(0, end - overlap)

    return chunks


def make_chunks_df(df_docs: pd.DataFrame, max_chars: int, overlap: int) -> pd.DataFrame:
    """
    Create a chunk-level DataFrame from the doc-level DataFrame.

    Output columns:
      - doc_id
      - chunk_id
      - condition
      - category
      - title
      - reading_level
      - chunk_text
    """
    records: List[Dict] = []

    for _, row in df_docs.iterrows():
        doc_id = row["id"]
        condition = row["condition"]
        category = row["category"]
        title = row["title"]
        reading_level = row.get("reading_level", None)
        full_text = row["full_text"]

        chunks = chunk_text(full_text, max_chars=max_chars, overlap=overlap)
        for idx, chunk in enumerate(chunks):
            records.append(
                {
                    "doc_id": doc_id,
                    "chunk_id": idx,
                    "condition": condition,
                    "category": category,
                    "title": title,
                    "reading_level": reading_level,
                    "chunk_text": chunk,
                }
            )

    return pd.DataFrame.from_records(records)


# -----------------------------
# CLI / entrypoint
# -----------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess patient education documents: build full_text and chunks."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=DEFAULT_RAW_PATH,
        help=f"Path to raw patient education Parquet (default: {DEFAULT_RAW_PATH})",
    )
    parser.add_argument(
        "--doc_output",
        type=str,
        default=DEFAULT_DOC_OUTPUT_PATH,
        help=f"Output path for doc-level Parquet with full_text (default: {DEFAULT_DOC_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--chunk_output",
        type=str,
        default=DEFAULT_CHUNK_OUTPUT_PATH,
        help=f"Output path for chunk-level Parquet (default: {DEFAULT_CHUNK_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=1200,
        help="Maximum number of characters per chunk (default: 1200)",
    )
    parser.add_argument(
        "--chunk_overlap",
        type=int,
        default=200,
        help="Character overlap between consecutive chunks (default: 200)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = args.input
    doc_output_path = args.doc_output
    chunk_output_path = args.chunk_output
    max_chars = args.chunk_size
    overlap = args.chunk_overlap

    # Ensure output directories exist
    os.makedirs(os.path.dirname(doc_output_path), exist_ok=True)
    os.makedirs(os.path.dirname(chunk_output_path), exist_ok=True)

    # 1. Load raw docs
    print(f"Loading raw docs from: {input_path}")
    df_raw = pd.read_parquet(input_path)
    print(f"Loaded {len(df_raw)} documents.")

    # 2. Build full_text column
    print("Building full_text column...")
    df_docs = build_full_text_column(df_raw)

    # 3. Save doc-level processed file
    df_docs.to_parquet(doc_output_path, index=False)
    print(f"Saved doc-level processed file with full_text to: {doc_output_path}")

    # 4. Build chunks
    print(
        f"Chunking documents into max {max_chars} chars with {overlap} char overlap..."
    )
    df_chunks = make_chunks_df(df_docs, max_chars=max_chars, overlap=overlap)
    print(f"Generated {len(df_chunks)} chunks from {len(df_docs)} documents.")

    # 5. Save chunk-level file
    df_chunks.to_parquet(chunk_output_path, index=False)
    print(f"Saved chunk-level file to: {chunk_output_path}")


if __name__ == "__main__":
    main()