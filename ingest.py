"""Ingest documents into the vector store.

Usage:
    python ingest.py                      # ingests data/recipes (the crafting corpus)
    python ingest.py --folder my_docs     # ingests your own .txt/.md files
"""
import os
import glob
import argparse

import config
from rag.embeddings import embed_texts
from rag.store import Store


def chunk(text, size, overlap):
    text = text.replace("\r\n", "\n")
    out, i = [], 0
    while i < len(text):
        piece = text[i : i + size].strip()
        if piece:
            out.append(piece)
        i += max(1, size - overlap)
    return out


def load_files(folder):
    docs = []
    for path in glob.glob(os.path.join(folder, "**", "*.*"), recursive=True):
        if path.lower().endswith((".txt", ".md")):
            with open(path, encoding="utf-8") as f:
                docs.append((os.path.basename(path), f.read()))
    return docs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", default="data/recipes")
    ap.add_argument("--recreate", action="store_true", help="drop the collection first (fresh start)")
    args = ap.parse_args()

    store = Store()
    if args.recreate:
        try:
            store.client.delete_collection(config.COLLECTION)
            print(f"Dropped collection '{config.COLLECTION}'.")
        except Exception:
            pass
    store.ensure()

    docs = load_files(args.folder)
    if not docs:
        print(f"No .txt/.md files found in {args.folder}")
        return

    # collect all chunks first, then embed in batches (far fewer API calls)
    all_chunks = []  # (text, source)
    for name, text in docs:
        for c in chunk(text, config.CHUNK_SIZE, config.CHUNK_OVERLAP):
            all_chunks.append((c, name))

    BATCH = 100
    total = 0
    for i in range(0, len(all_chunks), BATCH):
        batch = all_chunks[i : i + BATCH]
        vectors = embed_texts([c for c, _ in batch])
        store.upsert([{"text": c, "source": s, "vector": v} for (c, s), v in zip(batch, vectors)])
        total += len(batch)
        print(f"  embedded {total}/{len(all_chunks)} chunks")

    print(f"Done. {len(docs)} docs, {total} chunks -> collection '{config.COLLECTION}'")


if __name__ == "__main__":
    main()
