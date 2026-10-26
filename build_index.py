"""Step 1, once: chunk the SOPs, embed each chunk, save the index."""
import json
import numpy as np
from sentence_transformers import SentenceTransformer

import config
from chunking import split_markdown


def main():
    chunks = []
    for path in sorted(config.PROCEDURES_DIR.glob("*.md")):
        chunks.extend(split_markdown(path, config.MAX_CHUNK_WORDS))
    for i, c in enumerate(chunks):
        c["id"] = i

    # Embed the heading path with the text, so "5.4 Serious incident reporting" carries meaning too
    model = SentenceTransformer(config.EMBED_MODEL)
    texts = [f"{c['doc_title']} - {c['section']}: {c['text']}" for c in chunks]
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)

    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    np.save(config.INDEX_DIR / "vectors.npy", vectors)
    (config.INDEX_DIR / "chunks.json").write_text(json.dumps(chunks, indent=2), encoding="utf-8")
    print(f"Indexed {len(chunks)} chunks from {len(set(c['doc'] for c in chunks))} documents.")


if __name__ == "__main__":
    main()
