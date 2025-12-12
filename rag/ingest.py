import os, glob
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from sentence_transformers import SentenceTransformer

COLLECTION = "runbooks"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def chunk(text: str, max_chars=800):
    out, buf = [], []
    size = 0
    for line in text.splitlines():
        if size + len(line) > max_chars and buf:
            out.append("\n".join(buf))
            buf, size = [], 0
        buf.append(line)
        size += len(line)
    if buf:
        out.append("\n".join(buf))
    return [c.strip() for c in out if c.strip()]

def main():
    embedder = SentenceTransformer(MODEL_NAME)
    client = QdrantClient(url="http://localhost:6333")

    # recreate collection
    dim = embedder.get_sentence_embedding_dimension()
    client.recreate_collection(
        collection_name=COLLECTION,
        vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
    )

    points = []
    pid = 1
    for path in glob.glob("runbooks/*.md"):
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        for i, c in enumerate(chunk(text)):
            vec = embedder.encode(c).tolist()
            points.append(qm.PointStruct(
                id=pid,
                vector=vec,
                payload={"source": os.path.basename(path), "chunk": i, "text": c},
            ))
            pid += 1

    client.upsert(collection_name=COLLECTION, points=points)
    print(f"Upserted {len(points)} chunks into Qdrant collection '{COLLECTION}'.")

if __name__ == "__main__":
    main()
