import os
import base64
import io
import numpy as np
from PIL import Image
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from dotenv import load_dotenv
import cohere

load_dotenv()
co = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))

client = QdrantClient(path="qdrant_data")
COLLECTION_NAME = "agentrag_collection"

def embed_text(text):
    res = co.embed(
        model="embed-v4.0",
        input_type="search_document",
        embedding_types=["float"],
        texts=[text]
    )
    return np.array(res.embeddings.float[0])

def embed_image(path):
    img = Image.open(path).convert("RGB")
    img.thumbnail((1568, 1568))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    res = co.embed(
        model="embed-v4.0",
        input_type="search_document",
        embedding_types=["float"],
        inputs=[{"content": [{"type": "image", "image": f"data:image/png;base64,{b64}"}]}]
    )
    return np.array(res.embeddings.float[0])

def read_and_embed(filepath):
    ext = os.path.splitext(filepath)[1].lower()
    if ext in [".png", ".jpg", ".jpeg"]:
        emb = embed_image(filepath)
        payload = {"type": "image", "path": filepath}
    elif ext == ".csv":
        df = pd.read_csv(filepath)
        text = df.head().to_markdown()
        emb = embed_text(text)
        payload = {"type": "table", "path": filepath, "content": text}
    elif ext == ".txt":
        with open(filepath, "r") as f:
            text = f.read()
        emb = embed_text(text)
        payload = {"type": "text", "path": filepath, "content": text}
    else:
        return None, None
    return emb, payload

def ingest_all():
    if client.collection_exists(collection_name=COLLECTION_NAME):
        client.delete_collection(collection_name=COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
    )

    points = []
    for i, filename in enumerate(os.listdir("data")):
        path = os.path.join("data", filename)
        emb, payload = read_and_embed(path)
        if emb is not None:
            points.append(PointStruct(id=i, vector=emb.tolist(), payload=payload))

    client.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"{len(points)} 件のデータをインデックスしました。")

if __name__ == "__main__":
    ingest_all()
