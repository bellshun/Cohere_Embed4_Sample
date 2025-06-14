import os
import base64
import io
import uuid
from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np
from PIL import Image
import pandas as pd
import fitz
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from dotenv import load_dotenv
import cohere

# 設定
COHERE_MODEL = "embed-v4.0"
COLLECTION_NAME = "agentrag_collection"
VECTOR_SIZE = 1536
IMAGE_THUMBNAIL_SIZE = (1568, 1568)
PDF_DPI = 300
DATA_DIR = "data"
QDRANT_PATH = "qdrant_data"

load_dotenv()
co = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))
client = QdrantClient(path=QDRANT_PATH)


class EmbeddingService:
    """埋め込み生成サービス"""
    
    def embed_text(self, text: str) -> np.ndarray:
        res = co.embed(
            model=COHERE_MODEL,
            input_type="search_document",
            embedding_types=["float"],
            texts=[text]
        )
        return np.array(res.embeddings.float[0])
    
    def embed_image_from_pil(self, pil_img: Image.Image) -> np.ndarray:
        pil_img.thumbnail(IMAGE_THUMBNAIL_SIZE)
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        
        res = co.embed(
            model=COHERE_MODEL,
            input_type="search_document",
            embedding_types=["float"],
            inputs=[{"content": [{"type": "image", "image": f"data:image/png;base64,{b64}"}]}]
        )
        return np.array(res.embeddings.float[0])
    
    def embed_image(self, path: str) -> np.ndarray:
        img = Image.open(path).convert("RGB")
        return self.embed_image_from_pil(img)


class FileProcessor(ABC):
    """ファイル処理の基底クラス"""
    
    def __init__(self, embedding_service: EmbeddingService):
        self.embedding_service = embedding_service
    
    @abstractmethod
    def can_process(self, filepath: str) -> bool:
        pass
    
    @abstractmethod
    def process(self, filepath: str) -> List[Tuple[np.ndarray, dict]]:
        pass


class ImageProcessor(FileProcessor):
    """画像ファイル処理"""
    
    def can_process(self, filepath: str) -> bool:
        ext = os.path.splitext(filepath)[1].lower()
        return ext in [".png", ".jpg", ".jpeg"]
    
    def process(self, filepath: str) -> List[Tuple[np.ndarray, dict]]:
        emb = self.embedding_service.embed_image(filepath)
        payload = {"type": "image", "path": filepath}
        return [(emb, payload)]


class CSVProcessor(FileProcessor):
    """CSVファイル処理"""
    
    def can_process(self, filepath: str) -> bool:
        return os.path.splitext(filepath)[1].lower() == ".csv"
    
    def process(self, filepath: str) -> List[Tuple[np.ndarray, dict]]:
        df = pd.read_csv(filepath)
        text = df.head().to_markdown()
        emb = self.embedding_service.embed_text(text)
        payload = {"type": "table", "path": filepath, "content": text}
        return [(emb, payload)]


class TextProcessor(FileProcessor):
    """テキストファイル処理"""
    
    def can_process(self, filepath: str) -> bool:
        return os.path.splitext(filepath)[1].lower() == ".txt"
    
    def process(self, filepath: str) -> List[Tuple[np.ndarray, dict]]:
        with open(filepath, "r") as f:
            text = f.read()
        emb = self.embedding_service.embed_text(text)
        payload = {"type": "text", "path": filepath, "content": text}
        return [(emb, payload)]


class PDFProcessor(FileProcessor):
    """PDFファイル処理"""
    
    def can_process(self, filepath: str) -> bool:
        return os.path.splitext(filepath)[1].lower() == ".pdf"
    
    def process(self, filepath: str) -> List[Tuple[np.ndarray, dict]]:
        doc = fitz.open(filepath)
        points = []
        
        for i, page in enumerate(doc):
            pix = page.get_pixmap(dpi=PDF_DPI)
            img_data = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_data))
            
            emb = self.embedding_service.embed_image_from_pil(img)
            img_path = f"{DATA_DIR}/pdf_{os.path.basename(filepath)}_page_{i+1}.png"
            img.save(img_path)
            
            payload = {
                "type": "pdf_image",
                "path": img_path,
                "source_pdf": filepath,
                "page_num": i + 1
            }
            points.append((emb, payload))
        
        return points


class DocumentIndexer:
    """ドキュメントインデックス作成"""
    
    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.processors = [
            ImageProcessor(self.embedding_service),
            CSVProcessor(self.embedding_service),
            TextProcessor(self.embedding_service),
            PDFProcessor(self.embedding_service)
        ]
    
    def read_and_embed(self, filepath: str) -> List[Tuple[np.ndarray, dict]]:
        for processor in self.processors:
            if processor.can_process(filepath):
                points = processor.process(filepath)
                print(points)
                return points
        
        print(f"サポートされていないファイル形式: {filepath}")
        return []
    
    def ingest_all(self):
        # コレクション初期化
        if client.collection_exists(collection_name=COLLECTION_NAME):
            client.delete_collection(collection_name=COLLECTION_NAME)
        
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
            # 量子化を無効にして精度を最大化  
            quantization_config=None  
        )
        
        # ファイル処理
        point_structs = []
        for filename in os.listdir(DATA_DIR):
            path = os.path.join(DATA_DIR, filename)
            items = self.read_and_embed(path)
            
            for emb, payload in items:
                point_structs.append(
                    PointStruct(
                        id=uuid.uuid4().int >> 64,
                        vector=emb.tolist(),
                        payload=payload
                    )
                )
        
        client.upsert(collection_name=COLLECTION_NAME, points=point_structs)
        print(f"{len(point_structs)} 件のデータをインデックスしました。")


if __name__ == "__main__":
    indexer = DocumentIndexer()
    indexer.ingest_all()