import os
import numpy as np
import cohere
import google.generativeai as genai
from qdrant_client import QdrantClient
from PIL import Image
from dotenv import load_dotenv

# 初期化
load_dotenv()
co = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
gemini = genai.GenerativeModel("gemini-2.5-flash-preview-05-20")
client = QdrantClient(path="qdrant_data")

COLLECTION_NAME = "agentrag_collection"


def embed_question(text):
    """質問テキストを埋め込みベクトルに変換"""
    res = co.embed(
        model="embed-v4.0",
        input_type="search_query",
        embedding_types=["float"],
        texts=[text]
    )
    return np.array(res.embeddings.float[0])


def search_similar(query, top_k=3):
    """類似検索を実行"""
    query_vec = embed_question(query)
    result = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vec.tolist(),
        limit=top_k
    )
    return result.points


def build_prompt(query, search_results):
    """検索結果から回答生成用のプロンプトを構築"""
    prompt_parts = [
        "以下のコンテキスト（画像とテキスト）を総合的に判断して、質問に簡潔に答えてください。",
        "画像に基づいた回答には、画像ファイルパスも併記してください。",
        "---"
    ]
    
    for hit in search_results:
        payload = hit.payload
        
        if payload["type"] in ["image", "pdf_image"]:
            # 画像ファイルの処理
            if os.path.exists(payload['path']):
                img = Image.open(payload['path'])
                prompt_parts.append(img)
                
                if payload["type"] == "pdf_image":
                    source_info = f"（{payload.get('source_pdf', '不明')} ページ: {payload.get('page_num', '?')}）"
                    prompt_parts.append(f"[PDF画像: {payload['path']}{source_info}]")
                else:
                    prompt_parts.append(f"[画像ファイル: {payload['path']}]")
            else:
                print(f"警告: ファイルが見つかりません: {payload['path']}")
        
        elif payload["type"] in ["table", "text"]:
            # テキストファイルの処理
            prompt_parts.append(f"[参考資料: {payload['path']}]\n{payload['content']}\n---")
    
    prompt_parts.extend([
        "---",
        f"## 質問\n{query}",
        "## 回答"
    ])
    
    return prompt_parts


def generate_answer(query, search_results):
    """検索結果を使って回答を生成"""
    prompt_parts = build_prompt(query, search_results)
    print("プロンプト構築完了:", len(prompt_parts))
    
    response = gemini.generate_content(prompt_parts)
    return response.text


def print_search_results(results):
    """検索結果を表示"""
    print("\n--- 検索結果 ---")
    for hit in results:
        print(f"Score: {hit.score:.4f}, Type: {hit.payload['type']}, Path: {hit.payload['path']}")


def main():
    """メインループ"""
    print("RAG検索システム開始（Ctrl+Cで終了）")
    
    while True:
        try:
            query = input("\n質問を入力してください: ").strip()
            if not query:
                continue
            
            # 検索実行
            results = search_similar(query)
            print_search_results(results)
            
            # 回答生成
            print("\n--- 回答生成中 ---")
            answer = generate_answer(query, results)
            print("\n✅ 回答")
            print(answer)
            
        except KeyboardInterrupt:
            print("\n終了します。")
            break
        except Exception as e:
            print(f"\nエラーが発生しました: {e}")


if __name__ == "__main__":
    main()