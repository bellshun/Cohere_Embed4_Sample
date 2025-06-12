import numpy as np
import cohere
import os
from dotenv import load_dotenv
import google.generativeai as genai
from qdrant_client import QdrantClient
from PIL import Image

load_dotenv()

co = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
gemini = genai.GenerativeModel("gemini-2.5-flash-preview-05-20") 

client = QdrantClient(path="qdrant_data")
COLLECTION_NAME = "agentrag_collection"

def embed_question(text):
    res = co.embed(
        model="embed-v4.0",
        input_type="search_query",
        embedding_types=["float"],
        texts=[text]
    )
    return np.array(res.embeddings.float[0])

def search_similar(query, top_k=3):
    query_vec = embed_question(query)
    
    search_result = client.query_points(  
        collection_name=COLLECTION_NAME,  
        query=query_vec.tolist(),
        limit=top_k  
    )  
    return search_result.points

# 修正点: Geminiに画像データを渡せるように関数を全面的に修正
def generate_answer(query, top_results):
    """
    検索結果（テキストと画像）を使って、Geminiで回答を生成する。
    """

    prompt_parts = [
        "以下のコンテキスト（画像とテキスト）を総合的に判断して、質問に簡潔に答えてください。",
        "画像に基づいた回答には、画像ファイルパスも併記してください。",
        "---"
    ]

    # 検索結果からテキストと画像を抽出
    for hit in top_results:
        p = hit.payload
    
        if p["type"] == "image":
            try:
                img = Image.open(p['path'])
                prompt_parts.append(img)
                prompt_parts.append(f"[画像ファイル: {p['path']}]")
            except FileNotFoundError:
                print(f"警告: ファイルが見つかりません: {p['path']}")
        elif p["type"] in ["table", "text"]:
            prompt_parts.append(f"[参考資料: {p['path']}]\n{p['content']}\n---")
            
    # 最後に質問と回答セクションを追加
    prompt_parts.extend([
        "---",
        f"## 質問\n{query}",
        "## 回答"
    ])

    # 生成したプロンプトを出力
    print(prompt_parts)

    res = gemini.generate_content(prompt_parts)
    return res.text


def main():
    while True:
        try:
            query = input("\n質問を入力してください（Ctrl+Cで終了）: ")
            if not query:
                continue
            results = search_similar(query)
            print("\n--- 検索結果 ---")
            for hit in results:
                print(f"Score: {hit.score:.4f}, Type: {hit.payload['type']}, Path: {hit.payload['path']}")
            
            print("--- 回答生成中 ---")
            answer = generate_answer(query, results)
            print("\n✅ 回答\n", answer)
        
        except KeyboardInterrupt:
            print("\n終了します。")
            break
        except Exception as e:
            print(f"\nエラーが発生しました: {e}")


if __name__ == "__main__":
    main()