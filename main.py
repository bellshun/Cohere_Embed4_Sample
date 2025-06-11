from PIL import Image
from dotenv import load_dotenv
import os
import io
import base64
import numpy as np
import cohere
import google.generativeai as genai

load_dotenv()

# 設定
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
co = cohere.ClientV2(api_key=COHERE_API_KEY)
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

# ヘルパー：画像 → base64
def image_to_base64(path):
    img = Image.open(path).convert("RGB")
    img.thumbnail((1568, 1568))  # Cohereの制限サイズ
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"

# 画像の埋め込み
def embed_image(path):
    b64img = image_to_base64(path)
    doc = {"content": [{"type": "image", "image": b64img}]}
    res = co.embed(
        model="embed-v4.0",
        input_type="search_document",
        embedding_types=["float"],
        inputs=[doc]
    )
    return np.array(res.embeddings.float[0])

# 質問の埋め込み
def embed_question(question):
    res = co.embed(
        model="embed-v4.0",
        input_type="search_query",
        embedding_types=["float"],
        texts=[question]
    )
    return np.array(res.embeddings.float[0])

# 類似画像検索（画像1枚のみの場合はそのまま返す）
def find_best_image(query_emb, image_embs):
    sim = np.dot(query_emb, image_embs.T)
    return int(np.argmax(sim))

# Geminiで回答生成
def generate_answer(question, image_path):
    img = Image.open(image_path)
    res = gemini_model.generate_content([f"以下の画像を見て、質問に答えてください：{question}", img])
    return res.text

# メイン処理
def main():
    image_path = input("画像ファイルのパスを入力してください: ").strip()
    if not os.path.exists(image_path):
        print("画像が見つかりません。")
        return

    image_emb = embed_image(image_path)

    print("質問を日本語で入力してください（終了するには Ctrl+C）")
    while True:
        try:
            question = input("\n> 質問: ")
            q_emb = embed_question(question)
            idx = find_best_image(q_emb, image_emb[np.newaxis, :])
            answer = generate_answer(question, image_path)
            print("\n=== 回答 ===\n", answer)
        except KeyboardInterrupt:
            print("\n終了します。")
            break

if __name__ == "__main__":
    main()
