## 概要
このプロジェクトは、Cohere Embed 4を使用して、PDFやパワーポイントの資料を画像として埋め込み、マルチモーダルRAGを構築します。
主な特徴は以下の通りです。

## 技術スタック
- 埋め込みモデル: Cohere Embed 4
- LLM: Gemini 2.5
- ベクターストア: Qdrant

## 実行方法

1. 環境変数の設定
.env.sampleを参考に環境変数設定ファイル.envを作成します。

    ```bash
    COHERE_API_KEY=your_cohere_api_key
    OPENAI_API_KEY=your_openai_api_key
    GEMINI_API_KEY=your_gemini_api_key
    ```

2. データの準備
dataフォルダにテキスト、画像、pdfなどを格納します。

3. データの取り込み
    ```bash
    uv run ingest.py
    ```

4. クエリの実行
    ```bash
    uv run query.py
    ```