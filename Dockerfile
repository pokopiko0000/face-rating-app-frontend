# Pythonの公式イメージをベースにする
FROM python:3.11-slim

# 作業ディレクトリを設定
WORKDIR /app

# 必要なライブラリをインストールするためのファイルをコピー
COPY requirements.txt .

# pipをアップグレードし、requirements.txt に基づいてライブラリをインストール
# FaissのようなC++依存ライブラリのビルドに必要なものを先にインストール
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# アプリケーションのソースコードと関連ファイルをコピー
# face_index_combined.idx と meta_combined.pkl もイメージに含める
COPY ./api.py .
COPY ./face_index_combined.idx .
COPY ./meta_combined.pkl .
# もし ~/.insightface にダウンロードされるモデルをイメージに含めたい場合、
# それをダウンロードしてCOPYする処理が必要だが、通常は実行時にダウンロードさせる
# ただし、Renderのビルド時にダウンロードされるようにした方が良い場合もある

# Uvicornがリッスンするポートを指定
EXPOSE 8000

# アプリケーションの起動コマンド
# healthcheckを追加すると、Renderがアプリの正常性を確認しやすくなる
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]