import faiss
import pickle
import io
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware # CORSを許可する場合
import uvicorn
import insightface
import onnxruntime as ort # 明示的にインポート
import os
import time # パフォーマンス測定用 (オプション)
from fastapi.middleware.cors import CORSMiddleware

print("--- ライブラリバージョン情報 ---")
print(f"Faiss version: {faiss.__version__}")
print(f"NumPy version: {np.__version__}")
print(f"InsightFace version: {insightface.__version__}")
print(f"ONNX Runtime version: {ort.__version__}")
print("-----------------------------")

### ★★★ グローバル変数・モデルロード (API起動時に一度だけ実行) ★★★ ###
# 1. Faissインデックスとメタデータのファイルパス
FAISS_INDEX_FILE = "face_index_combined.idx"
META_FILE = "meta_combined.pkl"

# 2. InsightFace/ArcFaceモデルの準備
#    (CPU実行を想定。もしGPUを使いたい場合は providers=['CUDAExecutionProvider'] などに変更)
try:
    print("InsightFaceモデルの準備中...")
    face_analysis_app = insightface.app.FaceAnalysis(
        name='buffalo_l',
        root='~/.insightface',
        providers=['CPUExecutionProvider']
    )
    face_analysis_app.prepare(ctx_id=0, det_size=(640, 640))
    print("InsightFace FaceAnalysis model loaded successfully on CPU.")
except Exception as e:
    print(f"致命的エラー: InsightFaceモデルの初期化に失敗しました: {e}")
    # ここでプログラムを終了させるか、エラー状態を示すフラグを立てる
    face_analysis_app = None # モデルロード失敗

# 3. Faissインデックスとメタデータのロード
faiss_index = None
metadata = None
db_scores = None
db_countries = None

if os.path.exists(FAISS_INDEX_FILE) and os.path.exists(META_FILE):
    try:
        print(f"Faissインデックス '{FAISS_INDEX_FILE}' をロード中...")
        faiss_index = faiss.read_index(FAISS_INDEX_FILE)
        print(f"  Faissインデックスロード完了。登録ベクトル数: {faiss_index.ntotal if faiss_index else 'N/A'}")

        print(f"メタデータ '{META_FILE}' をロード中...")
        with open(META_FILE, "rb") as f:
            metadata = pickle.load(f)
        db_scores = metadata.get("scores")
        db_countries = metadata.get("countries")
        
        if db_scores is not None and db_countries is not None:
            print(f"  メタデータロード完了。スコア数: {len(db_scores)}, 国ラベル数: {len(db_countries)}")
            if faiss_index and faiss_index.ntotal != len(db_scores):
                print("警告: Faissインデックスのベクトル数とメタデータの数が一致しません。データを確認してください。")
        else:
            print("エラー: メタデータファイルに必要な 'scores' または 'countries' キーが含まれていません。")
            faiss_index = None # メタデータがないと機能しないのでインデックスも無効化
    except Exception as e:
        print(f"致命的エラー: Faissインデックスまたはメタデータのロードに失敗しました: {e}")
        faiss_index = None # ロード失敗
else:
    print(f"エラー: Faissインデックスファイル ({FAISS_INDEX_FILE}) またはメタデータファイル ({META_FILE}) が見つかりません。")
    print("Day6の build_index.py を実行してファイルを生成してください。")

### ★★★ FastAPIアプリケーションの初期化 ★★★ ###
app = FastAPI(title="Face Attractiveness Ranking API")

# CORSミドルウェアの設定
origins = [
    "http://localhost",  # もしフロントエンドを別のローカルサーバーで動かす場合
    "http://127.0.0.1", # localhostの別名
    "null",             # ローカルファイル (file:///) からのアクセスを許可する場合 (今回重要)
    # 必要に応じて、将来デプロイするフロントエンドのドメインも追加
    # "https://your-frontend-domain.com",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins, # 特定のオリジンを許可 (開発中は ["*"] ですべて許可も手軽)
    allow_credentials=True,
    allow_methods=["*"], # すべてのHTTPメソッドを許可 (GET, POST, PUTなど)
    allow_headers=["*"], # すべてのHTTPヘッダーを許可
)

### ★★★ APIエンドポイント ★★★ ###

def get_face_embedding_from_bytes(image_bytes):
    """アップロードされた画像バイト列から顔特徴ベクトルを抽出する"""
    if face_analysis_app is None:
        return None
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_np = np.array(img)
        faces = face_analysis_app.get(img_np)
        if not faces:
            return None # 顔が検出されなかった
        return faces[0].normed_embedding.astype(np.float32) # Faissはfloat32を期待
    except Exception as e:
        print(f"顔特徴ベクトルの抽出中にエラー: {e}")
        return None

def calculate_ranking(query_embedding, k_neighbors=40):
    """Faissで類似検索し、国ごとの平均スコアでランキングを計算する"""
    if faiss_index is None or db_scores is None or db_countries is None:
        return None

    try:
        # 類似ベクトルをk個検索 (D: 距離/類似度の配列, I: インデックスの配列)
        # IndexFlatIPの場合、Dは内積なので値が大きいほど類似している
        distances, indices = faiss_index.search(query_embedding.reshape(1, -1), k_neighbors)
        
        country_scores_sum = {}
        country_counts = {}

        for i in range(k_neighbors):
            db_index = indices[0][i]
            similarity_score = distances[0][i] # 内積スコア（類似度）
            
            if db_index < 0 or db_index >= len(db_countries): # 無効なインデックスはスキップ
                continue

            country = db_countries[db_index]
            beauty_score = db_scores[db_index]

            # 類似度で重み付けしたスコアを加算 (オプション、単純平均でも可)
            # ここでは類似度(内積)をそのまま重みとして使う (0に近いほど非類似なので影響小)
            # より類似しているもの（内積が大きい）のスコアを重視
            weighted_score = beauty_score * max(0, similarity_score) # 内積が負の場合は0にする

            country_scores_sum.setdefault(country, 0.0)
            country_scores_sum[country] += weighted_score
            
            country_counts.setdefault(country, 0)
            country_counts[country] += 1
            
            # 単純平均の場合の準備 (参考)
            # country_scores_sum.setdefault(country, []).append(beauty_score)


        # 国ごとの平均スコアを計算
        ranked_results = []
        for country, total_weighted_score in country_scores_sum.items():
            count = country_counts[country]
            if count > 0:
                # 重み付き平均スコア (重みの合計で割る必要があるが、ここでは単純に類似近傍のスコアの重み付き和とするか、
                # あるいは、近傍k個の「平均beauty_score」を「類似度で重み付け」するか、
                # または、あなたの提案通り「抜粋した顔の beauty_score を平均 or 加重平均（1-距離で重み付け）」
                # ここでは、あなたの提案の「抜粋した顔のbeauty_scoreを平均」に近い形で、
                # まずは類似した顔のbeauty_scoreを集め、その平均を取るロジックを考えます。
                # 簡単のため、ここでは集めたbeauty_scoreの単純平均とします。
                # より洗練させるなら、類似度(distances[0][i])で重み付けした平均が良いでしょう。
                # 以下は、近傍k個に含まれる各国のbeauty_scoreの単純平均を計算する例
                
                # 修正: 近傍に含まれるその国のスコアのリストを取得
                scores_for_country = [db_scores[idx] for j, idx in enumerate(indices[0]) if db_countries[idx] == country and j < k_neighbors]
                if scores_for_country:
                    average_score_for_country = float(np.mean(scores_for_country))
                    # スコアを0-100の範囲にスケーリング (元のスコアが0-1正規化されていると仮定)
                    # もし元のスコアが1-5や1-10なら、スケーリング係数を調整
                    scaled_score = average_score_for_country * 100 
                    ranked_results.append({"nation": country, "score": round(scaled_score, 2), "sample_count": len(scores_for_country)})

        # スコアの高い順にソートしてTop-5を取得
        ranked_results = sorted(ranked_results, key=lambda x: x["score"], reverse=True)[:5]
        return ranked_results

    except Exception as e:
        print(f"ランキング計算中にエラー: {e}")
        return None


@app.post("/rank", summary="顔画像をアップロードして、好まれそうな国・地域グループをランキング形式で取得")
async def rank_face_attractiveness(file: UploadFile = File(...)):
    """
    顔画像をアップロードすると、データベース内の美的評価に基づいて、
    どの国・地域グループで好まれそうかのTop-5を推定スコアと共に返します。
    - **file**: アップロードする顔画像ファイル (JPG, PNGなど)
    """
    start_time = time.time()

    if face_analysis_app is None or faiss_index is None:
        raise HTTPException(status_code=503, detail="サーバーの準備ができていません。モデルまたはインデックスがロードされていません。")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="画像ファイルが空です。")

    query_embedding = get_face_embedding_from_bytes(image_bytes)
    if query_embedding is None:
        raise HTTPException(status_code=400, detail="顔が検出されませんでした。または画像処理エラーが発生しました。")

    ranking = calculate_ranking(query_embedding)
    if ranking is None:
        raise HTTPException(status_code=500, detail="ランキングの計算中にエラーが発生しました。")

    end_time = time.time()
    print(f"Processing time for /rank: {end_time - start_time:.4f} seconds")
    
    return {"top_results": ranking}

# --- (オプション) ルートエンドポイント ---
@app.get("/", summary="APIのルートエンドポイント")
async def read_root():
    return {"message": "顔魅力ランキングAPIへようこそ！ /rank エンドポイントにPOSTリクエストで顔画像を送信してください。"}

# --- Uvicornでサーバーを起動 ---
if __name__ == "__main__":
    print("APIサーバーを起動します...")
    print(f"Faissインデックス ({FAISS_INDEX_FILE}) とメタデータ ({META_FILE}) のロード状態:")
    
    # ↓↓↓↓ このif文の条件を修正 ↓↓↓↓
    # if faiss_index and db_scores and db_countries: # 修正前
    if faiss_index is not None and \
       db_scores is not None and len(db_scores) > 0 and \
       db_countries is not None and len(db_countries) > 0: # 修正後
        print(f"  インデックスとメタデータは正常にロードされています。ベクトル数: {faiss_index.ntotal}")
    else:
        print("  警告: インデックスまたはメタデータがロードされていないか、データが空です。APIは正しく機能しない可能性があります。")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)