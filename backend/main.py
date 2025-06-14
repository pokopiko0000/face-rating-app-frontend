from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import cv2
import insightface
import numpy as np
import os
import io
import pycountry
import json
from pycountry_convert import (
    country_alpha2_to_continent_code,
    country_name_to_country_alpha2,
)
from pathlib import Path

# --- 定数と設定 ---

# このファイルの場所を基準にパスを構築
BASE_DIR = Path(__file__).resolve().parent

# 国別のメタデータ（大陸、意外度など）
# 意外度: ★1 (よく聞く) 〜 ★5 (かなり珍しい)
COUNTRY_METADATA = {
    "Japan": {"continent": "AS", "rarity": 1},
    "China": {"continent": "AS", "rarity": 1},
    "Korea, Republic of": {"continent": "AS", "rarity": 1},
    "United States": {"continent": "NA", "rarity": 1},
    "India": {"continent": "AS", "rarity": 1},
    "Brazil": {"continent": "SA", "rarity": 2},
    "France": {"continent": "EU", "rarity": 2},
    "Germany": {"continent": "EU", "rarity": 2},
    "United Kingdom": {"continent": "EU", "rarity": 2},
    "Nigeria": {"continent": "AF", "rarity": 3},
    "Egypt": {"continent": "AF", "rarity": 3},
    "Turkey": {"continent": "AS", "rarity": 3},
    "Argentina": {"continent": "SA", "rarity": 3},
    "Sweden": {"continent": "EU", "rarity": 4},
    "New Zealand": {"continent": "OC", "rarity": 4},
    "Peru": {"continent": "SA", "rarity": 4},
    "Zimbabwe": {"continent": "AF", "rarity": 5},
    "Papua New Guinea": {"continent": "OC", "rarity": 5},
    "Bhutan": {"continent": "AS", "rarity": 5},
    "Luxembourg": {"continent": "EU", "rarity": 5},
}

# ボーナス設定
GEO_BONUS = 0.08  # 違う大陸だった場合のボーナス (8%)
RARITY_BONUS_UNIT = 0.03  # 意外度★1つあたりのボーナス (3%)

# --- グローバル変数 ---
# モデルとプロトタイプは起動時に一度だけロードする
model = None
# 性別ごとにデータを保持する
prototypes = {"man": {}, "woman": {}}
representatives = {"man": {}, "woman": {}}
country_code_cache = {}  # 国コード検索を高速化するためのキャッシュ


# --- 類似度計算関数 ---
def cosine_similarity(v1, v2):
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    # ゼロ除算を避ける
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    return dot_product / (norm_v1 * norm_v2)


def get_continent(country_name):
    """国名から大陸コードを取得する"""
    try:
        # 事前定義したメタデータにあればそれを使う
        if country_name in COUNTRY_METADATA:
            return COUNTRY_METADATA[country_name]["continent"]

        # なければライブラリで変換を試みる
        country_alpha2 = country_name_to_country_alpha2(country_name)
        continent_code = country_alpha2_to_continent_code(country_alpha2)
        return continent_code
    except:
        # 変換に失敗した場合はNoneを返す
        return None


# --- 国コード取得関数 ---
def get_country_code(country_name):
    if country_name in country_code_cache:
        return country_code_cache[country_name]

    try:
        country = pycountry.countries.search_fuzzy(country_name)[0]
        code = country.alpha_2
        country_code_cache[country_name] = code
        return code
    except LookupError:
        country_code_cache[country_name] = None
        return None


# --- FastAPIアプリケーション ---
app = FastAPI()

# --- 静的ファイル配信の設定 ---
# BASE_DIRを使って、実行場所によらない絶対パスを指定
app.mount("/images", StaticFiles(directory=BASE_DIR / "cropped_images"), name="images")

# --- CORSミドルウェアの設定 ---
# ブラウザからのリクエストを許可するための設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # すべてのオリジンを許可（開発用）
    allow_credentials=True,
    allow_methods=["*"],  # すべてのHTTPメソッドを許可
    allow_headers=["*"],  # すべてのヘッダーを許可
)


# --- 起動時イベント ---
@app.on_event("startup")
def load_models():
    global model, prototypes, representatives
    # 1. 顔分析モデルを準備
    print("顔分析モデルを準備しています...")
    model = insightface.app.FaceAnalysis(providers=["CPUExecutionProvider"])
    # 性別・年齢推定モデルも有効化
    model.prepare(ctx_id=0, det_thresh=0.1, det_size=(640, 640))
    print("モデルの準備が完了しました。")

    # 2. 国の代表顔ベクトルをロード（性別ごと）
    prototypes_path = BASE_DIR / "country_prototypes_gender.npz"
    if not os.path.exists(prototypes_path):
        print(f"エラー: {prototypes_path} が見つかりません。")
        return

    print(f"{prototypes_path} から代表顔ベクトルを読み込んでいます...")
    country_prototypes_data = np.load(prototypes_path)

    for key, vec in country_prototypes_data.items():
        try:
            country, gender_str = key.rsplit("_", 1)
            gender = "man" if gender_str == "man" else "woman"
            prototypes[gender][country] = vec
        except ValueError:
            print(f"警告: キー '{key}' の形式が正しくありません。スキップします。")

    print(
        f"男性代表: {len(prototypes['man'])}件, 女性代表: {len(prototypes['woman'])}件 読み込みました。"
    )

    # 3. 国の代表画像のファイル名をロード（性別ごと）
    reps_path = BASE_DIR / "country_representatives_gender.json"
    if not os.path.exists(reps_path):
        print(f"警告: {reps_path} が見つかりません。")
        return

    print(f"{reps_path} から代表顔画像を読み込んでいます...")
    with open(reps_path, "r", encoding="utf-8") as f:
        reps_data = json.load(f)

    for key, filename in reps_data.items():
        try:
            country, gender_str = key.rsplit("_", 1)
            gender = "man" if gender_str == "man" else "woman"
            representatives[gender][country] = filename
        except ValueError:
            print(f"警告: キー '{key}' の形式が正しくありません。スキップします。")

    print(
        f"男性代表画像: {len(representatives['man'])}件, 女性代表画像: {len(representatives['woman'])}件 読み込みました。"
    )


# --- APIエンドポイント ---
@app.post("/rank-face/")
async def rank_face(
    file: UploadFile = File(...),
    gender_override: str = Form(None),  # フロントエンドからの性別指定を受け取る
):
    # 性別データがロードされているかチェック
    if model is None or not prototypes["man"] or not prototypes["woman"]:
        raise HTTPException(
            status_code=503,
            detail="モデルがまだ準備できていません。しばらくしてから再試行してください。",
        )

    # 3. アップロードされた画像を処理
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(
            status_code=400, detail="提供されたファイルは有効な画像ではありません。"
        )

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = model.get(img_rgb)

    if not faces:
        raise HTTPException(
            status_code=400, detail="画像から顔が検出できませんでした。"
        )

    user_face = faces[0]
    user_embedding = user_face.embedding

    # --- 性別の決定 ---
    # gender_overrideが指定されていればそれを使い、なければAIで判定
    if gender_override in ["man", "woman"]:
        user_gender_str = gender_override
        print(f"ユーザー指定の性別を使用: {user_gender_str}")
    else:
        # AIによる性別判定 (0: 男性, 1: 女性)
        user_gender_str = "man" if user_face.gender == 0 else "woman"
        print(f"AIが検出した性別: {user_gender_str} (年齢: {user_face.age})")

    # 判定した性別に基づいて比較対象を選択
    target_prototypes = prototypes[user_gender_str]
    target_representatives = representatives[user_gender_str]

    # 4. 類似度を計算
    similarities = {}

    # ユーザーの地域を仮定（将来的には拡張可能）
    user_continent = "AS"  # アジアと仮定

    for country, prototype_vec in target_prototypes.items():
        base_score = cosine_similarity(user_embedding, prototype_vec)

        # --- 意外性ボーナスの計算 ---
        geo_bonus = 0
        rarity_bonus = 0

        # 1. 地理ボーナス
        country_continent = get_continent(country)
        if country_continent and country_continent != user_continent:
            geo_bonus = GEO_BONUS

        # 2. 意外度ボーナス
        if country in COUNTRY_METADATA:
            rarity = COUNTRY_METADATA[country]["rarity"]
            # レア度が高いほどボーナス追加（★1はボーナス0）
            rarity_bonus = (rarity - 1) * RARITY_BONUS_UNIT

        # 最終スコア = 元のスコア + ボーナス
        final_score = base_score + geo_bonus + rarity_bonus
        similarities[country] = final_score

    # 5. ランキングを作成
    sorted_countries = sorted(
        similarities.items(), key=lambda item: item[1], reverse=True
    )

    # JSONで返せる形式に整形
    ranking_result = []
    for i, (country, score) in enumerate(sorted_countries[:5]):
        rank_data = {
            "rank": i + 1,
            "country": country,
            "score": float(score),
            "country_code": get_country_code(country),
        }
        # 1位の場合のみ、代表画像のファイル名を追加
        if i == 0 and country in target_representatives:
            rank_data["representative_image_filename"] = target_representatives[country]

        ranking_result.append(rank_data)

    # 最終的なレスポンスを作成
    response_content = {
        "detected_gender": user_gender_str,  # 実際に使用された性別
        "ranking": ranking_result,
    }

    return JSONResponse(content=response_content)


# --- フロントエンド配信（最後にマウント） ---
# APIルートなどをすべて定義した後に、残りのパスをフロントエンドに回す
# BASE_DIRを使って、実行場所によらない絶対パスを指定
app.mount("/", StaticFiles(directory=BASE_DIR / "frontend", html=True), name="frontend")


# --- メインの実行部分（デバッグ用） ---
if __name__ == "__main__":
    import uvicorn

    # サーバーを起動
    # uvicorn.run("main:app", host="0.0.0.0", port=8003)
    # 開発中はリロード機能を有効にすると便利
    uvicorn.run(
        "main:app", host="0.0.0.0", port=8003, reload=True, reload_dirs=[str(BASE_DIR)]
    )
