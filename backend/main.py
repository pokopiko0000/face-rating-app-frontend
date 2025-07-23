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
import random
from pycountry_convert import (
    country_alpha2_to_continent_code,
    country_name_to_country_alpha2,
)
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

# Debug環境変数
print(f"DEBUG: CORS_ORIGINS env var = '{os.getenv('CORS_ORIGINS', 'NOT SET')}'")
print(f"DEBUG: CORS_ORIGINS type = {type(os.getenv('CORS_ORIGINS'))}")
print(f"DEBUG: CORS_ORIGINS repr = {repr(os.getenv('CORS_ORIGINS'))}")

# Import configuration management
from config import settings

# Import constants
from constants import COUNTRY_FLAGS, COUNTRY_NAME_JP

# Import utilities  
from utils.similarity import cosine_similarity


# --- 定数と設定 ---

# このファイルの場所を基準にパスを構築
BASE_DIR = Path(__file__).resolve().parent.parent

# --- グローバル変数 ---
# モデルとプロトタイプは起動時に一度だけロードする
model: Optional[insightface.app.FaceAnalysis] = None
# 性別ごとにデータを保持する
prototypes: Dict[str, Dict[str, np.ndarray]] = {"man": {}, "woman": {}}
representatives: Dict[str, Dict[str, str]] = {"man": {}, "woman": {}}
country_code_cache: Dict[str, Optional[str]] = (
    {}
)  # 国コード検索を高速化するためのキャッシュ
country_metadata_g: Dict[str, Dict[str, Any]] = (
    {}
)  # 国ごとのメタデータ（大陸、意外性）をキャッシュ


# --- ヘルパー関数 ---
def get_country_flag(country_name):
    """国名から国旗絵文字を取得"""
    return COUNTRY_FLAGS.get(country_name, "🏳️")


def get_country_name_japanese(country_name_english):
    """英語の国名から日本語の国名を取得"""
    return COUNTRY_NAME_JP.get(country_name_english, country_name_english)


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
        # 事前に計算したメタデータにあればそれを使う
        if country_name in country_metadata_g:
            return country_metadata_g[country_name].get("continent")

        # なければライブラリで変換を試みる（フォールバック）
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
app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    debug=settings.debug,
)


# --- ヘルスチェック用エンドポイント ---
@app.api_route("/", methods=["GET", "HEAD"])
def read_root():
    return {"status": "ok"}


# --- CORSミドルウェアの設定 ---
app.add_middleware(
    CORSMiddleware,
    **settings.get_cors_config()
)


# --- 起動時イベント ---
@app.on_event("startup")
def load_models():
    global model, prototypes, representatives, country_metadata_g
    # 1. 顔分析モデルを準備
    print("顔分析モデルを準備しています...")
    face_config = settings.get_face_analysis_config()
    model = insightface.app.FaceAnalysis(providers=face_config["providers"])
    model.prepare(ctx_id=0, det_thresh=face_config["det_thresh"], det_size=face_config["det_size"])
    print("モデルの準備が完了しました。")

    # 2. 国の代表顔ベクトルをロード（性別ごと）
    prototypes_path = BASE_DIR / "backend" / "data" / "country_prototypes_gender.npz"
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

    print(f"男性代表: {len(prototypes['man'])}件, 女性代表: {len(prototypes['woman'])}件 読み込みました。")

    # 3. 国の代表画像のファイル名をロード（性別ごと）
    reps_path = BASE_DIR / "backend" / "data" / "country_representatives_gender.json"
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

    print(f"男性代表画像: {len(representatives['man'])}件, 女性代表画像: {len(representatives['woman'])}件 読み込みました。")

    # 4. 事前に生成した国別メタデータをファイルから読み込む
    metadata_path = BASE_DIR / "backend" / "data" / "country_metadata.json"
    if not metadata_path.exists():
        print(f"警告: {metadata_path} が見つかりません。ボーナス計算は無効になります。")
        return

    print(f"{metadata_path} から国別メタデータを読み込んでいます...")
    with open(metadata_path, "r", encoding="utf-8") as f:
        country_metadata_g = json.load(f)
    print(f"{len(country_metadata_g)}カ国分のメタデータを準備しました。")


async def _get_face_details(file: UploadFile) -> Any:
    """アップロードされた画像から顔の情報を抽出する"""
    if model is None:
        raise HTTPException(status_code=503, detail="モデルが初期化されていません。")

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="提供されたファイルは有効な画像ではありません。")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = model.get(img_rgb)

    if not faces:
        raise HTTPException(status_code=400, detail="画像から顔が検出できませんでした。")

    return faces[0]


def _calculate_ranking(user_embedding: np.ndarray, gender_str: str) -> List[Tuple[str, float]]:
    """顔の特徴量ベクトルと性別から国別ランキングを計算する"""
    target_prototypes = prototypes[gender_str]
    target_representatives = representatives[gender_str]

    available_countries = set(target_prototypes.keys()) & set(target_representatives.keys())
    print(f"分析対象国数: {len(available_countries)}カ国（性別: {gender_str}）")

    similarities = {}
    user_continent = "AS"  # アジアと仮定

    for country in available_countries:
        prototype_vec = target_prototypes[country]
        base_score = cosine_similarity(user_embedding, prototype_vec)

        geo_bonus = 0.0
        rarity_bonus = 0.0
        metadata = country_metadata_g.get(country)
        if metadata:
            country_continent = metadata.get("continent")
            if country_continent and country_continent != user_continent:
                geo_bonus = settings.geo_bonus

            rarity = metadata.get("rarity", 1)
            rarity_bonus = (rarity - 1) * settings.rarity_bonus_unit

        final_score = base_score + geo_bonus + rarity_bonus
        similarities[country] = final_score

    adjusted_scores = {}
    if similarities:
        max_original_score = max(similarities.values())
        target_top_score = random.randint(85, 99)
        if max_original_score > 0:
            for country, score in similarities.items():
                adjusted_scores[country] = (score / max_original_score) * target_top_score
        else:
            adjusted_scores = {country: 0 for country in similarities.keys()}

    return sorted(adjusted_scores.items(), key=lambda item: item[1], reverse=True)


# --- APIエンドポイント ---
@app.post("/analyze")
async def analyze_face(file: UploadFile = File(...), gender: str = Form(...)):
    """フロントエンド用の顔分析エンドポイント"""
    if model is None or not prototypes["man"] or not prototypes["woman"]:
        raise HTTPException(status_code=503, detail="モデルがまだ準備できていません。")

    user_face = await _get_face_details(file)
    user_embedding = user_face.embedding
    user_gender_str = "man" if gender == "male" else "woman"

    sorted_countries = _calculate_ranking(user_embedding, user_gender_str)

    ranking_result = []
    for country, score in sorted_countries[:10]:
        ranking_result.append({
            "country": get_country_name_japanese(country),
            "country_english": country,
            "similarity": float(score),
            "country_code": get_country_code(country),
        })

    # 1位の国の代表画像ファイル名を取得
    top_country_image_url = None
    if sorted_countries:
        top_country = sorted_countries[0][0]
        target_reps = representatives[user_gender_str]
        if top_country in target_reps:
            image_filename = target_reps[top_country]
            top_country_image_url = f"{settings.r2_public_url}/{image_filename}"

    return JSONResponse(content={
        "ranking": ranking_result,
        "top_country_image_url": top_country_image_url,
    })


@app.get("/health")
def health_check():
    """Fly.io用のヘルスチェックエンドポイント"""
    return {"status": "healthy", "message": "Face rating backend is running"}


@app.get("/debug/cors")
def debug_cors():
    """CORS設定のデバッグ情報を返す"""
    cors_config = settings.get_cors_config()
    return {
        "environment": settings.environment.value,
        "cors_origins_env": os.getenv("CORS_ORIGINS", "Not set"),
        "cors_origins_parsed": settings.cors_origins,
        "cors_config": cors_config,
        "api_url": settings.api_url,
    }


# --- メインの実行部分（デバッグ用） ---
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8003))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True, reload_dirs=[str(BASE_DIR)])
