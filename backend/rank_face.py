import cv2
import insightface
import numpy as np
import os


def cosine_similarity(v1, v2):
    """2つのベクトル間のコサイン類似度を計算する"""
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    return dot_product / (norm_v1 * norm_v2)


def main():
    # 1. 顔分析モデルを準備
    print("顔分析モデルを準備しています...")
    model = insightface.app.FaceAnalysis(providers=["CPUExecutionProvider"])
    model.prepare(ctx_id=0, det_thresh=0.1, det_size=(640, 640))
    print("モデルの準備が完了しました。")

    # 2. 国の代表顔ベクトルをロード
    prototypes_path = "face0/country_prototypes.npz"
    if not os.path.exists(prototypes_path):
        print(f"エラー: {prototypes_path} が見つかりません。")
        print("まず `generate_prototypes.py` を実行してください。")
        return

    print(f"{prototypes_path} から代表顔ベクトルを読み込んでいます...")
    country_prototypes_data = np.load(prototypes_path)
    country_prototypes = {
        country: vec for country, vec in country_prototypes_data.items()
    }
    print(f"{len(country_prototypes)}カ国分読み込みました。")

    # 3. ユーザーの画像を処理
    user_image_path = "face0/cropped_images/Kosovo_30_woman_6000_crop0.png"
    print(f"入力画像を処理中: {user_image_path}")

    img = cv2.imread(user_image_path)
    if img is None:
        print(f"エラー: 画像が読み込めませんでした: {user_image_path}")
        return

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = model.get(img_rgb)

    if not faces:
        print("エラー: 入力画像から顔が検出されませんでした。")
        return

    user_embedding = faces[0].embedding
    print("入力画像の顔ベクトルを抽出しました。")

    # 4. 類似度を計算
    print("\n各国の代表顔との類似度を計算しています...")
    similarities = {}
    for country, prototype_vec in country_prototypes.items():
        similarity = cosine_similarity(user_embedding, prototype_vec)
        similarities[country] = similarity

    # 5. ランキングを作成して表示
    # 類似度が高い順（降順）にソート
    sorted_countries = sorted(
        similarities.items(), key=lambda item: item[1], reverse=True
    )

    print("\n--- 魅力度ランキング TOP 5 ---")
    for i, (country, score) in enumerate(sorted_countries[:5]):
        print(f"{i+1}位: {country} (類似度: {score:.4f})")
    print("----------------------------")


if __name__ == "__main__":
    main()
