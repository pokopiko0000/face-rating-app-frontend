import os
import io
import pandas as pd
import numpy as np
from PIL import Image
from ultralytics import YOLO
from google.cloud import storage
from tqdm import tqdm  # tqdmを直接インポート

### ★★★ ユーザー設定項目 (ここを必ず編集してください) ★★★ ###
# 1. Google Cloud Storage (GCS) の設定
GCS_BUCKET_NAME = (
    "imagen4-faces-imagen-demo-460715"  # ★ あなたのGCSバケット名に書き換えてください
)
# GCSバケット内の画像ファイルが格納されている「フォルダパス」 (もしあれば。バケット直下なら空文字列 "")
GCS_IMAGE_FOLDER_PREFIX = ""  # 例: "generated_images_raw/" のように末尾に / を付ける

# 2. ローカルのメタデータCSVファイルのパス
METADATA_CSV_PATH = r"C:\Users\j_mar\OneDrive\ドキュメント\face\metadata.csv"  # ★ あなたのCSVファイルへの正しいパスに書き換えてください
# 例: "C:/Users/j_mar/OneDrive/ドキュメント/face/metadata.csv"

# 3. クロップされた顔画像を保存するローカルディレクトリ
LOCAL_CROPPED_FACE_DIR = r"C:\Users\j_mar\OneDrive\ドキュメント\face\cropped_images"  # この名前のフォルダがスクリプトと同じ場所に作成されます

# 4. YOLOv8モデルファイル (顔検出用)
YOLO_MODEL_PATH = "yolov8n.pt"  # 基本的な物体検出モデル (顔も検出可能)
# 顔検出特化モデルがあれば、そのパスを指定 (例: 'yolov8n-face.pt')

# 5. 顔検出の信頼度の閾値 (0.0 〜 1.0)
CONFIDENCE_THRESHOLD = 0.25

# 6. クロップする顔の最小サイズ (小さすぎる検出を除外する場合)
MIN_FACE_SIZE = (30, 30)  # (幅, 高さ) ピクセル

# 7. 処理する画像の枚数 (テスト用に制限する場合。全画像なら None)
# NUM_IMAGES_TO_PROCESS = 10 # 例: 最初の10枚だけ処理
NUM_IMAGES_TO_PROCESS = None
### ★★★ 設定項目ここまで ★★★ ###

# --- モデルのロード ---
print("YOLOv8モデルのロード中...")
try:
    detector = YOLO(YOLO_MODEL_PATH)
    print(f"YOLOv8モデル ({YOLO_MODEL_PATH}) をロードしました。")
except Exception as e:
    print(f"エラー: YOLOv8モデルのロードに失敗しました: {e}")
    print(
        "モデルファイルパスが正しいか、ultralyticsライブラリが正しくインストールされているか確認してください。"
    )
    exit()

# --- GCSクライアントの初期化 ---
print("\nGoogle Cloud Storageクライアントを初期化中...")
try:
    storage_client = storage.Client()
    bucket = storage_client.bucket(GCS_BUCKET_NAME)
    print(f"GCSバケット '{GCS_BUCKET_NAME}' に接続しました。")
except Exception as e:
    print(f"エラー: GCSへの接続に失敗しました: {e}")
    print(
        "Google Cloud SDKが正しく認証されているか、バケット名が正しいか確認してください。"
    )
    exit()

# --- メタデータCSVの読み込み ---
print(f"\nメタデータCSV '{METADATA_CSV_PATH}' を読み込み中...")
if not os.path.exists(METADATA_CSV_PATH):
    print(f"エラー: メタデータCSVファイルが見つかりません: {METADATA_CSV_PATH}")
    exit()
try:
    df_metadata = pd.read_csv(METADATA_CSV_PATH)
    # CSVの列名を確認・調整してください。
    # ここでは、画像ファイル名(拡張子なしの連番)が 'image_id' 列、
    # 国が 'country' 列、年齢が 'age' 列、性別が 'gender' 列にあると仮定します。
    # ★★★ あなたのCSVの実際の列名に合わせてください ★★★
    required_columns = [
        "index",
        "country",
        "age",
        "gender",
        "seed",
        "gcs_path",
    ]  # 仮の列名
    if not all(col in df_metadata.columns for col in required_columns):
        print(
            f"エラー: メタデータCSVに必要な列 {required_columns} が含まれていません。"
        )
        print(f"現在の列名: {df_metadata.columns.tolist()}")
        exit()
    print(f"メタデータCSVをロードしました。合計 {len(df_metadata)} レコード。")
except Exception as e:
    print(f"エラー: メタデータCSVの読み込みに失敗しました: {e}")
    exit()


# --- クロップ画像の保存先ディレクトリ作成 ---
if not os.path.exists(LOCAL_CROPPED_FACE_DIR):
    os.makedirs(LOCAL_CROPPED_FACE_DIR)
    print(f"クロップ画像保存先ディレクトリを作成しました: {LOCAL_CROPPED_FACE_DIR}")

# --- 画像処理ループ ---
print(f"\n顔検出とクロップ処理を開始します...")

# 処理対象のメタデータレコード数を決定
metadata_to_process = df_metadata
if NUM_IMAGES_TO_PROCESS is not None:
    metadata_to_process = df_metadata.head(NUM_IMAGES_TO_PROCESS)
    print(f"テストのため、最初の {NUM_IMAGES_TO_PROCESS} 画像のみを処理します。")

processed_count = 0
face_detected_count = 0

for index, row in tqdm(
    metadata_to_process.iterrows(), total=len(metadata_to_process), desc="画像処理中"
):
    try:
        # CSVから情報を取得 (★★★ 列名を実際のCSVに合わせてください ★★★)
        image_id_str = str(row["index"]).zfill(4)  # 例: 1 -> "0001"
        country = str(row["country"])
        age = str(row["age"])
        gender = str(row["gender"])
        # seed = str(row['seed']) # もしseedもファイル名に使うなら

        image_filename_on_gcs = f"{image_id_str}.png"
        gcs_blob_path = os.path.join(
            GCS_IMAGE_FOLDER_PREFIX, image_filename_on_gcs
        ).replace(
            "\\", "/"
        )  # GCSパスは / 区切り

        blob = bucket.blob(gcs_blob_path)
        if not blob.exists():
            # print(f"  警告: GCSに画像ファイルが見つかりません: gs://{GCS_BUCKET_NAME}/{gcs_blob_path}")
            continue

        # GCSから画像をメモリにダウンロード
        image_bytes = blob.download_as_bytes()
        img_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_np = np.array(img_pil)

        # YOLOv8で顔検出
        results = detector(img_np, conf=CONFIDENCE_THRESHOLD, verbose=False)

        if results and len(results[0].boxes) > 0:
            detected_faces_in_image = 0
            # 複数の顔が検出された場合、最大のものを選択するか、全て保存するか選択
            # ここでは最大の顔を1つだけクロップする例
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

            if len(areas) == 0:  # 信頼度でフィルタされた結果、ボックスがない場合
                # print(f"  情報: {image_filename_on_gcs} - 信頼度閾値以上の顔は検出されませんでした。")
                continue

            largest_face_idx = areas.argmax()
            x1, y1, x2, y2 = boxes[largest_face_idx]

            face_width = x2 - x1
            face_height = y2 - y1

            if face_width >= MIN_FACE_SIZE[0] and face_height >= MIN_FACE_SIZE[1]:
                face_crop_np = img_np[y1:y2, x1:x2]
                face_crop_pil = Image.fromarray(face_crop_np)

                # クロップ画像の保存ファイル名 (元の情報を含める)
                # 例: Japan_30_Female_0001_crop0.png
                # もし1画像から複数の顔を保存する場合は crop_idx をループさせる
                cropped_filename = f"{country}_{age}_{gender}_{image_id_str}_crop0.png"
                local_save_path = os.path.join(LOCAL_CROPPED_FACE_DIR, cropped_filename)

                face_crop_pil.save(local_save_path)
                # print(f"  成功: {image_filename_on_gcs} -> クロップ画像を保存: {local_save_path}")
                detected_faces_in_image += 1
            # else:
            # print(f"  情報: {image_filename_on_gcs} - 検出された顔が小さすぎます ({face_width}x{face_height})。")

            if detected_faces_in_image > 0:
                face_detected_count += 1
        # else:
        # print(f"  情報: {image_filename_on_gcs} - 顔が検出されませんでした。")

        processed_count += 1

    except Exception as e:
        print(
            f"エラー: 画像ID {row.get('image_id', 'N/A')} の処理中に問題が発生しました: {e}"
        )
        import traceback

        traceback.print_exc()  # 詳細なエラー内容を表示
        continue  # エラーが発生しても次の画像の処理を試みる

print(f"\n--- 処理完了 ---")
print(f"処理対象とした画像数: {processed_count}")
print(f"顔が検出されクロップされた画像数: {face_detected_count}")
print(f"クロップ画像は '{LOCAL_CROPPED_FACE_DIR}' フォルダに保存されました。")
