import os
import json
from collections import defaultdict
import random


def regenerate_representatives():
    """
    cropped_imagesディレクトリ内の実際のファイル名に基づいて、
    country_representatives_gender.jsonを再生成する。
    ファイル名の性別表記は 'Male'/'Female' を想定。
    """
    image_dir = "../cropped_images"
    output_json_path = "country_representatives_gender.json"

    if not os.path.isdir(image_dir):
        print(f"エラー: ディレクトリが見つかりません: {image_dir}")
        print("スクリプトは backend ディレクトリから実行してください。")
        return

    # 国、性別ごとにファイル名をグループ化
    # 例: grouped_files['Japan']['Male'] = ['Japan_20_Male_1234.png', ...]
    grouped_files = defaultdict(lambda: defaultdict(list))

    print(f"'{image_dir}' から画像をスキャンしています...")
    for filename in os.listdir(image_dir):
        if filename.endswith(".png"):
            try:
                parts = filename.split("_")
                # Kosovo_30_Male_5994_crop0.png -> ['Kosovo', '30', 'Male', '5994', 'crop0.png']
                country = parts[0]
                gender = parts[2]  # 'Male' or 'Female'

                if gender in ["Male", "Female"]:
                    grouped_files[country][gender].append(filename)

            except IndexError:
                # ファイル名が期待する形式でない場合はスキップ
                print(f"警告: ファイル名の形式が不正です。スキップします: {filename}")
                continue

    print(f"{len(grouped_files)}カ国のデータを検出しました。")

    # 国ごと、性別ごとに代表を1つ選出
    representatives = {}
    for country, genders in grouped_files.items():
        if "Male" in genders:
            # 男性用の代表をランダムに1つ選ぶ
            chosen_file = random.choice(genders["Male"])
            # jsonのキーは 'man' を使う
            key = f"{country}_man"
            representatives[key] = chosen_file

        if "Female" in genders:
            # 女性用の代表をランダムに1つ選ぶ
            chosen_file = random.choice(genders["Female"])
            # jsonのキーは 'woman' を使う
            key = f"{country}_woman"
            representatives[key] = chosen_file

    # JSONファイルに保存
    try:
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(representatives, f, indent=4, ensure_ascii=False)
        print(f"'{output_json_path}' が正常に再生成されました。")
        print(f"合計 {len(representatives)} 件の代表画像が保存されました。")
    except IOError as e:
        print(f"エラー: ファイルの書き込みに失敗しました: {e}")


if __name__ == "__main__":
    regenerate_representatives()
