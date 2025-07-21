import json
import numpy as np
import re
from pathlib import Path
from countryinfo import CountryInfo
from pycountry_convert import (
    country_alpha2_to_continent_code,
    country_name_to_country_alpha2,
)

# --- Path Settings ---
BASE_DIR = Path(__file__).resolve().parent
PROTOTYPES_PATH = BASE_DIR / "country_prototypes_gender.npz"
OUTPUT_PATH = BASE_DIR / "country_metadata.json"

# --- Helper Functions ---


def normalize_country_name(name: str) -> str:
    """countryinfoライブラリが認識しやすいように国名を正規化する"""
    # 特定の国名の変換ルールを先に適用
    replacements = {
        "Bolivia (Plurinational State of)": "Bolivia",
        "Brunei Darussalam": "Brunei",
        "Cabo Verde": "Cape Verde",
        "Congo (Democratic Republic of the)": "Congo",
        "Congo": "Congo",
        "Czechia": "Czech Republic",
        "Iran (Islamic Republic of)": "Iran",
        "Korea (Republic of)": "South Korea",
        "Korea (Democratic People's Republic of)": "North Korea",
        "Lao People's Democratic Republic": "Laos",
        "Micronesia (Federated States of)": "Micronesia",
        "Moldova (Republic of)": "Moldova",
        "Russian Federation": "Russia",
        "Syrian Arab Republic": "Syria",
        "Tanzania, United Republic of": "Tanzania",
        "Timor-Leste": "East Timor",
        "United Kingdom of Great Britain and Northern Ireland": "United Kingdom",
        "United States of America": "United States",
        "Venezuela (Bolivarian Republic of)": "Venezuela",
        "Viet Nam": "Vietnam",
        "Holy See (Vatican City State)": "Vatican",
        "Saint Martin (French part)": "Saint Martin",
        "Sint Maarten (Dutch part)": "Sint Maarten",
    }
    name = replacements.get(name, name)
    # 括弧（と中の文字）を削除
    name = re.sub(r"\(.*\)", "", name).strip()
    return name


def get_rarity_by_population(population: int) -> int:
    """人口から意外度（★1〜5）を計算する"""
    if population < 1_000_000:
        return 5
    elif population < 10_000_000:
        return 4
    elif population < 50_000_000:
        return 3
    elif population < 100_000_000:
        return 2
    else:
        return 1


def get_continent(country_name: str):
    """国名から大陸コードを取得する"""
    try:
        country_alpha2 = country_name_to_country_alpha2(country_name)
        return country_alpha2_to_continent_code(country_alpha2)
    except:
        # 特殊なケースに対応
        if country_name == "Kosovo":
            return "EU"
        return None


def main():
    """
    国別のプロトタイプデータから国リストを抽出し、
    各国のメタデータ（大陸、意外度）を計算してJSONファイルに保存する。
    """
    print(f"Loading prototypes from: {PROTOTYPES_PATH}")
    if not PROTOTYPES_PATH.exists():
        print(f"Error: Prototypes file not found at {PROTOTYPES_PATH}")
        return

    country_prototypes_data = np.load(PROTOTYPES_PATH)

    # "Japan_man"のようなキーから国名 "Japan" を抽出
    all_countries = set()
    for key in country_prototypes_data.keys():
        try:
            country_name = key.rsplit("_", 1)[0]
            all_countries.add(country_name)
        except IndexError:
            continue

    print(f"Found {len(all_countries)} unique countries.")

    country_metadata = {}
    successful_count = 0
    failed_countries = []

    for country_name in sorted(list(all_countries)):
        try:
            # 1. 大陸情報を取得
            continent = get_continent(country_name)

            # 2. 人口情報を取得して意外度を計算
            normalized_name = normalize_country_name(country_name)
            info = CountryInfo(normalized_name).info()
            population = info.get("population")

            if population:
                rarity = get_rarity_by_population(population)
                country_metadata[country_name] = {
                    "continent": continent,
                    "rarity": rarity,
                }
                successful_count += 1
            else:
                # 人口が取得できない場合はデフォルト値
                country_metadata[country_name] = {"continent": continent, "rarity": 1}
                failed_countries.append(
                    f"{country_name} -> {normalized_name} (Population not found)"
                )

        except Exception as e:
            # その他のエラーでもデフォルト値
            country_metadata[country_name] = {
                "continent": get_continent(country_name),
                "rarity": 1,
            }
            failed_countries.append(
                f"{country_name} -> {normalized_name} ({type(e).__name__})"
            )

    print(f"\nSuccessfully processed {successful_count} countries.")
    if failed_countries:
        print(
            "\nCould not retrieve full data for the following countries (default values used):"
        )
        for country in failed_countries:
            print(f"- {country}")

    # 生成したデータをJSONファイルに保存
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(country_metadata, f, indent=4, ensure_ascii=False)

    print(
        f"\n✅ Successfully generated and saved metadata for {len(country_metadata)} countries to {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
