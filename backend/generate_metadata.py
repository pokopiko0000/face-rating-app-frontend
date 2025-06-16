import pandas as pd
import numpy as np
import os

# --- 設定 ---
# 国のリスト (特殊文字をASCIIに置換)
countries = [
    "Afghanistan",
    "Aland Islands",
    "Albania",
    "Algeria",
    "American Samoa",
    "Andorra",
    "Angola",
    "Anguilla",
    "Antarctica",
    "Antigua and Barbuda",
    "Argentina",
    "Armenia",
    "Aruba",
    "Australia",
    "Austria",
    "Azerbaijan",
    "Bahamas",
    "Bahrain",
    "Bangladesh",
    "Barbados",
    "Belarus",
    "Belgium",
    "Belize",
    "Benin",
    "Bermuda",
    "Bhutan",
    "Bolivia (Plurinational State of)",
    "Bonaire, Sint Eustatius and Saba",
    "Bosnia and Herzegovina",
    "Botswana",
    "Bouvet Island",
    "Brazil",
    "British Indian Ocean Territory",
    "Brunei Darussalam",
    "Bulgaria",
    "Burkina Faso",
    "Burundi",
    "Cabo Verde",
    "Cambodia",
    "Cameroon",
    "Canada",
    "Cayman Islands",
    "Central African Republic",
    "Chad",
    "Chile",
    "China",
    "Christmas Island",
    "Cocos (Keeling) Islands",
    "Colombia",
    "Comoros",
    "Congo",
    "Congo (Democratic Republic of the)",
    "Cook Islands",
    "Costa Rica",
    "Cote d'Ivoire",
    "Croatia",
    "Cuba",
    "Curacao",
    "Cyprus",
    "Czechia",
    "Denmark",
    "Djibouti",
    "Dominica",
    "Dominican Republic",
    "Ecuador",
    "Egypt",
    "El Salvador",
    "Equatorial Guinea",
    "Eritrea",
    "Estonia",
    "Eswatini",
    "Ethiopia",
    "Falkland Islands (Malvinas)",
    "Faroe Islands",
    "Fiji",
    "Finland",
    "France",
    "French Guiana",
    "French Polynesia",
    "French Southern Territories",
    "Gabon",
    "Gambia",
    "Georgia",
    "Germany",
    "Ghana",
    "Gibraltar",
    "Greece",
    "Greenland",
    "Grenada",
    "Guadeloupe",
    "Guam",
    "Guatemala",
    "Guernsey",
    "Guinea",
    "Guinea-Bissau",
    "Guyana",
    "Haiti",
    "Heard Island and McDonald Islands",
    "Holy See",
    "Honduras",
    "Hong Kong",
    "Hungary",
    "Iceland",
    "India",
    "Indonesia",
    "Iran (Islamic Republic of)",
    "Iraq",
    "Ireland",
    "Isle of Man",
    "Israel",
    "Italy",
    "Jamaica",
    "Japan",
    "Jersey",
    "Jordan",
    "Kazakhstan",
    "Kenya",
    "Kiribati",
    "Korea (Democratic People's Republic of)",
    "Korea (Republic of)",
    "Kuwait",
    "Kyrgyzstan",
    "Lao People's Democratic Republic",
    "Latvia",
    "Lebanon",
    "Lesotho",
    "Liberia",
    "Libya",
    "Liechtenstein",
    "Lithuania",
    "Luxembourg",
    "Macao",
    "Madagascar",
    "Malawi",
    "Malaysia",
    "Maldives",
    "Mali",
    "Malta",
    "Marshall Islands",
    "Martinique",
    "Mauritania",
    "Mauritius",
    "Mayotte",
    "Mexico",
    "Micronesia (Federated States of)",
    "Moldova (Republic of)",
    "Monaco",
    "Mongolia",
    "Montenegro",
    "Montserrat",
    "Morocco",
    "Mozambique",
    "Myanmar",
    "Namibia",
    "Nauru",
    "Nepal",
    "Netherlands",
    "New Caledonia",
    "New Zealand",
    "Nicaragua",
    "Niger",
    "Nigeria",
    "Niue",
    "Norfolk Island",
    "North Macedonia",
    "Northern Mariana Islands",
    "Norway",
    "Oman",
    "Pakistan",
    "Palau",
    "Palestine, State of",
    "Panama",
    "Papua New Guinea",
    "Paraguay",
    "Peru",
    "Philippines",
    "Pitcairn",
    "Poland",
    "Portugal",
    "Puerto Rico",
    "Qatar",
    "Reunion",
    "Romania",
    "Russian Federation",
    "Rwanda",
    "Saint Barthelemy",
    "Saint Helena, Ascension and Tristan da Cunha",
    "Saint Kitts and Nevis",
    "Saint Lucia",
    "Saint Martin (French part)",
    "Saint Pierre and Miquelon",
    "Saint Vincent and the Grenadines",
    "Samoa",
    "San Marino",
    "Sao Tome and Principe",
    "Saudi Arabia",
    "Senegal",
    "Serbia",
    "Seychelles",
    "Sierra Leone",
    "Singapore",
    "Sint Maarten (Dutch part)",
    "Slovakia",
    "Slovenia",
    "Solomon Islands",
    "Somalia",
    "South Africa",
    "South Georgia and the South Sandwich Islands",
    "South Sudan",
    "Spain",
    "Sri Lanka",
    "Sudan",
    "Suriname",
    "Svalbard and Jan Mayen",
    "Sweden",
    "Switzerland",
    "Syrian Arab Republic",
    "Taiwan (Province of China)",
    "Tajikistan",
    "Tanzania, United Republic of",
    "Thailand",
    "Timor-Leste",
    "Togo",
    "Tokelau",
    "Tonga",
    "Trinidad and Tobago",
    "Tunisia",
    "Turkey",
    "Turkmenistan",
    "Turks and Caicos Islands",
    "Tuvalu",
    "Uganda",
    "Ukraine",
    "United Arab Emirates",
    "United Kingdom of Great Britain and Northern Ireland",
    "United States of America",
    "United States Minor Outlying Islands",
    "Uruguay",
    "Uzbekistan",
    "Vanuatu",
    "Venezuela (Bolivarian Republic of)",
    "Viet Nam",
    "Virgin Islands (British)",
    "Virgin Islands (U.S.)",
    "Wallis and Futuna",
    "Western Sahara",
    "Yemen",
    "Zambia",
    "Zimbabwe",
    "Kosovo",
]

# 生成するCSVファイルのパス (crop_face.py の設定に合わせる)
OUTPUT_CSV_PATH = r"C:\Users\j_mar\OneDrive\ドキュメント\face\metadata.csv"

# country.md ファイルのパス
COUNTRY_MD_PATH = "backend/country.md"

# GCSバケット名 (crop_face.py から)
GCS_BUCKET_NAME = "imagen4-faces-imagen-demo-460715"

# 生成する画像の総数
TOTAL_IMAGES = 6000

# 1カ国あたりの画像数
IMAGES_PER_COUNTRY = 24


# --- データ生成ロジック ---
def generate_metadata():
    """
    指定されたルールに基づいてメタデータCSVを生成します。
    """
    data = []

    for i in range(1, TOTAL_IMAGES + 1):
        # index: 1から6000
        image_index = i

        # 国の決定 (24枚ごとに変わる)
        country_index = (i - 1) // IMAGES_PER_COUNTRY
        country = countries[country_index]

        # 年齢と性別の決定 (24枚のサイクル内での位置)
        pos_in_cycle = (i - 1) % IMAGES_PER_COUNTRY

        if 0 <= pos_in_cycle < 6:  # 最初の6枚
            age = 20
            gender = "Male"
        elif 6 <= pos_in_cycle < 12:  # 次の6枚
            age = 20
            gender = "Female"
        elif 12 <= pos_in_cycle < 18:  # 次の6枚
            age = 30
            gender = "Male"
        else:  # 最後の6枚
            age = 30
            gender = "Female"

        # seed: ランダムな5桁の整数
        seed = np.random.randint(10000, 100000)

        # GCSパスの生成
        image_id_str = str(image_index).zfill(4)
        gcs_path = f"gs://{GCS_BUCKET_NAME}/{image_id_str}.png"

        # `crop_face.py` が必要とする列名に合わせてデータを追加
        # `crop_face.py`の`image_id_str = str(row['index']).zfill(4)`という行から、
        # CSVの列名は 'index' であると判断
        data.append(
            {
                "index": image_index,
                "country": country,
                "age": age,
                "gender": gender,
                "seed": seed,
                "gcs_path": gcs_path,
            }
        )

    df = pd.DataFrame(data)

    # ディレクトリが存在しない場合は作成
    output_dir = os.path.dirname(OUTPUT_CSV_PATH)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"ディレクトリを作成しました: {output_dir}")

    # CSVファイルに保存
    df.to_csv(OUTPUT_CSV_PATH, index=False, encoding="utf-8")
    print(f"メタデータファイルを作成しました: {OUTPUT_CSV_PATH}")

    # country.md ファイルを更新
    update_country_md()


def update_country_md():
    """
    country.md ファイルを更新された国リストで上書きします。
    """
    content = "countries = [\n"
    line = "    "
    for i, country in enumerate(countries):
        line += f'"{country}", '
        if (i + 1) % 5 == 0 and i < len(countries) - 1:
            content += line.rstrip() + "\n"
            line = "    "

    content += line.rstrip().rstrip(",")
    content += "\n]\n"

    with open(COUNTRY_MD_PATH, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"国リストファイルを更新しました: {COUNTRY_MD_PATH}")


if __name__ == "__main__":
    generate_metadata()
