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


# --- 定数と設定 ---

# このファイルの場所を基準にパスを構築
BASE_DIR = Path(__file__).resolve().parent.parent

# ボーナス設定
GEO_BONUS = 0.05  # 違う大陸だった場合のボーナス (5% に縮小)
RARITY_BONUS_UNIT = 0.01  # 意外度★1つあたりのボーナス (1% に縮小)

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


# --- 国旗の絵文字マッピング ---
# 250カ国のデータを網羅
COUNTRY_FLAGS = {
    "Afghanistan": "🇦🇫",
    "Aland Islands": "🇦🇽",
    "Albania": "🇦🇱",
    "Algeria": "🇩🇿",
    "American Samoa": "🇦🇸",
    "Andorra": "🇦🇩",
    "Angola": "🇦🇴",
    "Anguilla": "🇦🇮",
    "Antarctica": "🇦🇶",
    "Antigua and Barbuda": "🇦🇬",
    "Argentina": "🇦🇷",
    "Armenia": "🇦🇲",
    "Aruba": "🇦🇼",
    "Australia": "🇦🇺",
    "Austria": "🇦🇹",
    "Azerbaijan": "🇦🇿",
    "Bahamas": "🇧🇸",
    "Bahrain": "🇧🇭",
    "Bangladesh": "🇧🇩",
    "Barbados": "🇧🇧",
    "Belarus": "🇧🇾",
    "Belgium": "🇧🇪",
    "Belize": "🇧🇿",
    "Benin": "🇧🇯",
    "Bermuda": "🇧🇲",
    "Bhutan": "🇧🇹",
    "Bolivia (Plurinational State of)": "🇧🇴",
    "Bonaire, Sint Eustatius and Saba": "🇧🇶",
    "Bosnia and Herzegovina": "🇧🇦",
    "Botswana": "🇧🇼",
    "Bouvet Island": "🇧🇻",
    "Brazil": "🇧🇷",
    "British Indian Ocean Territory": "🇮🇴",
    "Brunei Darussalam": "🇧🇳",
    "Bulgaria": "🇧🇬",
    "Burkina Faso": "🇧🇫",
    "Burundi": "🇧🇮",
    "Cabo Verde": "🇨🇻",
    "Cambodia": "🇰🇭",
    "Cameroon": "🇨🇲",
    "Canada": "🇨🇦",
    "Cayman Islands": "🇰🇾",
    "Central African Republic": "🇨🇫",
    "Chad": "🇹🇩",
    "Chile": "🇨🇱",
    "China": "🇨🇳",
    "Christmas Island": "🇨🇽",
    "Cocos (Keeling) Islands": "🇨🇨",
    "Colombia": "🇨🇴",
    "Comoros": "🇰🇲",
    "Congo": "🇨🇬",
    "Congo (Democratic Republic of the)": "🇨🇩",
    "Cook Islands": "🇨🇰",
    "Costa Rica": "🇨🇷",
    "Cote d'Ivoire": "🇨🇮",
    "Croatia": "🇭🇷",
    "Cuba": "🇨🇺",
    "Curacao": "🇨🇼",
    "Cyprus": "🇨🇾",
    "Czechia": "🇨🇿",
    "Denmark": "🇩🇰",
    "Djibouti": "🇩🇯",
    "Dominica": "🇩🇲",
    "Dominican Republic": "🇩🇴",
    "Ecuador": "🇪🇨",
    "Egypt": "🇪🇬",
    "El Salvador": "🇸🇻",
    "Equatorial Guinea": "🇬🇶",
    "Eritrea": "🇪🇷",
    "Estonia": "🇪🇪",
    "Eswatini": "🇸🇿",
    "Ethiopia": "🇪🇹",
    "Falkland Islands (Malvinas)": "🇫🇰",
    "Faroe Islands": "🇫🇴",
    "Fiji": "🇫🇯",
    "Finland": "🇫🇮",
    "France": "🇫🇷",
    "French Guiana": "🇬🇫",
    "French Polynesia": "🇵🇫",
    "French Southern Territories": "🇹🇫",
    "Gabon": "🇬🇦",
    "Gambia": "🇬🇲",
    "Georgia": "🇬🇪",
    "Germany": "🇩🇪",
    "Ghana": "🇬🇭",
    "Gibraltar": "🇬🇮",
    "Greece": "🇬🇷",
    "Greenland": "🇬🇱",
    "Grenada": "🇬🇩",
    "Guadeloupe": "🇬🇵",
    "Guam": "🇬🇺",
    "Guatemala": "🇬🇹",
    "Guernsey": "🇬🇬",
    "Guinea": "🇬🇳",
    "Guinea-Bissau": "🇬🇼",
    "Guyana": "🇬🇾",
    "Haiti": "🇭🇹",
    "Heard Island and McDonald Islands": "🇭🇲",
    "Holy See": "🇻🇦",
    "Honduras": "🇭🇳",
    "Hong Kong": "🇭🇰",
    "Hungary": "🇭🇺",
    "Iceland": "🇮🇸",
    "India": "🇮🇳",
    "Indonesia": "🇮🇩",
    "Iran (Islamic Republic of)": "🇮🇷",
    "Iraq": "🇮🇶",
    "Ireland": "🇮🇪",
    "Isle of Man": "🇮🇲",
    "Israel": "🇮🇱",
    "Italy": "🇮🇹",
    "Jamaica": "🇯🇲",
    "Japan": "🇯🇵",
    "Jersey": "🇯🇪",
    "Jordan": "🇯🇴",
    "Kazakhstan": "🇰🇿",
    "Kenya": "🇰🇪",
    "Kiribati": "🇰🇮",
    "Korea (Democratic People's Republic of)": "🇰🇵",
    "Korea (Republic of)": "🇰🇷",
    "Kosovo": "🇽🇰",
    "Kuwait": "🇰🇼",
    "Kyrgyzstan": "🇰🇬",
    "Lao People's Democratic Republic": "🇱🇦",
    "Latvia": "🇱🇻",
    "Lebanon": "🇱🇧",
    "Lesotho": "🇱🇸",
    "Liberia": "🇱🇷",
    "Libya": "🇱🇾",
    "Liechtenstein": "🇱🇮",
    "Lithuania": "🇱🇹",
    "Luxembourg": "🇱🇺",
    "Macao": "🇲🇴",
    "Madagascar": "🇲🇬",
    "Malawi": "🇲🇼",
    "Malaysia": "🇲🇾",
    "Maldives": "🇲🇻",
    "Mali": "🇲🇱",
    "Malta": "🇲🇹",
    "Marshall Islands": "🇲🇭",
    "Martinique": "🇲🇶",
    "Mauritania": "🇲🇷",
    "Mauritius": "🇲🇺",
    "Mayotte": "🇾🇹",
    "Mexico": "🇲🇽",
    "Micronesia (Federated States of)": "🇫🇲",
    "Moldova (Republic of)": "🇲🇩",
    "Monaco": "🇲🇨",
    "Mongolia": "🇲🇳",
    "Montenegro": "🇲🇪",
    "Montserrat": "🇲🇸",
    "Morocco": "🇲🇦",
    "Mozambique": "🇲🇿",
    "Myanmar": "🇲🇲",
    "Namibia": "🇳🇦",
    "Nauru": "🇳🇷",
    "Nepal": "🇳🇵",
    "Netherlands": "🇳🇱",
    "New Caledonia": "🇳🇨",
    "New Zealand": "🇳🇿",
    "Nicaragua": "🇳🇮",
    "Niger": "🇳🇪",
    "Nigeria": "🇳🇬",
    "Niue": "🇳🇺",
    "Norfolk Island": "🇳🇫",
    "North Macedonia": "🇲🇰",
    "Northern Mariana Islands": "🇲🇵",
    "Norway": "🇳🇴",
    "Oman": "🇴🇲",
    "Pakistan": "🇵🇰",
    "Palau": "🇵🇼",
    "Palestine, State of": "🇵🇸",
    "Panama": "🇵🇦",
    "Papua New Guinea": "🇵🇬",
    "Paraguay": "🇵🇾",
    "Peru": "🇵🇪",
    "Philippines": "🇵🇭",
    "Pitcairn": "🇵🇳",
    "Poland": "🇵🇱",
    "Portugal": "🇵🇹",
    "Puerto Rico": "🇵🇷",
    "Qatar": "🇶🇦",
    "Reunion": "🇷🇪",
    "Romania": "🇷🇴",
    "Russian Federation": "🇷🇺",
    "Rwanda": "🇷🇼",
    "Saint Barthelemy": "🇧🇱",
    "Saint Helena, Ascension and Tristan da Cunha": "🇸🇭",
    "Saint Kitts and Nevis": "🇰🇳",
    "Saint Lucia": "🇱🇨",
    "Saint Martin (French part)": "🇲🇫",
    "Saint Pierre and Miquelon": "🇵🇲",
    "Saint Vincent and the Grenadines": "🇻🇨",
    "Samoa": "🇼🇸",
    "San Marino": "🇸🇲",
    "Sao Tome and Principe": "🇸🇹",
    "Saudi Arabia": "🇸🇦",
    "Senegal": "🇸🇳",
    "Serbia": "🇷🇸",
    "Seychelles": "🇸🇨",
    "Sierra Leone": "🇸🇱",
    "Singapore": "🇸🇬",
    "Sint Maarten (Dutch part)": "🇸🇽",
    "Slovakia": "🇸🇰",
    "Slovenia": "🇸🇮",
    "Solomon Islands": "🇸🇧",
    "Somalia": "🇸🇴",
    "South Africa": "🇿🇦",
    "South Georgia and the South Sandwich Islands": "🇬🇸",
    "South Sudan": "🇸🇸",
    "Spain": "🇪🇸",
    "Sri Lanka": "🇱🇰",
    "Sudan": "🇸🇩",
    "Suriname": "🇸🇷",
    "Svalbard and Jan Mayen": "🇸🇯",
    "Sweden": "🇸🇪",
    "Switzerland": "🇨🇭",
    "Syrian Arab Republic": "🇸🇾",
    "Taiwan (Province of China)": "🇹🇼",
    "Tajikistan": "🇹🇯",
    "Tanzania, United Republic of": "🇹🇿",
    "Thailand": "🇹🇭",
    "Timor-Leste": "🇹🇱",
    "Togo": "🇹🇬",
    "Tokelau": "🇹🇰",
    "Tonga": "🇹🇴",
    "Trinidad and Tobago": "🇹🇹",
    "Tunisia": "🇹🇳",
    "Turkey": "🇹🇷",
    "Turkmenistan": "🇹🇲",
    "Turks and Caicos Islands": "🇹🇨",
    "Tuvalu": "🇹🇻",
    "Uganda": "🇺🇬",
    "Ukraine": "🇺🇦",
    "United Arab Emirates": "🇦🇪",
    "United Kingdom of Great Britain and Northern Ireland": "🇬🇧",
    "United States Minor Outlying Islands": "🇺🇲",
    "United States of America": "🇺🇸",
    "Uruguay": "🇺🇾",
    "Uzbekistan": "🇺🇿",
    "Vanuatu": "🇻🇺",
    "Venezuela (Bolivarian Republic of)": "🇻🇪",
    "Viet Nam": "🇻🇳",
    "Virgin Islands (British)": "🇻🇬",
    "Virgin Islands (U.S.)": "🇻🇮",
    "Wallis and Futuna": "🇼🇫",
    "Western Sahara": "🇪🇭",
    "Yemen": "🇾🇪",
    "Zambia": "🇿🇲",
    "Zimbabwe": "🇿🇼",
}


# --- 国名の英語→日本語変換マッピング ---
COUNTRY_NAME_JP = {
    "Afghanistan": "アフガニスタン",
    "Aland Islands": "オーランド諸島",
    "Albania": "アルバニア",
    "Algeria": "アルジェリア",
    "American Samoa": "アメリカ領サモア",
    "Andorra": "アンドラ",
    "Angola": "アンゴラ",
    "Anguilla": "アンギラ",
    "Antarctica": "南極",
    "Antigua and Barbuda": "アンティグア・バーブーダ",
    "Argentina": "アルゼンチン",
    "Armenia": "アルメニア",
    "Aruba": "アルバ",
    "Australia": "オーストラリア",
    "Austria": "オーストリア",
    "Azerbaijan": "アゼルバイジャン",
    "Bahamas": "バハマ",
    "Bahrain": "バーレーン",
    "Bangladesh": "バングラデシュ",
    "Barbados": "バルバドス",
    "Belarus": "ベラルーシ",
    "Belgium": "ベルギー",
    "Belize": "ベリーズ",
    "Benin": "ベナン",
    "Bermuda": "バミューダ",
    "Bhutan": "ブータン",
    "Bolivia (Plurinational State of)": "ボリビア",
    "Bonaire, Sint Eustatius and Saba": "ボネール、シント・ユースタティウス、サバ",
    "Bosnia and Herzegovina": "ボスニア・ヘルツェゴビナ",
    "Botswana": "ボツワナ",
    "Bouvet Island": "ブーベ島",
    "Brazil": "ブラジル",
    "British Indian Ocean Territory": "イギリス領インド洋地域",
    "Brunei Darussalam": "ブルネイ",
    "Bulgaria": "ブルガリア",
    "Burkina Faso": "ブルキナファソ",
    "Burundi": "ブルンジ",
    "Cabo Verde": "カーボベルデ",
    "Cambodia": "カンボジア",
    "Cameroon": "カメルーン",
    "Canada": "カナダ",
    "Cayman Islands": "ケイマン諸島",
    "Central African Republic": "中央アフリカ共和国",
    "Chad": "チャド",
    "Chile": "チリ",
    "China": "中国",
    "Christmas Island": "クリスマス島",
    "Cocos (Keeling) Islands": "ココス諸島",
    "Colombia": "コロンビア",
    "Comoros": "コモロ",
    "Congo": "コンゴ共和国",
    "Congo (Democratic Republic of the)": "コンゴ民主共和国",
    "Cook Islands": "クック諸島",
    "Costa Rica": "コスタリカ",
    "Cote d'Ivoire": "コートジボワール",
    "Croatia": "クロアチア",
    "Cuba": "キューバ",
    "Curacao": "キュラソー",
    "Cyprus": "キプロス",
    "Czechia": "チェコ",
    "Denmark": "デンマーク",
    "Djibouti": "ジブチ",
    "Dominica": "ドミニカ国",
    "Dominican Republic": "ドミニカ共和国",
    "Ecuador": "エクアドル",
    "Egypt": "エジプト",
    "El Salvador": "エルサルバドル",
    "Equatorial Guinea": "赤道ギニア",
    "Eritrea": "エリトリア",
    "Estonia": "エストニア",
    "Eswatini": "エスワティニ",
    "Ethiopia": "エチオピア",
    "Falkland Islands (Malvinas)": "フォークランド諸島",
    "Faroe Islands": "フェロー諸島",
    "Fiji": "フィジー",
    "Finland": "フィンランド",
    "France": "フランス",
    "French Guiana": "フランス領ギアナ",
    "French Polynesia": "フランス領ポリネシア",
    "French Southern Territories": "フランス領南方・南極地域",
    "Gabon": "ガボン",
    "Gambia": "ガンビア",
    "Georgia": "ジョージア",
    "Germany": "ドイツ",
    "Ghana": "ガーナ",
    "Gibraltar": "ジブラルタル",
    "Greece": "ギリシャ",
    "Greenland": "グリーンランド",
    "Grenada": "グレナダ",
    "Guadeloupe": "グアドループ",
    "Guam": "グアム",
    "Guatemala": "グアテマラ",
    "Guernsey": "ガーンジー",
    "Guinea": "ギニア",
    "Guinea-Bissau": "ギニアビサウ",
    "Guyana": "ガイアナ",
    "Haiti": "ハイチ",
    "Heard Island and McDonald Islands": "ハード島とマクドナルド諸島",
    "Holy See": "バチカン市国",
    "Honduras": "ホンジュラス",
    "Hong Kong": "香港",
    "Hungary": "ハンガリー",
    "Iceland": "アイスランド",
    "India": "インド",
    "Indonesia": "インドネシア",
    "Iran (Islamic Republic of)": "イラン",
    "Iraq": "イラク",
    "Ireland": "アイルランド",
    "Isle of Man": "マン島",
    "Israel": "イスラエル",
    "Italy": "イタリア",
    "Jamaica": "ジャマイカ",
    "Japan": "日本",
    "Jersey": "ジャージー",
    "Jordan": "ヨルダン",
    "Kazakhstan": "カザフスタン",
    "Kenya": "ケニア",
    "Kiribati": "キリバス",
    "Korea (Democratic People's Republic of)": "北朝鮮",
    "Korea (Republic of)": "韓国",
    "Kosovo": "コソボ",
    "Kuwait": "クウェート",
    "Kyrgyzstan": "キルギス",
    "Lao People's Democratic Republic": "ラオス",
    "Latvia": "ラトビア",
    "Lebanon": "レバノン",
    "Lesotho": "レソト",
    "Liberia": "リベリア",
    "Libya": "リビア",
    "Liechtenstein": "リヒテンシュタイン",
    "Lithuania": "リトアニア",
    "Luxembourg": "ルクセンブルク",
    "Macao": "マカオ",
    "Madagascar": "マダガスカル",
    "Malawi": "マラウイ",
    "Malaysia": "マレーシア",
    "Maldives": "モルディブ",
    "Mali": "マリ",
    "Malta": "マルタ",
    "Marshall Islands": "マーシャル諸島",
    "Martinique": "マルティニーク",
    "Mauritania": "モーリタニア",
    "Mauritius": "モーリシャス",
    "Mayotte": "マヨット",
    "Mexico": "メキシコ",
    "Micronesia (Federated States of)": "ミクロネシア",
    "Moldova (Republic of)": "モルドバ",
    "Monaco": "モナコ",
    "Mongolia": "モンゴル",
    "Montenegro": "モンテネグロ",
    "Montserrat": "モントセラト",
    "Morocco": "モロッコ",
    "Mozambique": "モザンビーク",
    "Myanmar": "ミャンマー",
    "Namibia": "ナミビア",
    "Nauru": "ナウル",
    "Nepal": "ネパール",
    "Netherlands": "オランダ",
    "New Caledonia": "ニューカレドニア",
    "New Zealand": "ニュージーランド",
    "Nicaragua": "ニカラグア",
    "Niger": "ニジェール",
    "Nigeria": "ナイジェリア",
    "Niue": "ニウエ",
    "Norfolk Island": "ノーフォーク島",
    "North Macedonia": "北マケドニア",
    "Northern Mariana Islands": "北マリアナ諸島",
    "Norway": "ノルウェー",
    "Oman": "オマーン",
    "Pakistan": "パキスタン",
    "Palau": "パラオ",
    "Palestine, State of": "パレスチナ",
    "Panama": "パナマ",
    "Papua New Guinea": "パプアニューギニア",
    "Paraguay": "パラグアイ",
    "Peru": "ペルー",
    "Philippines": "フィリピン",
    "Pitcairn": "ピトケアン諸島",
    "Poland": "ポーランド",
    "Portugal": "ポルトガル",
    "Puerto Rico": "プエルトリコ",
    "Qatar": "カタール",
    "Reunion": "レユニオン",
    "Romania": "ルーマニア",
    "Russian Federation": "ロシア",
    "Rwanda": "ルワンダ",
    "Saint Barthelemy": "サン・バルテルミー",
    "Saint Helena, Ascension and Tristan da Cunha": "セントヘレナ・アセンション・トリスタンダクーニャ",
    "Saint Kitts and Nevis": "セントクリストファー・ネイビス",
    "Saint Lucia": "セントルシア",
    "Saint Martin (French part)": "サン・マルタン",
    "Saint Pierre and Miquelon": "サンピエール島・ミクロン島",
    "Saint Vincent and the Grenadines": "セントビンセント・グレナディーン",
    "Samoa": "サモア",
    "San Marino": "サンマリノ",
    "Sao Tome and Principe": "サントメ・プリンシペ",
    "Saudi Arabia": "サウジアラビア",
    "Senegal": "セネガル",
    "Serbia": "セルビア",
    "Seychelles": "セーシェル",
    "Sierra Leone": "シエラレオネ",
    "Singapore": "シンガポール",
    "Sint Maarten (Dutch part)": "シント・マールテン",
    "Slovakia": "スロバキア",
    "Slovenia": "スロベニア",
    "Solomon Islands": "ソロモン諸島",
    "Somalia": "ソマリア",
    "South Africa": "南アフリカ",
    "South Georgia and the South Sandwich Islands": "サウスジョージア・サウスサンドウィッチ諸島",
    "South Sudan": "南スーダン",
    "Spain": "スペイン",
    "Sri Lanka": "スリランカ",
    "Sudan": "スーダン",
    "Suriname": "スリナム",
    "Svalbard and Jan Mayen": "スヴァールバル諸島・ヤンマイエン島",
    "Sweden": "スウェーデン",
    "Switzerland": "スイス",
    "Syrian Arab Republic": "シリア",
    "Taiwan (Province of China)": "台湾",
    "Tajikistan": "タジキスタン",
    "Tanzania, United Republic of": "タンザニア",
    "Thailand": "タイ",
    "Timor-Leste": "東ティモール",
    "Togo": "トーゴ",
    "Tokelau": "トケラウ",
    "Tonga": "トンガ",
    "Trinidad and Tobago": "トリニダード・トバゴ",
    "Tunisia": "チュニジア",
    "Turkey": "トルコ",
    "Turkmenistan": "トルクメニスタン",
    "Turks and Caicos Islands": "タークス・カイコス諸島",
    "Tuvalu": "ツバル",
    "Uganda": "ウガンダ",
    "Ukraine": "ウクライナ",
    "United Arab Emirates": "アラブ首長国連邦",
    "United Kingdom of Great Britain and Northern Ireland": "イギリス",
    "United States Minor Outlying Islands": "合衆国領有小離島",
    "United States of America": "アメリカ",
    "Uruguay": "ウルグアイ",
    "Uzbekistan": "ウズベキスタン",
    "Vanuatu": "バヌアツ",
    "Venezuela (Bolivarian Republic of)": "ベネズエラ",
    "Viet Nam": "ベトナム",
    "Virgin Islands (British)": "イギリス領ヴァージン諸島",
    "Virgin Islands (U.S.)": "アメリカ領ヴァージン諸島",
    "Wallis and Futuna": "ウォリス・フツナ",
    "Western Sahara": "西サハラ",
    "Yemen": "イエメン",
    "Zambia": "ザンビア",
    "Zimbabwe": "ジンバブエ",
}


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
app = FastAPI()

# Cloudflare R2の公開URL
R2_PUBLIC_URL = "https://pub-20801d1056e542a99ab766366e3a3124.r2.dev"


# --- ヘルスチェック用エンドポイント ---
@app.api_route("/", methods=["GET", "HEAD"])
def read_root():
    return {"status": "ok"}


# --- 静的ファイル配信の設定 ---
# BASE_DIRを使って、実行場所によらない絶対パスを指定
# app.mount("/images", StaticFiles(directory=BASE_DIR / "cropped_images"), name="images") # R2を使うため不要

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
    global model, prototypes, representatives, country_metadata_g
    # 1. 顔分析モデルを準備
    print("顔分析モデルを準備しています...")
    model = insightface.app.FaceAnalysis(providers=["CPUExecutionProvider"])
    # 性別・年齢推定モデルも有効化
    model.prepare(ctx_id=0, det_thresh=0.1, det_size=(640, 640))
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

    print(
        f"男性代表: {len(prototypes['man'])}件, 女性代表: {len(prototypes['woman'])}件 読み込みました。"
    )

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

    print(
        f"男性代表画像: {len(representatives['man'])}件, 女性代表画像: {len(representatives['woman'])}件 読み込みました。"
    )

    # 4. 事前に生成した国別メタデータをファイルから読み込む
    metadata_path = BASE_DIR / "backend" / "data" / "country_metadata.json"
    if not metadata_path.exists():
        print(f"警告: {metadata_path} が見つかりません。ボーナス計算は無効になります。")
        return

    print(f"{metadata_path} から国別メタデータを読み込んでいます...")
    with open(metadata_path, "r", encoding="utf-8") as f:
        country_metadata_g = json.load(f)
    print(f"{len(country_metadata_g)}カ国分のメタデータを準備しました。")


# --- リファクタリングされたヘルパー関数 ---


async def _get_face_details(file: UploadFile) -> Any:
    """アップロードされた画像から顔の情報を抽出する"""
    if model is None:
        # このチェックはエンドポイント側で既に行われているが、念のため追加
        raise HTTPException(status_code=503, detail="モデルが初期化されていません。")

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

    return faces[0]


def _calculate_ranking(
    user_embedding: np.ndarray, gender_str: str
) -> List[Tuple[str, float]]:
    """顔の特徴量ベクトルと性別から国別ランキングを計算する"""
    target_prototypes = prototypes[gender_str]
    target_representatives = representatives[gender_str]

    available_countries = set(target_prototypes.keys()) & set(
        target_representatives.keys()
    )
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
                geo_bonus = GEO_BONUS

            rarity = metadata.get("rarity", 1)
            rarity_bonus = (rarity - 1) * RARITY_BONUS_UNIT

        final_score = base_score + geo_bonus + rarity_bonus
        similarities[country] = final_score

    adjusted_scores = {}
    if similarities:
        max_original_score = max(similarities.values())
        target_top_score = random.randint(85, 99)
        if max_original_score > 0:
            for country, score in similarities.items():
                adjusted_scores[country] = (
                    score / max_original_score
                ) * target_top_score
        else:
            adjusted_scores = {country: 0 for country in similarities.keys()}

    return sorted(adjusted_scores.items(), key=lambda item: item[1], reverse=True)


# --- APIエンドポイント ---
@app.post("/analyze")
async def analyze_face(
    file: UploadFile = File(...),
    gender: str = Form(...),  # フロントエンドからの性別指定（'male' or 'female'）
):
    """フロントエンド用の顔分析エンドポイント"""
    if model is None or not prototypes["man"] or not prototypes["woman"]:
        raise HTTPException(status_code=503, detail="モデルがまだ準備できていません。")

    user_face = await _get_face_details(file)
    user_embedding = user_face.embedding
    # フロントからの 'male'/'female' を 'man'/'woman' に変換
    user_gender_str = "man" if gender == "male" else "woman"

    sorted_countries = _calculate_ranking(user_embedding, user_gender_str)

    ranking_result = []
    for country, score in sorted_countries[:10]:
        ranking_result.append(
            {
                "country": get_country_name_japanese(country),
                "country_english": country,  # 元の英語名も保持（デバッグ用）
                "similarity": float(score),
                "country_code": get_country_code(country),
            }
        )

    # 1位の国の代表画像ファイル名を取得
    top_country_image_url = None
    if sorted_countries:
        top_country = sorted_countries[0][0]
        target_reps = representatives[user_gender_str]
        if top_country in target_reps:
            image_filename = target_reps[top_country]
            top_country_image_url = f"{R2_PUBLIC_URL}/{image_filename}"

    return JSONResponse(
        content={
            "ranking": ranking_result,
            "top_country_image_url": top_country_image_url,
        }
    )


@app.get("/comparison")
async def get_comparison_image(
    country: str,
    gender: str,
):
    """このエンドポイントは使われなくなるため削除または無効化"""
    raise HTTPException(
        status_code=410,
        detail="このエンドポイントは廃止されました。代わりに/analyzeレスポンスのURLを使用してください。",
    )


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

    user_face = await _get_face_details(file)
    user_embedding = user_face.embedding

    # --- 性別の決定 ---
    if gender_override in ["man", "woman"]:
        user_gender_str = gender_override
        print(f"ユーザー指定の性別を使用: {user_gender_str}")
    else:
        user_gender_str = "man" if user_face.gender == 0 else "woman"
        print(f"AIが検出した性別: {user_gender_str} (年齢: {user_face.age})")

    sorted_countries = _calculate_ranking(user_embedding, user_gender_str)

    target_representatives = representatives[user_gender_str]

    # JSONで返せる形式に整形
    ranking_result = []
    for i, (country, score) in enumerate(sorted_countries[:5]):
        rank_data = {
            "rank": i + 1,
            "country": get_country_name_japanese(country),
            "country_english": country,  # 元の英語名も保持（デバッグ用）
            "score": float(score),
            "country_code": get_country_code(country),
        }
        if i == 0 and country in target_representatives:
            rank_data["representative_image_filename"] = target_representatives[country]
        ranking_result.append(rank_data)

    response_content = {
        "detected_gender": user_gender_str,
        "ranking": ranking_result,
    }

    return JSONResponse(content=response_content)


# --- フロントエンド配信（最後にマウント） ---
# APIルートなどをすべて定義した後に、残りのパスをフロントエンドに回す
# BASE_DIRを使って、実行場所によらない絶対パスを指定
# app.mount("/", StaticFiles(directory=BASE_DIR / "frontend", html=True), name="frontend")


# --- メインの実行部分（デバッグ用） ---
if __name__ == "__main__":
    import uvicorn

    # サーバーを起動
    # uvicorn.run("main:app", host="0.0.0.0", port=8003)
    # 開発中はリロード機能を有効にすると便利
    uvicorn.run(
        "main:app", host="0.0.0.0", port=8003, reload=True, reload_dirs=[str(BASE_DIR)]
    )
