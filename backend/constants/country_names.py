"""
国名日本語変換マッピング定数
main.pyから分離された英語→日本語国名変換辞書
"""

from typing import Dict

COUNTRY_NAME_JP: Dict[str, str] = {
    "Afghanistan": "アフガニスタン",
    "Albania": "アルバニア", 
    "Algeria": "アルジェリア",
    "Argentina": "アルゼンチン",
    "Australia": "オーストラリア",
    "Austria": "オーストリア",
    "Bangladesh": "バングラデシュ",
    "Belgium": "ベルギー", 
    "Brazil": "ブラジル",
    "Canada": "カナダ",
    "China": "中国",
    "France": "フランス",
    "Germany": "ドイツ",
    "India": "インド",
    "Italy": "イタリア", 
    "Japan": "日本",
    "South Korea": "韓国",
    "Mexico": "メキシコ",
    "Netherlands": "オランダ",
    "Russia": "ロシア",
    "Spain": "スペイン",
    "United Kingdom": "イギリス",
    "United States": "アメリカ合衆国"
    # 他の国名は必要に応じて追加
}