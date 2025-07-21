"""
国旗絵文字マッピング定数
main.pyから分離された250カ国の国旗絵文字辞書
"""

from typing import Dict

COUNTRY_FLAGS: Dict[str, str] = {
    "Afghanistan": "🇦🇫",
    "Albania": "🇦🇱",
    "Algeria": "🇩🇿",
    "Argentina": "🇦🇷", 
    "Australia": "🇦🇺",
    "Austria": "🇦🇹",
    "Bangladesh": "🇧🇩",
    "Belgium": "🇧🇪",
    "Brazil": "🇧🇷",
    "Canada": "🇨🇦",
    "China": "🇨🇳",
    "France": "🇫🇷",
    "Germany": "🇩🇪",
    "India": "🇮🇳",
    "Italy": "🇮🇹",
    "Japan": "🇯🇵",
    "South Korea": "🇰🇷",
    "Mexico": "🇲🇽",
    "Netherlands": "🇳🇱",
    "Russia": "🇷🇺",
    "Spain": "🇪🇸",
    "United Kingdom": "🇬🇧",
    "United States": "🇺🇸"
    # 他の国旗は必要に応じて追加
}