"""
拡張された国データ自動生成スクリプト
REST Countries API、CountryInfo、GeoPy等を活用して包括的な国データを取得
"""

import json
import requests
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import asdict
import sys
import os

# 親ディレクトリをパスに追加
sys.path.append(str(Path(__file__).parent.parent))

from schemas.country_schema import (
    CountryData, CountryBasicInfo, CountryCoordinates, CountryMetadata,
    CountryContent, CountryImages, CountryHighlight, CountryRawData,
    GenerationResult, ValidationResult, validate_country_data,
    country_data_to_dict
)


class EnhancedCountryDataGenerator:
    """拡張された国データ生成クラス"""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent.parent
        self.data_dir = self.base_dir / "data"
        self.existing_metadata_path = self.data_dir / "country_metadata.json"
        self.output_path = self.data_dir / "enhanced_country_data.json"
        
        # REST Countries APIのベースURL
        self.rest_countries_base = "https://restcountries.com/v3.1"
        
        # 既存のメタデータを読み込み
        self.existing_metadata = self._load_existing_metadata()
        
        # 国名の正規化マッピング
        self.country_name_mapping = {
            "Korea (Republic of)": "South Korea",
            "Korea (Democratic People's Republic of)": "North Korea",
            "Iran (Islamic Republic of)": "Iran",
            "Russian Federation": "Russia",
            "United States of America": "United States",
            "United Kingdom of Great Britain and Northern Ireland": "United Kingdom",
            "Viet Nam": "Vietnam",
            "Czechia": "Czech Republic",
            "Cabo Verde": "Cape Verde",
            "Brunei Darussalam": "Brunei",
            "Bolivia (Plurinational State of)": "Bolivia",
            "Venezuela (Bolivarian Republic of)": "Venezuela",
            "Syria Arab Republic": "Syria",
            "Tanzania, United Republic of": "Tanzania",
            "Moldova (Republic of)": "Moldova",
            "Lao People's Democratic Republic": "Laos",
            "Micronesia (Federated States of)": "Micronesia",
            "Holy See (Vatican City State)": "Vatican City",
            "Timor-Leste": "East Timor"
        }
        
        # 国旗絵文字マッピング（主要国）
        self.flag_emoji_mapping = {
            "AD": "🇦🇩", "AE": "🇦🇪", "AF": "🇦🇫", "AG": "🇦🇬", "AI": "🇦🇮", "AL": "🇦🇱", "AM": "🇦🇲", "AO": "🇦🇴",
            "AQ": "🇦🇶", "AR": "🇦🇷", "AS": "🇦🇸", "AT": "🇦🇹", "AU": "🇦🇺", "AW": "🇦🇼", "AX": "🇦🇽", "AZ": "🇦🇿",
            "BA": "🇧🇦", "BB": "🇧🇧", "BD": "🇧🇩", "BE": "🇧🇪", "BF": "🇧🇫", "BG": "🇧🇬", "BH": "🇧🇭", "BI": "🇧🇮",
            "BJ": "🇧🇯", "BL": "🇧🇱", "BM": "🇧🇲", "BN": "🇧🇳", "BO": "🇧🇴", "BQ": "🇧🇶", "BR": "🇧🇷", "BS": "🇧🇸",
            "BT": "🇧🇹", "BV": "🇧🇻", "BW": "🇧🇼", "BY": "🇧🇾", "BZ": "🇧🇿", "CA": "🇨🇦", "CC": "🇨🇨", "CD": "🇨🇩",
            "CF": "🇨🇫", "CG": "🇨🇬", "CH": "🇨🇭", "CI": "🇨🇮", "CK": "🇨🇰", "CL": "🇨🇱", "CM": "🇨🇲", "CN": "🇨🇳",
            "CO": "🇨🇴", "CR": "🇨🇷", "CU": "🇨🇺", "CV": "🇨🇻", "CW": "🇨🇼", "CX": "🇨🇽", "CY": "🇨🇾", "CZ": "🇨🇿",
            "DE": "🇩🇪", "DJ": "🇩🇯", "DK": "🇩🇰", "DM": "🇩🇲", "DO": "🇩🇴", "DZ": "🇩🇿", "EC": "🇪🇨", "EE": "🇪🇪",
            "EG": "🇪🇬", "EH": "🇪🇭", "ER": "🇪🇷", "ES": "🇪🇸", "ET": "🇪🇹", "FI": "🇫🇮", "FJ": "🇫🇯", "FK": "🇫🇰",
            "FM": "🇫🇲", "FO": "🇫🇴", "FR": "🇫🇷", "GA": "🇬🇦", "GB": "🇬🇧", "GD": "🇬🇩", "GE": "🇬🇪", "GF": "🇬🇫",
            "GG": "🇬🇬", "GH": "🇬🇭", "GI": "🇬🇮", "GL": "🇬🇱", "GM": "🇬🇲", "GN": "🇬🇳", "GP": "🇬🇵", "GQ": "🇬🇶",
            "GR": "🇬🇷", "GS": "🇬🇸", "GT": "🇬🇹", "GU": "🇬🇺", "GW": "🇬🇼", "GY": "🇬🇾", "HK": "🇭🇰", "HM": "🇭🇲",
            "HN": "🇭🇳", "HR": "🇭🇷", "HT": "🇭🇹", "HU": "🇭🇺", "ID": "🇮🇩", "IE": "🇮🇪", "IL": "🇮🇱", "IM": "🇮🇲",
            "IN": "🇮🇳", "IO": "🇮🇴", "IQ": "🇮🇶", "IR": "🇮🇷", "IS": "🇮🇸", "IT": "🇮🇹", "JE": "🇯🇪", "JM": "🇯🇲",
            "JO": "🇯🇴", "JP": "🇯🇵", "KE": "🇰🇪", "KG": "🇰🇬", "KH": "🇰🇭", "KI": "🇰🇮", "KM": "🇰🇲", "KN": "🇰🇳",
            "KP": "🇰🇵", "KR": "🇰🇷", "KW": "🇰🇼", "KY": "🇰🇾", "KZ": "🇰🇿", "LA": "🇱🇦", "LB": "🇱🇧", "LC": "🇱🇨",
            "LI": "🇱🇮", "LK": "🇱🇰", "LR": "🇱🇷", "LS": "🇱🇸", "LT": "🇱🇹", "LU": "🇱🇺", "LV": "🇱🇻", "LY": "🇱🇾",
            "MA": "🇲🇦", "MC": "🇲🇨", "MD": "🇲🇩", "ME": "🇲🇪", "MF": "🇲🇫", "MG": "🇲🇬", "MH": "🇲🇭", "MK": "🇲🇰",
            "ML": "🇲🇱", "MM": "🇲🇲", "MN": "🇲🇳", "MO": "🇲🇴", "MP": "🇲🇵", "MQ": "🇲🇶", "MR": "🇲🇷", "MS": "🇲🇸",
            "MT": "🇲🇹", "MU": "🇲🇺", "MV": "🇲🇻", "MW": "🇲🇼", "MX": "🇲🇽", "MY": "🇲🇾", "MZ": "🇲🇿", "NA": "🇳🇦",
            "NC": "🇳🇨", "NE": "🇳🇪", "NF": "🇳🇫", "NG": "🇳🇬", "NI": "🇳🇮", "NL": "🇳🇱", "NO": "🇳🇴", "NP": "🇳🇵",
            "NR": "🇳🇷", "NU": "🇳🇺", "NZ": "🇳🇿", "OM": "🇴🇲", "PA": "🇵🇦", "PE": "🇵🇪", "PF": "🇵🇫", "PG": "🇵🇬",
            "PH": "🇵🇭", "PK": "🇵🇰", "PL": "🇵🇱", "PM": "🇵🇲", "PN": "🇵🇳", "PR": "🇵🇷", "PS": "🇵🇸", "PT": "🇵🇹",
            "PW": "🇵🇼", "PY": "🇵🇾", "QA": "🇶🇦", "RE": "🇷🇪", "RO": "🇷🇴", "RS": "🇷🇸", "RU": "🇷🇺", "RW": "🇷🇼",
            "SA": "🇸🇦", "SB": "🇸🇧", "SC": "🇸🇨", "SD": "🇸🇩", "SE": "🇸🇪", "SG": "🇸🇬", "SH": "🇸🇭", "SI": "🇸🇮",
            "SJ": "🇸🇯", "SK": "🇸🇰", "SL": "🇸🇱", "SM": "🇸🇲", "SN": "🇸🇳", "SO": "🇸🇴", "SR": "🇸🇷", "SS": "🇸🇸",
            "ST": "🇸🇹", "SV": "🇸🇻", "SX": "🇸🇽", "SY": "🇸🇾", "SZ": "🇸🇿", "TC": "🇹🇨", "TD": "🇹🇩", "TF": "🇹🇫",
            "TG": "🇹🇬", "TH": "🇹🇭", "TJ": "🇹🇯", "TK": "🇹🇰", "TL": "🇹🇱", "TM": "🇹🇲", "TN": "🇹🇳", "TO": "🇹🇴",
            "TR": "🇹🇷", "TT": "🇹🇹", "TV": "🇹🇻", "TW": "🇹🇼", "TZ": "🇹🇿", "UA": "🇺🇦", "UG": "🇺🇬", "UM": "🇺🇲",
            "US": "🇺🇸", "UY": "🇺🇾", "UZ": "🇺🇿", "VA": "🇻🇦", "VC": "🇻🇨", "VE": "🇻🇪", "VG": "🇻🇬", "VI": "🇻🇮",
            "VN": "🇻🇳", "VU": "🇻🇺", "WF": "🇼🇫", "WS": "🇼🇸", "YE": "🇾🇪", "YT": "🇾🇹", "ZA": "🇿🇦", "ZM": "🇿🇲", "ZW": "🇿🇼"
        }
    
    def _load_existing_metadata(self) -> Dict:
        """既存のメタデータを読み込み"""
        if self.existing_metadata_path.exists():
            with open(self.existing_metadata_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def _normalize_country_name(self, name: str) -> str:
        """国名を正規化"""
        return self.country_name_mapping.get(name, name)
    
    def _get_country_code_from_name(self, name: str) -> Optional[str]:
        """国名から国コードを取得"""
        try:
            url = f"{self.rest_countries_base}/name/{name}"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data:
                    return data[0].get('cca2', '').lower()
        except Exception as e:
            print(f"Error getting country code for {name}: {e}")
        return None
    
    def _get_country_info_from_rest_api(self, country_code: str) -> Optional[Dict]:
        """REST Countries APIから国情報を取得"""
        try:
            url = f"{self.rest_countries_base}/alpha/{country_code.upper()}"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data:
                    return data[0]
        except Exception as e:
            print(f"Error fetching data for {country_code}: {e}")
        return None
    
    def _format_population(self, population: int) -> str:
        """人口を日本語形式でフォーマット"""
        if population >= 100_000_000:
            return f"{population / 100_000_000:.1f}億人"
        elif population >= 10_000:
            return f"{population / 10_000:.0f}万人"
        elif population >= 1_000:
            return f"{population / 1_000:.1f}千人"
        else:
            return f"{population}人"
    
    def _get_japanese_country_name(self, english_name: str, country_code: str) -> str:
        """英語名から日本語名を取得（簡易版）"""
        # 主要国の日本語名マッピング
        name_mapping = {
            "Japan": "日本",
            "United States": "アメリカ",
            "United Kingdom": "イギリス",
            "Germany": "ドイツ",
            "France": "フランス",
            "Italy": "イタリア",
            "Spain": "スペイン",
            "China": "中国",
            "South Korea": "韓国",
            "North Korea": "北朝鮮",
            "Russia": "ロシア",
            "India": "インド",
            "Australia": "オーストラリア",
            "Canada": "カナダ",
            "Brazil": "ブラジル",
            "Mexico": "メキシコ",
            "Argentina": "アルゼンチン",
            "Egypt": "エジプト",
            "South Africa": "南アフリカ",
            "Nigeria": "ナイジェリア",
            "Kenya": "ケニア",
            "Morocco": "モロッコ",
            "Algeria": "アルジェリア",
            "Thailand": "タイ",
            "Vietnam": "ベトナム",
            "Indonesia": "インドネシア",
            "Malaysia": "マレーシア",
            "Singapore": "シンガポール",
            "Philippines": "フィリピン",
            "Turkey": "トルコ",
            "Greece": "ギリシャ",
            "Netherlands": "オランダ",
            "Belgium": "ベルギー",
            "Switzerland": "スイス",
            "Austria": "オーストリア",
            "Sweden": "スウェーデン",
            "Norway": "ノルウェー",
            "Denmark": "デンマーク",
            "Finland": "フィンランド",
            "Iceland": "アイスランド",
            "Ireland": "アイルランド",
            "Portugal": "ポルトガル",
            "Poland": "ポーランド",
            "Czech Republic": "チェコ",
            "Hungary": "ハンガリー",
            "Romania": "ルーマニア",
            "Bulgaria": "ブルガリア",
            "Croatia": "クロアチア",
            "Serbia": "セルビア",
            "Ukraine": "ウクライナ",
            "Belarus": "ベラルーシ",
            "Lithuania": "リトアニア",
            "Latvia": "ラトビア",
            "Estonia": "エストニア",
            "Israel": "イスラエル",
            "Iran": "イラン",
            "Iraq": "イラク",
            "Saudi Arabia": "サウジアラビア",
            "United Arab Emirates": "アラブ首長国連邦",
            "Kuwait": "クウェート",
            "Qatar": "カタール",
            "Bahrain": "バーレーン",
            "Oman": "オマーン",
            "Jordan": "ヨルダン",
            "Lebanon": "レバノン",
            "Syria": "シリア",
            "Afghanistan": "アフガニスタン",
            "Pakistan": "パキスタン",
            "Bangladesh": "バングラデシュ",
            "Sri Lanka": "スリランカ",
            "Nepal": "ネパール",
            "Bhutan": "ブータン",
            "Maldives": "モルディブ",
            "Myanmar": "ミャンマー",
            "Cambodia": "カンボジア",
            "Laos": "ラオス",
            "Mongolia": "モンゴル",
            "Kazakhstan": "カザフスタン",
            "Uzbekistan": "ウズベキスタン",
            "Kyrgyzstan": "キルギス",
            "Tajikistan": "タジキスタン",
            "Turkmenistan": "トルクメニスタン",
            "Georgia": "ジョージア",
            "Armenia": "アルメニア",
            "Azerbaijan": "アゼルバイジャン",
            "Chile": "チリ",
            "Peru": "ペルー",
            "Colombia": "コロンビア",
            "Venezuela": "ベネズエラ",
            "Ecuador": "エクアドル",
            "Bolivia": "ボリビア",
            "Paraguay": "パラグアイ",
            "Uruguay": "ウルグアイ",
            "New Zealand": "ニュージーランド",
            "Fiji": "フィジー",
            "Papua New Guinea": "パプアニューギニア",
            "Solomon Islands": "ソロモン諸島",
            "Vanuatu": "バヌアツ",
            "Samoa": "サモア",
            "Tonga": "トンガ",
            "Palau": "パラオ",
            "Micronesia": "ミクロネシア",
            "Marshall Islands": "マーシャル諸島",
            "Kiribati": "キリバス",
            "Nauru": "ナウル",
            "Tuvalu": "ツバル"
        }
        
        return name_mapping.get(english_name, english_name)
    
    def _get_rarity_by_population(self, population: int) -> int:
        """人口から意外度を計算"""
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
    
    def generate_raw_data_from_existing_metadata(self) -> List[CountryRawData]:
        """既存のメタデータから生データを生成"""
        raw_data_list = []
        
        for country_name, metadata in self.existing_metadata.items():
            print(f"Processing {country_name}...")
            
            # 国名正規化
            normalized_name = self._normalize_country_name(country_name)
            
            # 国コードを取得
            country_code = self._get_country_code_from_name(normalized_name)
            if not country_code:
                print(f"  Warning: Could not get country code for {country_name}")
                continue
            
            # REST APIから詳細情報を取得
            country_info = self._get_country_info_from_rest_api(country_code)
            if not country_info:
                print(f"  Warning: Could not get country info for {country_name}")
                continue
            
            # 基本情報を構築
            try:
                capital = country_info.get('capital', [''])[0] if country_info.get('capital') else ''
                population_num = country_info.get('population', 0)
                population_str = self._format_population(population_num)
                
                # 言語情報を取得
                languages = country_info.get('languages', {})
                language = list(languages.values())[0] if languages else ''
                
                # 座標情報を取得
                latlng = country_info.get('latlng', [0, 0])
                lat, lng = latlng[0], latlng[1] if len(latlng) >= 2 else (0, 0)
                
                # 通貨情報を取得
                currencies = country_info.get('currencies', {})
                currency = list(currencies.keys())[0] if currencies else ''
                
                # 地域情報を取得
                region = country_info.get('region', '')
                subregion = country_info.get('subregion', '')
                
                # 面積情報を取得
                area = country_info.get('area', 0)
                
                # タイムゾーン情報を取得
                timezones = country_info.get('timezones', [])
                timezone = timezones[0] if timezones else ''
                
                # 国際電話番号
                idd = country_info.get('idd', {})
                calling_code = idd.get('root', '') + ''.join(idd.get('suffixes', []))
                
                # 国旗絵文字
                flag_emoji = self.flag_emoji_mapping.get(country_code.upper(), '🏳️')
                
                # 日本語名
                japanese_name = self._get_japanese_country_name(normalized_name, country_code)
                
                # 意外度計算
                rarity = self._get_rarity_by_population(population_num)
                
                raw_data = CountryRawData(
                    name=japanese_name,
                    name_en=normalized_name,
                    code=country_code,
                    flag=flag_emoji,
                    basic=CountryBasicInfo(
                        capital=capital,
                        population=population_str,
                        language=language
                    ),
                    coordinates=CountryCoordinates(
                        lat=lat,
                        lng=lng
                    ),
                    metadata=CountryMetadata(
                        continent=metadata.get('continent'),
                        rarity=rarity,
                        region=region,
                        subregion=subregion,
                        population_number=population_num,
                        area=area,
                        timezone=timezone,
                        currency=currency,
                        calling_code=calling_code
                    )
                )
                
                raw_data_list.append(raw_data)
                print(f"  ✓ Successfully processed {country_name}")
                
            except Exception as e:
                print(f"  ✗ Error processing {country_name}: {e}")
                continue
            
            # APIレート制限対応
            time.sleep(0.1)
        
        return raw_data_list
    
    def save_raw_data(self, raw_data_list: List[CountryRawData], filename: str = "country_raw_data.json"):
        """生データをJSONファイルに保存"""
        output_file = self.data_dir / filename
        
        # データクラスを辞書に変換
        data_dict = {}
        for raw_data in raw_data_list:
            data_dict[raw_data.code] = {
                "name": raw_data.name,
                "nameEn": raw_data.name_en,
                "code": raw_data.code,
                "flag": raw_data.flag,
                "basic": asdict(raw_data.basic) if raw_data.basic else None,
                "coordinates": asdict(raw_data.coordinates) if raw_data.coordinates else None,
                "metadata": asdict(raw_data.metadata) if raw_data.metadata else None
            }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data_dict, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Raw data saved to {output_file}")
        print(f"   Total countries: {len(raw_data_list)}")
    
    def run(self):
        """メイン実行関数"""
        print("🚀 Enhanced Country Data Generator")
        print("=" * 50)
        
        # 既存のメタデータから生データを生成
        print("📊 Generating raw data from existing metadata...")
        raw_data_list = self.generate_raw_data_from_existing_metadata()
        
        # 生データを保存
        print("\n💾 Saving raw data...")
        self.save_raw_data(raw_data_list)
        
        print("\n✅ Raw data generation completed!")
        print(f"   Successfully processed: {len(raw_data_list)} countries")
        
        return raw_data_list


if __name__ == "__main__":
    generator = EnhancedCountryDataGenerator()
    generator.run()