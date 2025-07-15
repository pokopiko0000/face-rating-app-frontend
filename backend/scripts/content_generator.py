"""
コンテンツ生成テンプレートシステム
各国の魅力的なdescription、highlights、whyVisitを自動生成
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys
import os

# 親ディレクトリをパスに追加
sys.path.append(str(Path(__file__).parent.parent))

from schemas.country_schema import CountryContent, CountryHighlight, CountryRawData


class ContentGenerator:
    """コンテンツ自動生成クラス"""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent.parent
        self.data_dir = self.base_dir / "data"
        
        # 既存のコンテンツを参考に、テンプレートを定義
        self.description_templates = self._define_description_templates()
        self.highlight_templates = self._define_highlight_templates()
        self.why_visit_templates = self._define_why_visit_templates()
        
        # 大陸別特徴
        self.continent_features = self._define_continent_features()
        
        # 人口規模別特徴
        self.population_features = self._define_population_features()
        
        # 地理的特徴
        self.geographic_features = self._define_geographic_features()
    
    def _define_description_templates(self) -> Dict[str, List[str]]:
        """description用テンプレート定義"""
        return {
            "large_country": [
                "{country_name}は{continent}に位置する大国で、{population}の人々が暮らしています。{unique_feature}で知られ、{landscape}が魅力的です。",
                "広大な国土を持つ{country_name}。{population}の多様な文化が息づき、{unique_feature}は世界中から注目されています。",
                "{population}を擁する{country_name}は、{unique_feature}で有名です。{landscape}と豊かな文化が訪れる人々を魅了します。"
            ],
            "medium_country": [
                "{continent}に位置する{country_name}は、{population}の人々が暮らす魅力的な国です。{unique_feature}で知られています。",
                "{country_name}は{unique_feature}で有名な{continent}の国。{population}の温かい人々と{landscape}が印象的です。",
                "{population}の{country_name}。{unique_feature}と{landscape}が調和した美しい国です。"
            ],
            "small_country": [
                "{continent}の小さな宝石のような国、{country_name}。{population}ながら{unique_feature}で世界に知られています。",
                "人口{population}の{country_name}は、{unique_feature}で有名な{continent}の隠れた名所です。",
                "小さいながらも{unique_feature}で魅力あふれる{country_name}。{population}の国民が紡ぐ独特の文化があります。"
            ],
            "island_nation": [
                "{continent}の美しい島国{country_name}。{population}の人々が{unique_feature}と共に暮らしています。",
                "青い海に囲まれた{country_name}は、{unique_feature}で知られる楽園です。{population}の島民文化が魅力的。",
                "{population}の島国{country_name}。{unique_feature}と美しい海が織りなす絶景が待っています。"
            ]
        }
    
    def _define_highlight_templates(self) -> Dict[str, List[Dict]]:
        """highlight用テンプレート定義"""
        return {
            "natural_wonders": [
                {
                    "title": "息をのむ自然の絶景",
                    "template": "{natural_feature}は{country_name}の代表的な自然の驚異。{natural_description}で訪れる人々を魅了します。"
                },
                {
                    "title": "壮大な自然のパノラマ",
                    "template": "{natural_feature}の雄大な景色。{natural_description}は一生忘れられない体験となるでしょう。"
                },
                {
                    "title": "神秘的な自然現象",
                    "template": "{natural_feature}で見られる{natural_description}。自然の神秘を間近で体感できます。"
                }
            ],
            "cultural_heritage": [
                {
                    "title": "豊かな文化遺産",
                    "template": "{cultural_feature}は{country_name}の文化的象徴。{cultural_description}として世界に知られています。"
                },
                {
                    "title": "伝統と歴史の宝庫",
                    "template": "{cultural_feature}に見る{cultural_description}。長い歴史が育んだ文化の深さを感じられます。"
                },
                {
                    "title": "独特な文化体験",
                    "template": "{cultural_feature}で体験する{cultural_description}。この国ならではの文化に触れることができます。"
                }
            ],
            "modern_attractions": [
                {
                    "title": "現代的な魅力",
                    "template": "{modern_feature}は{country_name}の現代的な顔。{modern_description}で注目を集めています。"
                },
                {
                    "title": "革新的な都市体験",
                    "template": "{modern_feature}の{modern_description}。伝統と革新が融合した都市の魅力を体感できます。"
                },
                {
                    "title": "最先端の技術と文化",
                    "template": "{modern_feature}で見る{modern_description}。現代{country_name}の躍動感を感じられます。"
                }
            ],
            "local_lifestyle": [
                {
                    "title": "地域の生活文化",
                    "template": "{lifestyle_feature}は{country_name}の日常の魅力。{lifestyle_description}を体験できます。"
                },
                {
                    "title": "人々の暮らしと文化",
                    "template": "{lifestyle_feature}に見る{lifestyle_description}。地元の人々の温かさに触れられます。"
                },
                {
                    "title": "独特のライフスタイル",
                    "template": "{lifestyle_feature}で体験する{lifestyle_description}。この国の人々の生き方を知ることができます。"
                }
            ]
        }
    
    def _define_why_visit_templates(self) -> Dict[str, List[str]]:
        """whyVisit用テンプレート定義"""
        return {
            "adventure": [
                "冒険心をくすぐる{adventure_type}の国",
                "{adventure_type}を体験できる唯一無二の場所",
                "スリリングな{adventure_type}が待つ冒険の地"
            ],
            "culture": [
                "深い{culture_type}に触れられる文化の宝庫",
                "{culture_type}が息づく伝統と現代の融合地",
                "独特な{culture_type}を体験できる特別な国"
            ],
            "nature": [
                "美しい{nature_type}に癒される自然の楽園",
                "{nature_type}の絶景に出会える奇跡の場所",
                "手つかずの{nature_type}が残る地球の宝石"
            ],
            "relaxation": [
                "心と体を癒す{relaxation_type}の国",
                "{relaxation_type}でリフレッシュできる理想郷",
                "日常を忘れて{relaxation_type}を楽しめる場所"
            ],
            "unique": [
                "他では味わえない{unique_type}の体験ができる国",
                "{unique_type}で知られる世界でここだけの場所",
                "一生に一度は訪れたい{unique_type}の国"
            ]
        }
    
    def _define_continent_features(self) -> Dict[str, Dict]:
        """大陸別特徴定義"""
        return {
            "AS": {  # Asia
                "name": "アジア",
                "features": ["古代文明", "多様な宗教", "伝統文化", "美食", "急速な発展"],
                "landscapes": ["雄大な山脈", "熱帯雨林", "古都", "現代都市", "棚田"],
                "unique_aspects": ["仏教・ヒンドゥー教文化", "モンスーン気候", "多言語社会"]
            },
            "EU": {  # Europe
                "name": "ヨーロッパ",
                "features": ["歴史的建造物", "芸術文化", "ワイン文化", "王室文化", "クラシック音楽"],
                "landscapes": ["古城", "美しい街並み", "アルプス山脈", "地中海", "森林"],
                "unique_aspects": ["ルネサンス文化", "多様な建築様式", "EU統合"]
            },
            "AF": {  # Africa
                "name": "アフリカ",
                "features": ["野生動物", "部族文化", "古代文明", "音楽・ダンス", "手工芸"],
                "landscapes": ["サバンナ", "砂漠", "熱帯雨林", "大地溝帯", "川"],
                "unique_aspects": ["人類発祥の地", "多様な部族", "野生動物の宝庫"]
            },
            "NA": {  # North America
                "name": "北アメリカ",
                "features": ["多文化社会", "先進技術", "エンターテイメント", "国立公園", "都市文化"],
                "landscapes": ["大峡谷", "大湖", "大平原", "ロッキー山脈", "海岸線"],
                "unique_aspects": ["移民文化", "イノベーション", "スポーツ文化"]
            },
            "SA": {  # South America
                "name": "南アメリカ",
                "features": ["先住民文化", "熱帯自然", "ラテン文化", "音楽・ダンス", "コーヒー文化"],
                "landscapes": ["アマゾン", "アンデス山脈", "パタゴニア", "熱帯雨林", "高原"],
                "unique_aspects": ["インカ文明", "生物多様性", "情熱的な文化"]
            },
            "OC": {  # Oceania
                "name": "オセアニア",
                "features": ["海洋文化", "先住民文化", "サーフィン", "野生動物", "自然保護"],
                "landscapes": ["サンゴ礁", "熱帯島嶼", "砂漠", "雨林", "海岸"],
                "unique_aspects": ["固有種動物", "アボリジニ文化", "海洋国家"]
            }
        }
    
    def _define_population_features(self) -> Dict[str, Dict]:
        """人口規模別特徴定義"""
        return {
            "very_large": {  # 100M+
                "size_type": "大国",
                "features": ["多様性", "経済力", "国際的影響力", "文化的豊富さ"],
                "advantages": ["選択肢の豊富さ", "インフラの充実", "多文化体験"]
            },
            "large": {  # 50M-100M
                "size_type": "大規模国",
                "features": ["地域多様性", "経済発展", "文化的深さ", "観光インフラ"],
                "advantages": ["バランスの良さ", "文化的深み", "適度な規模感"]
            },
            "medium": {  # 10M-50M
                "size_type": "中規模国",
                "features": ["国民的統一感", "文化的独自性", "管理の良さ", "質の高さ"],
                "advantages": ["親しみやすさ", "文化的純粋性", "効率的な社会"]
            },
            "small": {  # 1M-10M
                "size_type": "小国",
                "features": ["親密さ", "独自性", "質の高さ", "特色の明確さ"],
                "advantages": ["特別感", "個性的体験", "きめ細かいサービス"]
            },
            "micro": {  # <1M
                "size_type": "小さな国",
                "features": ["排他性", "プレミアム感", "純粋性", "希少性"],
                "advantages": ["唯一無二の体験", "プライベート感", "特別な思い出"]
            }
        }
    
    def _define_geographic_features(self) -> Dict[str, List[str]]:
        """地理的特徴定義"""
        return {
            "mountain": ["雄大な山脈", "高原地帯", "アルプス風景", "峡谷", "火山"],
            "coastal": ["美しい海岸線", "リゾートビーチ", "港町", "岬", "入江"],
            "island": ["青い海", "サンゴ礁", "トロピカルビーチ", "火山島", "環礁"],
            "forest": ["深い森林", "熱帯雨林", "針葉樹林", "紅葉", "原生林"],
            "desert": ["砂漠の絶景", "オアシス", "砂丘", "岩石砂漠", "塩湖"],
            "river": ["大河", "川沿いの都市", "デルタ地帯", "渓谷", "滝"],
            "lake": ["美しい湖", "湖畔リゾート", "高山湖", "火山湖", "塩湖"],
            "plain": ["広大な平原", "草原", "農業地帯", "ステップ", "サバンナ"]
        }
    
    def _get_country_category(self, raw_data: CountryRawData) -> str:
        """国のカテゴリを判定"""
        if not raw_data.metadata or not raw_data.metadata.population_number:
            return "medium_country"
        
        population = raw_data.metadata.population_number
        
        if population >= 100_000_000:
            return "large_country"
        elif population >= 10_000_000:
            return "medium_country"
        elif population >= 1_000_000:
            return "small_country"
        else:
            return "small_country"
    
    def _get_population_category(self, raw_data: CountryRawData) -> str:
        """人口カテゴリを判定"""
        if not raw_data.metadata or not raw_data.metadata.population_number:
            return "medium"
        
        population = raw_data.metadata.population_number
        
        if population >= 100_000_000:
            return "very_large"
        elif population >= 50_000_000:
            return "large"
        elif population >= 10_000_000:
            return "medium"
        elif population >= 1_000_000:
            return "small"
        else:
            return "micro"
    
    def _get_continent_info(self, raw_data: CountryRawData) -> Dict:
        """大陸情報を取得"""
        if not raw_data.metadata or not raw_data.metadata.continent:
            return self.continent_features["AS"]  # デフォルト
        
        continent_code = raw_data.metadata.continent
        return self.continent_features.get(continent_code, self.continent_features["AS"])
    
    def _generate_description(self, raw_data: CountryRawData) -> str:
        """descriptionを生成"""
        category = self._get_country_category(raw_data)
        continent_info = self._get_continent_info(raw_data)
        
        # テンプレートを選択
        templates = self.description_templates.get(category, self.description_templates["medium_country"])
        template = random.choice(templates)
        
        # 変数を置換
        description = template.format(
            country_name=raw_data.name,
            continent=continent_info["name"],
            population=raw_data.basic.population if raw_data.basic else "多くの人々",
            unique_feature=random.choice(continent_info["features"]),
            landscape=random.choice(continent_info["landscapes"])
        )
        
        return description
    
    def _generate_highlights(self, raw_data: CountryRawData) -> List[CountryHighlight]:
        """highlightsを生成"""
        continent_info = self._get_continent_info(raw_data)
        
        # 4つのハイライトカテゴリから1つずつ選択
        highlight_categories = ["natural_wonders", "cultural_heritage", "modern_attractions", "local_lifestyle"]
        highlights = []
        
        for category in highlight_categories:
            templates = self.highlight_templates[category]
            template_info = random.choice(templates)
            
            # カテゴリに応じた特徴を選択
            if category == "natural_wonders":
                feature = random.choice(continent_info["landscapes"])
                description = f"{feature}の美しさ"
            elif category == "cultural_heritage":
                feature = random.choice(continent_info["features"])
                description = f"{feature}の伝統"
            elif category == "modern_attractions":
                feature = f"{raw_data.basic.capital if raw_data.basic else raw_data.name}の都市部"
                description = "現代的な魅力"
            else:  # local_lifestyle
                feature = f"{raw_data.name}の人々"
                description = "温かいライフスタイル"
            
            highlight_description = template_info["template"].format(
                country_name=raw_data.name,
                natural_feature=feature,
                natural_description=description,
                cultural_feature=feature,
                cultural_description=description,
                modern_feature=feature,
                modern_description=description,
                lifestyle_feature=feature,
                lifestyle_description=description
            )
            
            highlights.append(CountryHighlight(
                title=template_info["title"],
                description=highlight_description
            ))
        
        return highlights
    
    def _generate_why_visit(self, raw_data: CountryRawData) -> str:
        """whyVisitを生成"""
        continent_info = self._get_continent_info(raw_data)
        pop_category = self._get_population_category(raw_data)
        pop_info = self.population_features[pop_category]
        
        # カテゴリを選択
        categories = ["adventure", "culture", "nature", "relaxation", "unique"]
        category = random.choice(categories)
        
        templates = self.why_visit_templates[category]
        template = random.choice(templates)
        
        # カテゴリに応じた特徴を選択
        if category == "adventure":
            feature = random.choice(continent_info["landscapes"])
        elif category == "culture":
            feature = random.choice(continent_info["features"])
        elif category == "nature":
            feature = random.choice(continent_info["landscapes"])
        elif category == "relaxation":
            feature = random.choice(pop_info["advantages"])
        else:  # unique
            feature = random.choice(continent_info["unique_aspects"])
        
        why_visit = template.format(
            adventure_type=feature,
            culture_type=feature,
            nature_type=feature,
            relaxation_type=feature,
            unique_type=feature
        )
        
        return why_visit
    
    def generate_content(self, raw_data: CountryRawData) -> CountryContent:
        """完全なコンテンツを生成"""
        description = self._generate_description(raw_data)
        highlights = self._generate_highlights(raw_data)
        why_visit = self._generate_why_visit(raw_data)
        
        return CountryContent(
            description=description,
            highlights=highlights,
            why_visit=why_visit
        )
    
    def generate_contents_for_countries(self, countries_raw_data: Dict[str, Dict]) -> Dict[str, CountryContent]:
        """複数国のコンテンツを一括生成"""
        contents = {}
        
        for country_code, raw_data_dict in countries_raw_data.items():
            print(f"Generating content for {raw_data_dict.get('nameEn', country_code)}...")
            
            try:
                # 辞書からCountryRawDataオブジェクトを再構築
                raw_data = CountryRawData(
                    name=raw_data_dict.get('name', ''),
                    name_en=raw_data_dict.get('nameEn', ''),
                    code=raw_data_dict.get('code', ''),
                    flag=raw_data_dict.get('flag', ''),
                )
                
                if raw_data_dict.get('basic'):
                    from schemas.country_schema import CountryBasicInfo
                    raw_data.basic = CountryBasicInfo(**raw_data_dict['basic'])
                
                if raw_data_dict.get('coordinates'):
                    from schemas.country_schema import CountryCoordinates
                    raw_data.coordinates = CountryCoordinates(**raw_data_dict['coordinates'])
                
                if raw_data_dict.get('metadata'):
                    from schemas.country_schema import CountryMetadata
                    raw_data.metadata = CountryMetadata(**raw_data_dict['metadata'])
                
                content = self.generate_content(raw_data)
                contents[country_code] = content
                
                print(f"  ✓ Generated content for {raw_data.name}")
                
            except Exception as e:
                print(f"  ✗ Error generating content for {country_code}: {e}")
                continue
        
        return contents
    
    def save_contents_data(self, contents: Dict[str, CountryContent], filename: str = "country_contents.json"):
        """コンテンツデータをJSONファイルに保存"""
        output_file = self.data_dir / filename
        
        # CountryContentオブジェクトを辞書に変換
        contents_dict = {}
        for country_code, content in contents.items():
            contents_dict[country_code] = {
                "description": content.description,
                "highlights": [
                    {"title": h.title, "description": h.description}
                    for h in content.highlights
                ],
                "whyVisit": content.why_visit
            }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(contents_dict, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Contents data saved to {output_file}")
    
    def run_with_raw_data(self, countries_raw_data: Dict[str, Dict]):
        """生データを使用してコンテンツ生成を実行"""
        print("✍️  Content Generator")
        print("=" * 50)
        
        # コンテンツを生成
        print("🔄 Generating contents...")
        contents = self.generate_contents_for_countries(countries_raw_data)
        
        # JSONファイルに保存
        print("\n💾 Saving contents data...")
        self.save_contents_data(contents)
        
        print("\n✅ Content generation completed!")
        print(f"   Generated contents for: {len(contents)} countries")
        
        return contents


if __name__ == "__main__":
    # 単体テスト用
    generator = ContentGenerator()
    
    # テスト用の生データ
    test_data = {
        "jp": {
            "name": "日本",
            "nameEn": "Japan",
            "code": "jp",
            "flag": "🇯🇵",
            "basic": {
                "capital": "東京",
                "population": "1億2,500万人",
                "language": "日本語"
            },
            "coordinates": {
                "lat": 35.6762,
                "lng": 139.6503
            },
            "metadata": {
                "continent": "AS",
                "rarity": 2,
                "population_number": 125000000
            }
        }
    }
    
    generator.run_with_raw_data(test_data)