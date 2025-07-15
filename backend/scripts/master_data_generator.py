"""
マスターデータ生成スクリプト
すべてのコンポーネントを統合して243カ国の完全なデータセットを生成
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import sys
import argparse

# 親ディレクトリをパスに追加
sys.path.append(str(Path(__file__).parent.parent))

from schemas.country_schema import (
    CountryData, CountryImages, CountryContent, GenerationConfig,
    country_data_to_dict, dict_to_country_data
)

from enhanced_country_data_generator import EnhancedCountryDataGenerator
from image_url_generator import ImageUrlGenerator
from content_generator import ContentGenerator
from data_quality_manager import DataQualityManager


class MasterDataGenerator:
    """マスターデータ生成クラス"""
    
    def __init__(self, config: Optional[GenerationConfig] = None):
        self.base_dir = Path(__file__).parent.parent
        self.data_dir = self.base_dir / "data"
        self.output_dir = self.data_dir / "generated"
        self.output_dir.mkdir(exist_ok=True)
        
        # 設定
        self.config = config or GenerationConfig()
        
        # 各コンポーネントを初期化
        self.country_generator = EnhancedCountryDataGenerator()
        self.image_generator = ImageUrlGenerator()
        self.content_generator = ContentGenerator()
        self.quality_manager = DataQualityManager()
        
        # 進捗追跡
        self.progress = {
            "total_countries": 0,
            "completed_countries": 0,
            "failed_countries": 0,
            "current_step": "",
            "steps": [
                "基本情報取得",
                "画像URL生成",
                "コンテンツ生成",
                "データ統合",
                "品質チェック",
                "最終出力"
            ]
        }
    
    def print_progress(self, message: str, step: Optional[str] = None):
        """進捗表示"""
        if step:
            self.progress["current_step"] = step
        
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {message}")
        
        if self.progress["total_countries"] > 0:
            completion = (self.progress["completed_countries"] / self.progress["total_countries"]) * 100
            print(f"  進捗: {self.progress['completed_countries']}/{self.progress['total_countries']} ({completion:.1f}%)")
    
    def generate_raw_data(self) -> Dict[str, Dict]:
        """基本情報の生成"""
        self.print_progress("🔍 基本情報を取得中...", "基本情報取得")
        
        raw_data_list = self.country_generator.generate_raw_data_from_existing_metadata()
        self.progress["total_countries"] = len(raw_data_list)
        
        # 辞書形式に変換
        raw_data_dict = {}
        for raw_data in raw_data_list:
            raw_data_dict[raw_data.code] = {
                "name": raw_data.name,
                "nameEn": raw_data.name_en,
                "code": raw_data.code,
                "flag": raw_data.flag,
                "basic": {
                    "capital": raw_data.basic.capital,
                    "population": raw_data.basic.population,
                    "language": raw_data.basic.language
                } if raw_data.basic else {},
                "coordinates": {
                    "lat": raw_data.coordinates.lat,
                    "lng": raw_data.coordinates.lng
                } if raw_data.coordinates else {},
                "metadata": {
                    "continent": raw_data.metadata.continent,
                    "rarity": raw_data.metadata.rarity,
                    "region": raw_data.metadata.region,
                    "subregion": raw_data.metadata.subregion,
                    "populationNumber": raw_data.metadata.population_number,
                    "area": raw_data.metadata.area,
                    "timezone": raw_data.metadata.timezone,
                    "currency": raw_data.metadata.currency,
                    "callingCode": raw_data.metadata.calling_code
                } if raw_data.metadata else {}
            }
        
        self.progress["completed_countries"] = len(raw_data_dict)
        self.print_progress(f"✅ 基本情報取得完了: {len(raw_data_dict)}カ国")
        
        return raw_data_dict
    
    def generate_images(self, raw_data_dict: Dict[str, Dict], unsplash_key: Optional[str] = None) -> Dict[str, Dict]:
        """画像URLの生成"""
        self.print_progress("🖼️  画像URLを生成中...", "画像URL生成")
        
        try:
            country_images = self.image_generator.generate_country_images(raw_data_dict)
            
            # 辞書形式に変換
            images_dict = {}
            for country_code, images in country_images.items():
                images_dict[country_code] = {
                    "primary": images.primary,
                    "highlights": images.highlights,
                    "fallback": images.fallback
                }
            
            self.print_progress(f"✅ 画像URL生成完了: {len(images_dict)}カ国")
            return images_dict
            
        except Exception as e:
            self.print_progress(f"❌ 画像URL生成でエラー: {str(e)}")
            
            # フォールバック画像を使用
            images_dict = {}
            for country_code in raw_data_dict.keys():
                images_dict[country_code] = {
                    "primary": self.image_generator.fallback_image,
                    "highlights": [self.image_generator.fallback_image] * 4,
                    "fallback": self.image_generator.fallback_image
                }
            
            return images_dict
    
    def generate_contents(self, raw_data_dict: Dict[str, Dict]) -> Dict[str, Dict]:
        """コンテンツの生成"""
        self.print_progress("✍️  コンテンツを生成中...", "コンテンツ生成")
        
        try:
            contents = self.content_generator.generate_contents_for_countries(raw_data_dict)
            
            # 辞書形式に変換
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
            
            self.print_progress(f"✅ コンテンツ生成完了: {len(contents_dict)}カ国")
            return contents_dict
            
        except Exception as e:
            self.print_progress(f"❌ コンテンツ生成でエラー: {str(e)}")
            raise
    
    def merge_data(self, raw_data_dict: Dict[str, Dict], images_dict: Dict[str, Dict], 
                   contents_dict: Dict[str, Dict]) -> Dict[str, Dict]:
        """データを統合"""
        self.print_progress("🔄 データを統合中...", "データ統合")
        
        merged_data = {}
        current_timestamp = datetime.now().isoformat()
        
        for country_code, raw_data in raw_data_dict.items():
            try:
                # 各データを統合
                merged_country = {
                    "name": raw_data.get("name", ""),
                    "nameEn": raw_data.get("nameEn", ""),
                    "flag": raw_data.get("flag", ""),
                    "code": raw_data.get("code", ""),
                    "basic": raw_data.get("basic", {}),
                    "coordinates": raw_data.get("coordinates", {}),
                    "metadata": raw_data.get("metadata", {}),
                    "content": contents_dict.get(country_code, {
                        "description": f"{raw_data.get('name', '')}の魅力的な国です。",
                        "highlights": [
                            {"title": "文化的魅力", "description": "この国独特の文化を体験できます。"},
                            {"title": "自然の美しさ", "description": "美しい自然景観を楽しめます。"},
                            {"title": "歴史的価値", "description": "豊かな歴史を感じることができます。"},
                            {"title": "地域の特色", "description": "地域ならではの特色があります。"}
                        ],
                        "whyVisit": "独特な体験ができる魅力的な国"
                    }),
                    "images": images_dict.get(country_code, {
                        "primary": self.image_generator.fallback_image,
                        "highlights": [self.image_generator.fallback_image] * 4,
                        "fallback": self.image_generator.fallback_image
                    }),
                    "lastUpdated": current_timestamp
                }
                
                merged_data[country_code] = merged_country
                
            except Exception as e:
                self.print_progress(f"❌ {country_code}の統合でエラー: {str(e)}")
                self.progress["failed_countries"] += 1
                continue
        
        self.print_progress(f"✅ データ統合完了: {len(merged_data)}カ国")
        return merged_data
    
    def run_quality_check(self, merged_data: Dict[str, Dict]) -> Dict[str, Dict]:
        """品質チェックと修正"""
        self.print_progress("🔍 品質チェック中...", "品質チェック")
        
        try:
            # 品質チェック実行
            quality_report = self.quality_manager.validate_countries_data(merged_data)
            
            # レポート保存
            self.quality_manager.save_quality_report(quality_report)
            
            # 結果サマリー
            valid_ratio = (quality_report.valid_countries / quality_report.total_countries) * 100
            self.print_progress(f"✅ 品質チェック完了: {quality_report.valid_countries}/{quality_report.total_countries} ({valid_ratio:.1f}%) 有効")
            
            if quality_report.invalid_countries > 0:
                self.print_progress(f"⚠️  {quality_report.invalid_countries}カ国でエラーが検出されました")
            
            return merged_data
            
        except Exception as e:
            self.print_progress(f"❌ 品質チェックでエラー: {str(e)}")
            return merged_data
    
    def save_final_data(self, merged_data: Dict[str, Dict]):
        """最終データの保存"""
        self.print_progress("💾 最終データを保存中...", "最終出力")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 完全データセット保存
        complete_file = self.output_dir / f"complete_country_data_{timestamp}.json"
        with open(complete_file, 'w', encoding='utf-8') as f:
            json.dump(merged_data, f, ensure_ascii=False, indent=2)
        
        # フロントエンド用の軽量版作成
        frontend_data = {}
        for country_code, data in merged_data.items():
            frontend_data[country_code] = {
                "name": data["name"],
                "nameEn": data["nameEn"],
                "flag": data["flag"],
                "code": data["code"],
                "basic": data["basic"],
                "coordinates": data["coordinates"],
                "description": data["content"]["description"],
                "highlights": data["content"]["highlights"],
                "whyVisit": data["content"]["whyVisit"]
            }
        
        frontend_file = self.output_dir / f"frontend_country_data_{timestamp}.json"
        with open(frontend_file, 'w', encoding='utf-8') as f:
            json.dump(frontend_data, f, ensure_ascii=False, indent=2)
        
        # TypeScript countries.tsファイル生成
        self.generate_typescript_countries_file(frontend_data)
        
        # 統計情報保存
        stats = {
            "generation_timestamp": datetime.now().isoformat(),
            "total_countries": len(merged_data),
            "successful_countries": len(merged_data),
            "failed_countries": self.progress["failed_countries"],
            "continents": {},
            "rarity_distribution": {}
        }
        
        # 大陸別統計
        for data in merged_data.values():
            continent = data.get("metadata", {}).get("continent", "Unknown")
            stats["continents"][continent] = stats["continents"].get(continent, 0) + 1
            
            rarity = data.get("metadata", {}).get("rarity", 1)
            stats["rarity_distribution"][str(rarity)] = stats["rarity_distribution"].get(str(rarity), 0) + 1
        
        stats_file = self.output_dir / f"generation_stats_{timestamp}.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        self.print_progress(f"✅ 最終データ保存完了:")
        self.print_progress(f"   完全データ: {complete_file}")
        self.print_progress(f"   フロントエンド用: {frontend_file}")
        self.print_progress(f"   統計情報: {stats_file}")
    
    def generate_typescript_countries_file(self, frontend_data: Dict[str, Dict]):
        """TypeScript countries.tsファイル生成"""
        output_file = self.base_dir.parent / "project" / "src" / "data" / "countries.ts"
        
        content = """import { CountryDataMap } from '../types/country';

export const countryData: CountryDataMap = {
"""
        
        for country_code, data in frontend_data.items():
            content += f"  '{country_code}': {{\n"
            content += f"    name: '{data['name']}',\n"
            content += f"    nameEn: '{data['nameEn']}',\n"
            content += f"    flag: '{data['flag']}',\n"
            content += f"    code: '{data['code']}',\n"
            content += f"    basic: {{\n"
            content += f"      capital: '{data['basic']['capital']}',\n"
            content += f"      population: '{data['basic']['population']}',\n"
            content += f"      language: '{data['basic']['language']}'\n"
            content += f"    }},\n"
            content += f"    coordinates: {{\n"
            content += f"      lat: {data['coordinates']['lat']},\n"
            content += f"      lng: {data['coordinates']['lng']}\n"
            content += f"    }},\n"
            content += f"    description: '{data['description']}',\n"
            content += f"    highlights: [\n"
            
            for highlight in data['highlights']:
                content += f"      {{\n"
                content += f"        title: '{highlight['title']}',\n"
                content += f"        description: '{highlight['description']}'\n"
                content += f"      }},\n"
            
            content += f"    ],\n"
            content += f"    whyVisit: '{data['whyVisit']}'\n"
            content += f"  }},\n\n"
        
        content += "};\n"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        self.print_progress(f"✅ TypeScript countries.tsファイル生成完了: {output_file}")
    
    def run(self, unsplash_key: Optional[str] = None):
        """メイン実行関数"""
        self.print_progress("🚀 マスターデータ生成を開始します")
        self.print_progress("=" * 60)
        
        start_time = datetime.now()
        
        try:
            # Step 1: 基本情報取得
            raw_data_dict = self.generate_raw_data()
            
            # Step 2: 画像URL生成
            images_dict = self.generate_images(raw_data_dict, unsplash_key)
            
            # Step 3: コンテンツ生成
            contents_dict = self.generate_contents(raw_data_dict)
            
            # Step 4: データ統合
            merged_data = self.merge_data(raw_data_dict, images_dict, contents_dict)
            
            # Step 5: 品質チェック
            merged_data = self.run_quality_check(merged_data)
            
            # Step 6: 最終出力
            self.save_final_data(merged_data)
            
            # 完了レポート
            end_time = datetime.now()
            duration = end_time - start_time
            
            self.print_progress("🎉 マスターデータ生成完了！")
            self.print_progress(f"   処理時間: {duration}")
            self.print_progress(f"   生成国数: {len(merged_data)}")
            self.print_progress(f"   失敗国数: {self.progress['failed_countries']}")
            self.print_progress(f"   成功率: {((len(merged_data) / self.progress['total_countries']) * 100):.1f}%")
            
        except Exception as e:
            self.print_progress(f"❌ 生成プロセスでエラーが発生しました: {str(e)}")
            raise


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="Master Country Data Generator")
    parser.add_argument("--unsplash-key", help="Unsplash API key for image generation")
    parser.add_argument("--batch-size", type=int, default=10, help="Batch size for processing")
    parser.add_argument("--retry-count", type=int, default=3, help="Number of retries for failed requests")
    
    args = parser.parse_args()
    
    # 設定作成
    config = GenerationConfig(
        use_ai=True,
        languages=["ja", "en"],
        image_source="unsplash",
        batch_size=args.batch_size,
        retry_count=args.retry_count
    )
    
    # 生成実行
    generator = MasterDataGenerator(config)
    generator.run(args.unsplash_key)


if __name__ == "__main__":
    main()