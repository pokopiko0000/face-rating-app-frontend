"""
データ品質管理システム
生成されたデータの品質チェック、修正、レポート生成
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
import sys
import os

# 親ディレクトリをパスに追加
sys.path.append(str(Path(__file__).parent.parent))

from schemas.country_schema import (
    CountryData, ValidationResult, validate_country_data,
    dict_to_country_data, country_data_to_dict
)


@dataclass
class QualityReport:
    """品質レポート"""
    timestamp: str
    total_countries: int
    valid_countries: int
    invalid_countries: int
    countries_with_warnings: int
    error_summary: Dict[str, int] = field(default_factory=dict)
    warning_summary: Dict[str, int] = field(default_factory=dict)
    detailed_results: List[ValidationResult] = field(default_factory=list)


class DataQualityManager:
    """データ品質管理クラス"""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent.parent
        self.data_dir = self.base_dir / "data"
        self.reports_dir = self.data_dir / "quality_reports"
        self.reports_dir.mkdir(exist_ok=True)
        
        # 品質基準を定義
        self.quality_standards = self._define_quality_standards()
        
        # 修正可能なエラーのパターン
        self.fixable_patterns = self._define_fixable_patterns()
    
    def _define_quality_standards(self) -> Dict[str, Any]:
        """品質基準を定義"""
        return {
            "description": {
                "min_length": 50,
                "max_length": 250,
                "forbidden_words": ["TODO", "FIXME", "placeholder", "example"],
                "required_elements": ["国名", "人口", "特徴"]
            },
            "highlights": {
                "count": 4,
                "title_max_length": 30,
                "description_min_length": 30,
                "description_max_length": 150,
                "uniqueness": True  # タイトルが重複していないか
            },
            "why_visit": {
                "min_length": 15,
                "max_length": 60,
                "forbidden_words": ["TODO", "FIXME", "placeholder"]
            },
            "basic_info": {
                "capital_required": True,
                "population_format": r"^[\d,]+[万千億]?人$",
                "language_required": True
            },
            "coordinates": {
                "lat_range": (-90, 90),
                "lng_range": (-180, 180),
                "precision": 4  # 小数点以下の桁数
            },
            "images": {
                "url_pattern": r"^https://.*\.(jpg|jpeg|png|webp)(\?.*)?$",
                "required_params": ["w=", "h="],
                "min_dimensions": (800, 600)
            }
        }
    
    def _define_fixable_patterns(self) -> Dict[str, Dict]:
        """修正可能なエラーパターンを定義"""
        return {
            "population_format": {
                "patterns": [
                    (r"(\d+)million", r"\1百万"),
                    (r"(\d+)billion", r"\1十億"),
                    (r"(\d+),(\d+),(\d+)", r"\1\2\3"),
                    (r"(\d+)k", r"\1千")
                ],
                "field": "basic.population"
            },
            "coordinate_precision": {
                "max_decimal_places": 4,
                "fields": ["coordinates.lat", "coordinates.lng"]
            },
            "text_cleanup": {
                "patterns": [
                    (r"\s+", " "),  # 複数の空白を1つに
                    (r"^\s+|\s+$", ""),  # 前後の空白を削除
                    (r"。。+", "。"),  # 複数の句点を1つに
                    (r"！！+", "！"),  # 複数の感嘆符を1つに
                ],
                "fields": ["content.description", "content.why_visit"]
            }
        }
    
    def _validate_description(self, description: str, country_code: str) -> Tuple[List[str], List[str]]:
        """description品質チェック"""
        errors = []
        warnings = []
        standards = self.quality_standards["description"]
        
        # 文字数チェック
        if len(description) < standards["min_length"]:
            errors.append(f"Description too short ({len(description)} < {standards['min_length']})")
        elif len(description) > standards["max_length"]:
            warnings.append(f"Description too long ({len(description)} > {standards['max_length']})")
        
        # 禁止語チェック
        for word in standards["forbidden_words"]:
            if word.lower() in description.lower():
                errors.append(f"Forbidden word '{word}' found in description")
        
        # 必須要素チェック（簡易版）
        if "人" not in description:
            warnings.append("Population information might be missing")
        
        return errors, warnings
    
    def _validate_highlights(self, highlights: List[Dict], country_code: str) -> Tuple[List[str], List[str]]:
        """highlights品質チェック"""
        errors = []
        warnings = []
        standards = self.quality_standards["highlights"]
        
        # 数量チェック
        if len(highlights) != standards["count"]:
            errors.append(f"Wrong number of highlights ({len(highlights)} != {standards['count']})")
        
        # タイトルの重複チェック
        titles = [h.get("title", "") for h in highlights]
        if len(set(titles)) != len(titles):
            errors.append("Duplicate highlight titles found")
        
        # 各ハイライトの品質チェック
        for i, highlight in enumerate(highlights):
            title = highlight.get("title", "")
            description = highlight.get("description", "")
            
            # タイトル長チェック
            if len(title) > standards["title_max_length"]:
                warnings.append(f"Highlight {i+1} title too long ({len(title)} > {standards['title_max_length']})")
            
            # 説明文長チェック
            if len(description) < standards["description_min_length"]:
                errors.append(f"Highlight {i+1} description too short ({len(description)} < {standards['description_min_length']})")
            elif len(description) > standards["description_max_length"]:
                warnings.append(f"Highlight {i+1} description too long ({len(description)} > {standards['description_max_length']})")
        
        return errors, warnings
    
    def _validate_why_visit(self, why_visit: str, country_code: str) -> Tuple[List[str], List[str]]:
        """whyVisit品質チェック"""
        errors = []
        warnings = []
        standards = self.quality_standards["why_visit"]
        
        # 文字数チェック
        if len(why_visit) < standards["min_length"]:
            errors.append(f"WhyVisit too short ({len(why_visit)} < {standards['min_length']})")
        elif len(why_visit) > standards["max_length"]:
            warnings.append(f"WhyVisit too long ({len(why_visit)} > {standards['max_length']})")
        
        # 禁止語チェック
        for word in standards["forbidden_words"]:
            if word.lower() in why_visit.lower():
                errors.append(f"Forbidden word '{word}' found in whyVisit")
        
        return errors, warnings
    
    def _validate_basic_info(self, basic: Dict, country_code: str) -> Tuple[List[str], List[str]]:
        """基本情報品質チェック"""
        errors = []
        warnings = []
        standards = self.quality_standards["basic_info"]
        
        # 首都チェック
        if standards["capital_required"] and not basic.get("capital"):
            errors.append("Capital is required")
        
        # 人口形式チェック
        population = basic.get("population", "")
        if population and not re.match(standards["population_format"], population):
            warnings.append(f"Population format might be incorrect: {population}")
        
        # 言語チェック
        if standards["language_required"] and not basic.get("language"):
            errors.append("Language is required")
        
        return errors, warnings
    
    def _validate_coordinates(self, coordinates: Dict, country_code: str) -> Tuple[List[str], List[str]]:
        """座標品質チェック"""
        errors = []
        warnings = []
        standards = self.quality_standards["coordinates"]
        
        lat = coordinates.get("lat", 0)
        lng = coordinates.get("lng", 0)
        
        # 範囲チェック
        lat_range = standards["lat_range"]
        lng_range = standards["lng_range"]
        
        if not (lat_range[0] <= lat <= lat_range[1]):
            errors.append(f"Latitude out of range: {lat}")
        
        if not (lng_range[0] <= lng <= lng_range[1]):
            errors.append(f"Longitude out of range: {lng}")
        
        # 精度チェック
        if len(str(lat).split('.')[-1]) > standards["precision"]:
            warnings.append(f"Latitude precision too high: {lat}")
        
        if len(str(lng).split('.')[-1]) > standards["precision"]:
            warnings.append(f"Longitude precision too high: {lng}")
        
        return errors, warnings
    
    def _validate_images(self, images: Dict, country_code: str) -> Tuple[List[str], List[str]]:
        """画像品質チェック"""
        errors = []
        warnings = []
        standards = self.quality_standards["images"]
        
        primary_url = images.get("primary", "")
        
        # URL形式チェック
        if primary_url and not re.match(standards["url_pattern"], primary_url, re.IGNORECASE):
            errors.append(f"Primary image URL format invalid: {primary_url}")
        
        # 必須パラメータチェック
        for param in standards["required_params"]:
            if param not in primary_url:
                warnings.append(f"Required parameter '{param}' missing in primary image URL")
        
        return errors, warnings
    
    def _enhanced_validate_country_data(self, data: Dict, country_code: str) -> ValidationResult:
        """拡張データ品質チェック"""
        errors = []
        warnings = []
        
        # 基本バリデーション
        try:
            country_data = dict_to_country_data(data)
            basic_validation = validate_country_data(country_data)
            errors.extend(basic_validation.errors)
            warnings.extend(basic_validation.warnings)
        except Exception as e:
            errors.append(f"Data structure validation failed: {str(e)}")
            return ValidationResult(
                valid=False,
                country_code=country_code,
                errors=errors,
                warnings=warnings
            )
        
        # 詳細品質チェック
        content = data.get("content", {})
        
        # Description チェック
        description = content.get("description", "")
        if description:
            desc_errors, desc_warnings = self._validate_description(description, country_code)
            errors.extend(desc_errors)
            warnings.extend(desc_warnings)
        
        # Highlights チェック
        highlights = content.get("highlights", [])
        if highlights:
            hl_errors, hl_warnings = self._validate_highlights(highlights, country_code)
            errors.extend(hl_errors)
            warnings.extend(hl_warnings)
        
        # WhyVisit チェック
        why_visit = content.get("whyVisit", "")
        if why_visit:
            wv_errors, wv_warnings = self._validate_why_visit(why_visit, country_code)
            errors.extend(wv_errors)
            warnings.extend(wv_warnings)
        
        # Basic Info チェック
        basic = data.get("basic", {})
        if basic:
            basic_errors, basic_warnings = self._validate_basic_info(basic, country_code)
            errors.extend(basic_errors)
            warnings.extend(basic_warnings)
        
        # Coordinates チェック
        coordinates = data.get("coordinates", {})
        if coordinates:
            coord_errors, coord_warnings = self._validate_coordinates(coordinates, country_code)
            errors.extend(coord_errors)
            warnings.extend(coord_warnings)
        
        # Images チェック
        images = data.get("images", {})
        if images:
            img_errors, img_warnings = self._validate_images(images, country_code)
            errors.extend(img_errors)
            warnings.extend(img_warnings)
        
        return ValidationResult(
            valid=len(errors) == 0,
            country_code=country_code,
            errors=errors,
            warnings=warnings
        )
    
    def _apply_auto_fixes(self, data: Dict, country_code: str) -> Dict:
        """自動修正を適用"""
        fixed_data = data.copy()
        
        # テキストクリーンアップ
        text_patterns = self.fixable_patterns["text_cleanup"]["patterns"]
        for field in self.fixable_patterns["text_cleanup"]["fields"]:
            field_parts = field.split(".")
            if len(field_parts) == 2:
                section, key = field_parts
                if section in fixed_data and key in fixed_data[section]:
                    text = fixed_data[section][key]
                    for pattern, replacement in text_patterns:
                        text = re.sub(pattern, replacement, text)
                    fixed_data[section][key] = text
        
        # 座標精度修正
        if "coordinates" in fixed_data:
            coords = fixed_data["coordinates"]
            max_places = self.fixable_patterns["coordinate_precision"]["max_decimal_places"]
            
            if "lat" in coords:
                coords["lat"] = round(coords["lat"], max_places)
            if "lng" in coords:
                coords["lng"] = round(coords["lng"], max_places)
        
        return fixed_data
    
    def validate_countries_data(self, countries_data: Dict[str, Dict]) -> QualityReport:
        """複数国データの品質チェック"""
        results = []
        error_counts = {}
        warning_counts = {}
        
        print("🔍 Starting data quality validation...")
        
        for country_code, data in countries_data.items():
            print(f"  Validating {country_code}...")
            
            # 自動修正を適用
            fixed_data = self._apply_auto_fixes(data, country_code)
            
            # 品質チェック実行
            validation_result = self._enhanced_validate_country_data(fixed_data, country_code)
            results.append(validation_result)
            
            # エラー・警告の集計
            for error in validation_result.errors:
                error_type = error.split(':')[0] if ':' in error else error
                error_counts[error_type] = error_counts.get(error_type, 0) + 1
            
            for warning in validation_result.warnings:
                warning_type = warning.split(':')[0] if ':' in warning else warning
                warning_counts[warning_type] = warning_counts.get(warning_type, 0) + 1
            
            # 修正されたデータを元に戻す
            if fixed_data != data:
                countries_data[country_code] = fixed_data
        
        # レポート作成
        valid_count = sum(1 for r in results if r.valid)
        invalid_count = len(results) - valid_count
        warnings_count = sum(1 for r in results if r.warnings)
        
        report = QualityReport(
            timestamp=datetime.now().isoformat(),
            total_countries=len(results),
            valid_countries=valid_count,
            invalid_countries=invalid_count,
            countries_with_warnings=warnings_count,
            error_summary=error_counts,
            warning_summary=warning_counts,
            detailed_results=results
        )
        
        return report
    
    def generate_quality_report(self, report: QualityReport) -> str:
        """品質レポートを生成"""
        report_content = f"""
# データ品質レポート
生成日時: {report.timestamp}

## 概要
- 総国数: {report.total_countries}
- 有効な国: {report.valid_countries} ({report.valid_countries/report.total_countries*100:.1f}%)
- 無効な国: {report.invalid_countries} ({report.invalid_countries/report.total_countries*100:.1f}%)
- 警告のある国: {report.countries_with_warnings} ({report.countries_with_warnings/report.total_countries*100:.1f}%)

## エラー統計
"""
        
        if report.error_summary:
            for error_type, count in sorted(report.error_summary.items()):
                report_content += f"- {error_type}: {count}件\n"
        else:
            report_content += "エラーなし\n"
        
        report_content += "\n## 警告統計\n"
        
        if report.warning_summary:
            for warning_type, count in sorted(report.warning_summary.items()):
                report_content += f"- {warning_type}: {count}件\n"
        else:
            report_content += "警告なし\n"
        
        # 詳細結果（エラーのある国のみ）
        error_countries = [r for r in report.detailed_results if not r.valid]
        if error_countries:
            report_content += "\n## エラー詳細\n"
            for result in error_countries:
                report_content += f"\n### {result.country_code}\n"
                for error in result.errors:
                    report_content += f"- ❌ {error}\n"
                for warning in result.warnings:
                    report_content += f"- ⚠️ {warning}\n"
        
        return report_content
    
    def save_quality_report(self, report: QualityReport, filename: Optional[str] = None):
        """品質レポートをファイルに保存"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"quality_report_{timestamp}.md"
        
        report_file = self.reports_dir / filename
        report_content = self.generate_quality_report(report)
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        # JSON版も保存
        json_file = self.reports_dir / filename.replace('.md', '.json')
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump({
                "timestamp": report.timestamp,
                "total_countries": report.total_countries,
                "valid_countries": report.valid_countries,
                "invalid_countries": report.invalid_countries,
                "countries_with_warnings": report.countries_with_warnings,
                "error_summary": report.error_summary,
                "warning_summary": report.warning_summary,
                "detailed_results": [
                    {
                        "country_code": r.country_code,
                        "valid": r.valid,
                        "errors": r.errors,
                        "warnings": r.warnings
                    } for r in report.detailed_results
                ]
            }, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Quality report saved to {report_file}")
        print(f"✅ JSON report saved to {json_file}")
    
    def run_quality_check(self, data_file: str = "enhanced_country_data.json"):
        """品質チェックを実行"""
        print("🔍 Data Quality Manager")
        print("=" * 50)
        
        # データを読み込み
        data_path = self.data_dir / data_file
        if not data_path.exists():
            print(f"❌ Data file not found: {data_path}")
            return
        
        with open(data_path, 'r', encoding='utf-8') as f:
            countries_data = json.load(f)
        
        # 品質チェック実行
        report = self.validate_countries_data(countries_data)
        
        # レポート生成・保存
        self.save_quality_report(report)
        
        # 結果サマリー表示
        print(f"\n📊 Quality Check Summary:")
        print(f"   Total countries: {report.total_countries}")
        print(f"   Valid countries: {report.valid_countries} ({report.valid_countries/report.total_countries*100:.1f}%)")
        print(f"   Invalid countries: {report.invalid_countries}")
        print(f"   Countries with warnings: {report.countries_with_warnings}")
        
        if report.error_summary:
            print(f"\n❌ Top errors:")
            for error_type, count in sorted(report.error_summary.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"   {error_type}: {count} countries")
        
        if report.warning_summary:
            print(f"\n⚠️  Top warnings:")
            for warning_type, count in sorted(report.warning_summary.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"   {warning_type}: {count} countries")
        
        return report


if __name__ == "__main__":
    manager = DataQualityManager()
    manager.run_quality_check()