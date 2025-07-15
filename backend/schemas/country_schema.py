"""
統一国データスキーマ - Python版
TypeScriptスキーマと同期を保つ
"""

from typing import Dict, List, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class ContinentCode(Enum):
    AF = "Africa"
    AS = "Asia"
    EU = "Europe"
    NA = "North America"
    SA = "South America"
    OC = "Oceania"
    AN = "Antarctica"


class RarityLevel(Enum):
    VERY_COMMON = 1
    COMMON = 2
    MODERATE = 3
    RARE = 4
    VERY_RARE = 5


@dataclass
class CountryHighlight:
    title: str
    description: str


@dataclass
class CountryBasicInfo:
    capital: str
    population: str
    language: str


@dataclass
class CountryCoordinates:
    lat: float
    lng: float


@dataclass
class CountryMetadata:
    continent: Optional[str] = None
    rarity: int = 1
    region: Optional[str] = None
    subregion: Optional[str] = None
    population_number: Optional[int] = None
    area: Optional[float] = None
    timezone: Optional[str] = None
    currency: Optional[str] = None
    calling_code: Optional[str] = None


@dataclass
class CountryContent:
    description: str
    highlights: List[CountryHighlight]
    why_visit: str


@dataclass
class CountryImages:
    primary: str
    highlights: List[str] = field(default_factory=list)
    fallback: str = ""


@dataclass
class CountryData:
    name: str  # 日本語名
    name_en: str  # 英語名
    flag: str  # 国旗絵文字
    code: str  # 2文字国コード
    basic: CountryBasicInfo
    coordinates: CountryCoordinates
    metadata: CountryMetadata
    content: CountryContent
    images: CountryImages
    last_updated: str  # ISO 8601形式


@dataclass
class CountryRawData:
    name: str
    name_en: str
    code: str
    flag: str
    basic: Optional[CountryBasicInfo] = None
    coordinates: Optional[CountryCoordinates] = None
    metadata: Optional[CountryMetadata] = None


@dataclass
class GenerationConfig:
    use_ai: bool = True
    languages: List[str] = field(default_factory=lambda: ["ja", "en"])
    image_source: str = "unsplash"  # 'unsplash' | 'local' | 'mixed'
    batch_size: int = 10
    retry_count: int = 3


@dataclass
class GenerationResult:
    success: bool
    country_code: str
    data: Optional[CountryData] = None
    error: Optional[str] = None
    warnings: List[str] = field(default_factory=list)


@dataclass
class ValidationResult:
    valid: bool
    country_code: str
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


# 型変換ユーティリティ
def country_data_to_dict(data: CountryData) -> Dict:
    """CountryDataをJSONシリアライズ可能な辞書に変換"""
    return {
        "name": data.name,
        "nameEn": data.name_en,
        "flag": data.flag,
        "code": data.code,
        "basic": {
            "capital": data.basic.capital,
            "population": data.basic.population,
            "language": data.basic.language
        },
        "coordinates": {
            "lat": data.coordinates.lat,
            "lng": data.coordinates.lng
        },
        "metadata": {
            "continent": data.metadata.continent,
            "rarity": data.metadata.rarity,
            "region": data.metadata.region,
            "subregion": data.metadata.subregion,
            "populationNumber": data.metadata.population_number,
            "area": data.metadata.area,
            "timezone": data.metadata.timezone,
            "currency": data.metadata.currency,
            "callingCode": data.metadata.calling_code
        },
        "content": {
            "description": data.content.description,
            "highlights": [
                {"title": h.title, "description": h.description}
                for h in data.content.highlights
            ],
            "whyVisit": data.content.why_visit
        },
        "images": {
            "primary": data.images.primary,
            "highlights": data.images.highlights,
            "fallback": data.images.fallback
        },
        "lastUpdated": data.last_updated
    }


def dict_to_country_data(data: Dict) -> CountryData:
    """辞書からCountryDataオブジェクトを生成"""
    return CountryData(
        name=data["name"],
        name_en=data["nameEn"],
        flag=data["flag"],
        code=data["code"],
        basic=CountryBasicInfo(**data["basic"]),
        coordinates=CountryCoordinates(**data["coordinates"]),
        metadata=CountryMetadata(**data["metadata"]),
        content=CountryContent(
            description=data["content"]["description"],
            highlights=[
                CountryHighlight(**h) for h in data["content"]["highlights"]
            ],
            why_visit=data["content"]["whyVisit"]
        ),
        images=CountryImages(**data["images"]),
        last_updated=data["lastUpdated"]
    )


# バリデーション関数
def validate_country_data(data: CountryData) -> ValidationResult:
    """CountryDataの妥当性を検証"""
    errors = []
    warnings = []
    
    # 必須フィールドチェック
    if not data.name:
        errors.append("Name is required")
    if not data.name_en:
        errors.append("English name is required")
    if not data.code or len(data.code) != 2:
        errors.append("Valid 2-letter country code is required")
    if not data.flag:
        errors.append("Flag emoji is required")
    
    # 基本情報チェック
    if not data.basic.capital:
        errors.append("Capital is required")
    if not data.basic.population:
        errors.append("Population is required")
    if not data.basic.language:
        errors.append("Language is required")
    
    # 座標チェック
    if not (-90 <= data.coordinates.lat <= 90):
        errors.append("Latitude must be between -90 and 90")
    if not (-180 <= data.coordinates.lng <= 180):
        errors.append("Longitude must be between -180 and 180")
    
    # コンテンツチェック
    if not data.content.description:
        errors.append("Description is required")
    if len(data.content.highlights) != 4:
        errors.append("Exactly 4 highlights are required")
    if not data.content.why_visit:
        errors.append("WhyVisit is required")
    
    # 画像チェック
    if not data.images.primary:
        errors.append("Primary image URL is required")
    
    # 文字数チェック
    if len(data.content.description) > 200:
        warnings.append("Description is longer than 200 characters")
    if len(data.content.why_visit) > 50:
        warnings.append("WhyVisit is longer than 50 characters")
    
    return ValidationResult(
        valid=len(errors) == 0,
        country_code=data.code,
        errors=errors,
        warnings=warnings
    )