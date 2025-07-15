// 国データ型定義 - フロントエンド用

export interface CountryHighlight {
  title: string;
  description: string;
}

export interface CountryBasicInfo {
  capital: string;
  population: string;
  language: string;
}

export interface CountryCoordinates {
  lat: number;
  lng: number;
}

// 現在使用されている国データ構造（legacy format）
export interface CountryDataLegacy {
  name: string;
  nameEn: string;
  flag: string;
  code: string;
  basic: CountryBasicInfo;
  coordinates: CountryCoordinates;
  description: string;
  highlights: CountryHighlight[];
  whyVisit: string;
}

export type CountryDataMap = Record<string, CountryDataLegacy>;