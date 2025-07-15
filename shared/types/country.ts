// 統一国データスキーマ - バックエンドとフロントエンドで共通使用

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

export interface CountryMetadata {
  continent: string | null;
  rarity: number; // 1-5 (1=common, 5=rare)
  region?: string;
  subregion?: string;
  populationNumber?: number; // 数値版人口
  area?: number; // 面積（km²）
  timezone?: string;
  currency?: string;
  callingCode?: string;
}

export interface CountryContent {
  description: string;
  highlights: CountryHighlight[];
  whyVisit: string;
}

export interface CountryImages {
  primary: string; // メイン画像URL
  highlights: string[]; // ハイライト用画像URL（4つ）
  fallback: string; // フォールバック画像URL
}

// 完全な国データ構造
export interface CountryData {
  name: string; // 日本語名
  nameEn: string; // 英語名
  flag: string; // 国旗絵文字
  code: string; // 2文字国コード
  basic: CountryBasicInfo;
  coordinates: CountryCoordinates;
  metadata: CountryMetadata;
  content: CountryContent;
  images: CountryImages;
  lastUpdated: string; // ISO 8601形式の更新日時
}

// 自動生成用の中間データ構造
export interface CountryRawData {
  name: string;
  nameEn: string;
  code: string;
  flag: string;
  basic: Partial<CountryBasicInfo>;
  coordinates: Partial<CountryCoordinates>;
  metadata: Partial<CountryMetadata>;
}

// フロントエンド用の軽量版（後方互換性維持）
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

export type CountryDataMap = Record<string, CountryData>;
export type CountryDataMapLegacy = Record<string, CountryDataLegacy>;

// 生成設定
export interface GenerationConfig {
  useAI: boolean; // AI生成を使用するか
  languages: string[]; // 対応言語
  imageSource: 'unsplash' | 'local' | 'mixed'; // 画像ソース
  batchSize: number; // 一括処理サイズ
  retryCount: number; // リトライ回数
}

// 生成結果
export interface GenerationResult {
  success: boolean;
  countryCode: string;
  data?: CountryData;
  error?: string;
  warnings?: string[];
}

// バリデーション結果
export interface ValidationResult {
  valid: boolean;
  countryCode: string;
  errors: string[];
  warnings: string[];
}

// 大陸コード定義
export const CONTINENTS = {
  AF: 'Africa',
  AS: 'Asia', 
  EU: 'Europe',
  NA: 'North America',
  SA: 'South America',
  OC: 'Oceania',
  AN: 'Antarctica'
} as const;

export type ContinentCode = keyof typeof CONTINENTS;

// 意外度レベル定義
export const RARITY_LEVELS = {
  1: 'Very Common',
  2: 'Common',
  3: 'Moderate',
  4: 'Rare',
  5: 'Very Rare'
} as const;

export type RarityLevel = keyof typeof RARITY_LEVELS;