// 統一型定義エクスポート - プロジェクト全体で使用

// 国データ関連型定義
export * from './country';

// 有名人データ型定義
export interface Celebrity {
  id: string;
  name: string;
  nationality: string;
  image: string;
  gender: 'male' | 'female';
}

// 診断結果型定義
export interface DiagnosisResult {
  country: string;
  score: number;
  confidence: number;
  gender: 'male' | 'female';
  timestamp: string;
}

// API応答型定義
export interface ApiResponse<T> {
  success: boolean;
  data?: T;
  error?: string;
  message?: string;
}

// エラー型定義
export interface AppError {
  code: string;
  message: string;
  details?: string;
  timestamp: string;
}