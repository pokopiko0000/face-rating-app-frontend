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

// 診断結果型定義は project/src/types/index.ts で定義されています（現在のAPIレスポンス構造に合わせて）

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