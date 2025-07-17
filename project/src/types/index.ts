// 統一型定義の再エクスポート - 後方互換性のため
export * from '../../../shared/types';

// 診断リクエストの型
export interface DiagnosisRequest {
  image: File;
  gender: 'male' | 'female';
}

// バックエンドのレスポンスの ranking 配列の要素の型
export interface CountryRanking {
  country: string;
  similarity: number;
  country_code: string | null;
}

// 診断結果の型 - バックエンドAPIレスポンスに対応
export interface DiagnosisResult {
  ranking: CountryRanking[];
  top_country_image_url: string | null;
}