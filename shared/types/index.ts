// 共有型定義 - フロントエンドとバックエンドの統一型定義

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

// APIレスポンス全体の型
export interface DiagnosisApiResponse {
  ranking: CountryRanking[];
  top_country_image_url: string | null;
}

// useDiagnosisフックが最終的にコンポーネントに渡す結果の型
export type DiagnosisResult = DiagnosisApiResponse;

// Celebrity型定義
export interface Celebrity {
  id: string;
  name: string;
  nationality: string;
  image: string;
  gender: 'male' | 'female';
}

// 再エクスポート：country.tsの型定義
export * from './country';