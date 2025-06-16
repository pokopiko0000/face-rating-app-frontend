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
}

// useDiagnosisフックが最終的にコンポーネントに渡す結果の型
export type DiagnosisResult = DiagnosisApiResponse;