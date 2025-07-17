import { DiagnosisRequest, DiagnosisResponse } from '../types';

// 実AI診断サービス
export const diagnoseFace = async (request: DiagnosisRequest): Promise<DiagnosisResponse> => {
  const { image, gender } = request;

  const formData = new FormData();
  formData.append('file', image);
  formData.append('gender', gender);

  // 環境変数からAPIのベースURLを取得。なければローカルのURLをデフォルトにする。
  const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8003';

  try {
    const response = await fetch(`${API_BASE_URL}/analyze`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ detail: 'サーバーから不明なエラーが返されました。' }));
      throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
    }

    const result: DiagnosisResponse = await response.json();
    return result;
  } catch (error) {
    console.error("Diagnosis service error:", error);
    // エラーオブジェクトがErrorインスタンスか確認
    const errorMessage = error instanceof Error ? error.message : '診断の実行中に不明なネットワークエラーが発生しました。';
    // ユーザーフレンドリーなメッセージをスローする
    throw new Error(errorMessage);
  }
};