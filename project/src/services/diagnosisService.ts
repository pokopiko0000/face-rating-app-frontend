import { DiagnosisRequest, DiagnosisResult } from '../types';

// 実AI診断サービス
export const diagnoseFace = async (request: DiagnosisRequest): Promise<DiagnosisResult> => {
  const { image, gender } = request;

  const formData = new FormData();
  formData.append('file', image);
  formData.append('gender', gender);

  try {
    const response = await fetch('http://localhost:8003/analyze', {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ detail: 'サーバーから不明なエラーが返されました。' }));
      throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
    }

    const result: DiagnosisResult = await response.json();
    return result;
  } catch (error) {
    console.error("Diagnosis service error:", error);
    // エラーオブジェクトがErrorインスタンスか確認
    const errorMessage = error instanceof Error ? error.message : '診断の実行中に不明なネットワークエラーが発生しました。';
    // ユーザーフレンドリーなメッセージをスローする
    throw new Error(errorMessage);
  }
};