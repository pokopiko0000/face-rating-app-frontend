import { useState, useCallback } from 'react';
import { DiagnosisResponse, DiagnosisRequest } from '../types';
import { diagnoseFace } from '../services/diagnosisService';

export const useDiagnosis = () => {
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<DiagnosisResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const diagnose = useCallback(async (request: DiagnosisRequest) => {
    setIsLoading(true);
    setError(null);
    setResult(null);

    try {
      const diagnosisResult = await diagnoseFace(request);
      setResult(diagnosisResult);
    } catch {
      setError('診断中にエラーが発生しました。もう一度お試しください。');
    } finally {
      setIsLoading(false);
    }
  }, []);

  const reset = useCallback(() => {
    setResult(null);
    setError(null);
    setIsLoading(false);
  }, []);

  return {
    isLoading,
    result,
    error,
    diagnose,
    reset
  };
};