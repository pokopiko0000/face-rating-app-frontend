'use client';

import { useState } from 'react';
import ImageUploader from '@/components/ImageUploader';
import GenderSelector from '@/components/GenderSelector';
import ResultsDisplay from '@/components/ResultsDisplay';

interface AnalysisResult {
  country: string;
  similarity: number;
  confidence: number;
  features: string[];
}

export default function Home() {
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [selectedGender, setSelectedGender] = useState<'male' | 'female'>('male');
  const [results, setResults] = useState<AnalysisResult[] | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleImageUpload = (file: File) => {
    setSelectedImage(file);
    setResults(null);
    setError(null);
  };

  const handleAnalyze = async () => {
    if (!selectedImage) return;

    setIsLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', selectedImage);
    formData.append('gender', selectedGender);

    try {
      const response = await fetch('http://127.0.0.1:8003/analyze', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('分析に失敗しました');
      }

      const data = await response.json();
      
      // APIレスポンスを新しい形式に変換
      const transformedResults: AnalysisResult[] = data.ranking.map((item: any) => ({
        country: item.country,
        similarity: Math.round(item.score * 100),
        confidence: Math.round(Math.random() * 20 + 80), // 仮の信頼度
        features: ['顔の輪郭', '目の形', '鼻の高さ'] // 仮の特徴
      }));

      setResults(transformedResults);
    } catch (err) {
      setError(err instanceof Error ? err.message : '分析中にエラーが発生しました');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-950">
      {/* ヘッダー */}
      <div className="relative">
        <div className="max-w-3xl mx-auto px-6 py-20 text-center">
          <h1 className="text-6xl md:text-7xl font-bold text-white mb-8 tracking-tight">
            どの国の理想顔？
          </h1>
          <p className="text-xl text-slate-400 mb-16 max-w-2xl mx-auto leading-relaxed">
            AIがあなたの顔写真を分析し、どの国の「理想的な顔」に最も近いかを判定します。
          </p>
        </div>
      </div>

      {/* メインコンテンツ */}
      <div className="max-w-2xl mx-auto px-6 pb-20">
        <div className="bg-slate-900/50 backdrop-blur-xl rounded-3xl border border-slate-800 p-10 shadow-2xl">
          
          {/* 画像アップロード */}
          <div className="mb-10">
            <ImageUploader onImageUpload={handleImageUpload} />
          </div>

          {/* 性別選択 */}
          {selectedImage && (
            <div className="mb-10">
              <GenderSelector 
                selectedGender={selectedGender} 
                onGenderChange={setSelectedGender} 
              />
            </div>
          )}

          {/* 分析ボタン */}
          {selectedImage && (
            <div className="mb-10">
              <button
                onClick={handleAnalyze}
                disabled={isLoading}
                className="w-full bg-white hover:bg-slate-100 disabled:bg-slate-700 text-slate-900 disabled:text-slate-400 font-semibold py-4 px-8 rounded-2xl transition-all duration-200 transform hover:scale-[1.02] disabled:scale-100 disabled:cursor-not-allowed shadow-lg"
              >
                {isLoading ? (
                  <div className="flex items-center justify-center space-x-3">
                    <div className="w-5 h-5 border-2 border-slate-400 border-t-slate-900 rounded-full animate-spin"></div>
                    <span>分析中...</span>
                  </div>
                ) : (
                  '顔を分析する'
                )}
              </button>
            </div>
          )}

          {/* エラー表示 */}
          {error && (
            <div className="mb-10 p-6 bg-red-950/50 border border-red-800/50 rounded-2xl">
              <p className="text-red-300 text-center font-medium">{error}</p>
            </div>
          )}

          {/* 結果表示 */}
          {results && (
            <div className="space-y-8">
              <h2 className="text-3xl font-bold text-white text-center mb-8">
                分析結果
              </h2>
              <ResultsDisplay results={results} />
            </div>
          )}

          {/* 初期状態のメッセージ */}
          {!selectedImage && !results && (
            <div className="text-center py-16">
              <div className="text-7xl mb-6">📸</div>
              <p className="text-slate-400 text-xl font-medium">
                顔写真をアップロードして分析を開始しましょう
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
