import React, { useState, useEffect } from 'react';
import { Share2, RotateCcw, Trophy, Globe } from 'lucide-react';
import { DiagnosisResult } from '../types';

interface ResultDisplayProps {
  result: DiagnosisResult;
  userImage: string;
  onReset: () => void;
  gender: 'male' | 'female';
}

// 国コードを国旗絵文字に変換するヘルパー関数
const getFlagEmoji = (countryCode: string | null): string => {
  if (!countryCode) return '🏳️';
  // countryCode は 'JP' のような2文字のコードを想定
  const codePoints = countryCode
    .toUpperCase()
    .split('')
    .map(char => 127397 + char.charCodeAt(0));
  return String.fromCodePoint(...codePoints);
};

export default function ResultDisplay({ result, userImage, onReset, gender }: ResultDisplayProps) {
  // デバッグ用に結果をコンソールに出力
  useEffect(() => {
    console.log('Diagnosis Result:', result);
  }, [result]);

  const [topCountryImage, setTopCountryImage] = useState<string>('');
  const topResult = result.ranking[0];
  const otherResults = result.ranking.slice(1, 5); // 2位から5位まで

  useEffect(() => {
    const fetchTopCountryImage = async () => {
      if (!topResult) return;
      try {
        const response = await fetch(`http://localhost:8003/comparison?country=${encodeURIComponent(topResult.country)}&gender=${gender}`);
        if (response.ok) {
          const imageBlob = await response.blob();
          setTopCountryImage(URL.createObjectURL(imageBlob));
        }
      } catch (error) {
        console.error('Error fetching top country image:', error);
      }
    };
    fetchTopCountryImage();

    // Clean up the object URL on component unmount
    return () => {
      if (topCountryImage) {
        URL.revokeObjectURL(topCountryImage);
      }
    };
  }, [result, gender]);


  const handleShare = () => {
    if (!topResult) return;
    const text = `AI顔診断の結果、私は${topResult.country}の顔と${Math.round(topResult.similarity * 100)}%似ているそうです！ あなたも試してみよう！ #AI顔診断 #似ている国の顔`;
    const url = window.location.href;
    const twitterUrl = `https://twitter.com/intent/tweet?text=${encodeURIComponent(text)}&url=${encodeURIComponent(url)}`;
    window.open(twitterUrl, '_blank');
  };

  if (!topResult) {
    return (
      <div className="text-center">
        <p>診断結果がありません。</p>
        <button onClick={onReset}>もう一度試す</button>
      </div>
    );
  }

  return (
    <div className="w-full max-w-2xl mx-auto">
      <div className="text-center mb-8">
        <div className="inline-flex items-center gap-2 px-4 py-2 bg-gradient-to-r from-yellow-400 to-orange-500 text-white rounded-full font-medium mb-4">
          <Trophy size={18} />
          診断結果
        </div>
        <h2 className="text-2xl font-bold text-gray-800 mb-2">
          最も似ているのは...
        </h2>
      </div>

      <div className="bg-white rounded-3xl shadow-2xl overflow-hidden mb-8">
        <div className="bg-gradient-to-r from-purple-500 to-pink-500 text-white p-6 text-center">
          <div className="text-5xl font-bold mb-2">{getFlagEmoji(topResult.country_code)} {topResult.country}</div>
          <div className="text-3xl font-bold mb-2">{Math.round(topResult.similarity * 100)}%</div>
          <div className="text-lg opacity-90">類似度</div>
        </div>

        <div className="p-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
            <div className="text-center">
              <div className="relative">
                <img
                  src={userImage}
                  alt="あなたの写真"
                  className="w-48 h-48 mx-auto rounded-2xl object-cover shadow-lg"
                />
                <div className="absolute -bottom-3 left-1/2 transform -translate-x-1/2 bg-blue-500 text-white px-4 py-1 rounded-full text-sm font-medium">
                  あなた
                </div>
              </div>
            </div>

            <div className="text-center">
              <div className="relative">
                 {topCountryImage ? (
                  <img
                    src={topCountryImage}
                    alt={`${topResult.country}の代表画像`}
                    className="w-48 h-48 mx-auto rounded-2xl object-cover shadow-lg bg-gray-200"
                  />
                ) : (
                  <div className="w-48 h-48 mx-auto rounded-2xl bg-gray-200 flex items-center justify-center">
                    <p className="text-sm text-gray-500">画像読込中...</p>
                  </div>
                )}
                <div className="absolute -bottom-3 left-1/2 transform -translate-x-1/2 bg-purple-500 text-white px-4 py-1 rounded-full text-sm font-medium">
                  {topResult.country}
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {otherResults.length > 0 && (
        <div className="bg-white rounded-3xl shadow-xl p-6 mb-8">
          <h3 className="text-xl font-bold text-center text-gray-700 mb-4">トップ5ランキング</h3>
          <ul className="space-y-3">
            {otherResults.map((item, index) => (
              <li key={item.country} className="flex items-center p-3 bg-gray-50 rounded-lg">
                <span className="text-lg font-bold text-gray-600 w-8">{index + 2}</span>
                <span className="text-2xl mr-3">{getFlagEmoji(item.country_code)}</span>
                <span className="text-lg text-gray-800 font-medium flex-1">{item.country}</span>
                <span className="text-lg font-bold text-purple-600">{Math.round(item.similarity * 100)}%</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      <div className="flex flex-col sm:flex-row gap-4 justify-center">
        <button
          onClick={handleShare}
          className="flex items-center justify-center gap-2 px-8 py-4 bg-gradient-to-r from-blue-500 to-blue-600 text-white rounded-2xl font-medium hover:from-blue-600 hover:to-blue-700 transition-all duration-300 shadow-lg hover:shadow-xl transform hover:scale-105"
        >
          <Share2 size={20} />
          結果をシェア
        </button>
        
        <button
          onClick={onReset}
          className="flex items-center justify-center gap-2 px-8 py-4 bg-gray-100 text-gray-700 rounded-2xl font-medium hover:bg-gray-200 transition-all duration-300 shadow-lg hover:shadow-xl transform hover:scale-105"
        >
          <RotateCcw size={20} />
          もう一度診断
        </button>
      </div>
    </div>
  );
}