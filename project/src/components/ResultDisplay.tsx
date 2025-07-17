import React from 'react';
import { Share2, RotateCcw, Trophy } from 'lucide-react';
import { DiagnosisResult } from '../types';
import AdBanner from './AdBanner';
import { ADS_CONFIG } from '../config/ads';
import TopResultDisplay from './TopResultDisplay';
import CountryRankingItem from './CountryRankingItem';

interface ResultDisplayProps {
  result: DiagnosisResult;
  userImage: string;
  onReset: () => void;
  gender: 'male' | 'female';
}

export default function ResultDisplay({ result, userImage, onReset }: ResultDisplayProps) {
  const topResult = result.ranking[0];
  const otherResults = result.ranking.slice(1, 5); // 2位から5位まで
  const topCountryImage = result.top_country_image_url; // バックエンドから直接URLを取得

  const handleShare = () => {
    if (!topResult) {
      return;
    }
    const text = `AI顔診断の結果、私と${topResult.country}の顔の相性は${Math.round(topResult.similarity)}点でした！ あなたも試してみよう！ #AI顔診断 #顔面相性スコア`;
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
          あなたが最も輝く国は...
        </h2>
      </div>

      <TopResultDisplay 
        topResult={topResult}
        userImage={userImage}
        topCountryImage={topCountryImage}
      />

      {otherResults.length > 0 && (
        <div className="bg-white rounded-3xl shadow-xl p-6 mb-8">
          <h3 className="text-xl font-bold text-center text-gray-700 mb-4">トップ5ランキング</h3>
          <ul className="space-y-3">
            {otherResults.map((item, index) => (
              <CountryRankingItem 
                key={item.country} 
                item={item} 
                rank={index + 2} 
              />
            ))}
          </ul>
        </div>
      )}

      {/* Advertisement */}
      <div className="bg-white rounded-3xl shadow-xl p-6 mb-8 text-center">
        <p className="text-sm text-gray-500 mb-4">おすすめ</p>
        <AdBanner 
          adSlot={ADS_CONFIG.SLOTS.RESULT}
          adFormat="rectangle"
          className="mx-auto"
          style={{ minHeight: '250px', maxWidth: '336px' }}
        />
      </div>

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