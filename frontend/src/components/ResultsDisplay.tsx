'use client';

interface AnalysisResult {
  country: string;
  similarity: number;
  confidence: number;
  features: string[];
}

interface ResultsDisplayProps {
  results: AnalysisResult[];
}

export default function ResultsDisplay({ results }: ResultsDisplayProps) {
  const getCountryFlag = (country: string) => {
    const flags: { [key: string]: string } = {
      '日本': '🇯🇵',
      '韓国': '🇰🇷',
      '中国': '🇨🇳',
      'アメリカ': '🇺🇸',
      'イギリス': '🇬🇧',
      'フランス': '🇫🇷',
      'ドイツ': '🇩🇪',
      'イタリア': '🇮🇹',
      'スペイン': '🇪🇸',
      'ブラジル': '🇧🇷',
      'インド': '🇮🇳',
      'ロシア': '🇷🇺',
      'オーストラリア': '🇦🇺',
      'カナダ': '🇨🇦',
      'メキシコ': '🇲🇽',
    };
    return flags[country] || '🌍';
  };

  const getSimilarityColor = (similarity: number) => {
    if (similarity >= 80) return 'text-green-400 bg-green-500/10 border-green-500/20';
    if (similarity >= 60) return 'text-yellow-400 bg-yellow-500/10 border-yellow-500/20';
    return 'text-red-400 bg-red-500/10 border-red-500/20';
  };

  const getProgressColor = (similarity: number) => {
    if (similarity >= 80) return 'bg-green-500';
    if (similarity >= 60) return 'bg-yellow-500';
    return 'bg-red-500';
  };

  const handleShare = () => {
    if (results.length === 0) return;
    const topResult = results[0];
    const shareText = `AI顔診断の結果、私の顔は「${topResult.country}」の理想顔と${topResult.similarity}%似ていました！✨\nあなたも試してみては？\n\n#理想顔診断 #AI顔診断`;
    const appUrl = 'http://localhost:3000';
    const twitterUrl = `https://twitter.com/intent/tweet?text=${encodeURIComponent(shareText)}&url=${encodeURIComponent(appUrl)}`;
    window.open(twitterUrl, '_blank');
  };

  return (
    <div className="space-y-8">
      {/* トップ結果 */}
      {results.length > 0 && (
        <div className="bg-slate-800/30 border border-slate-700/50 rounded-3xl p-8">
          <div className="text-center mb-6">
            <div className="text-5xl mb-4">{getCountryFlag(results[0].country)}</div>
            <h3 className="text-3xl font-bold text-white mb-2">
              {results[0].country}
            </h3>
            <p className="text-slate-400 text-base">最も似ている国</p>
          </div>
          
          <div className="flex items-center justify-center space-x-8 mb-8">
            <div className="text-center">
              <div className="text-4xl font-bold text-white">
                {results[0].similarity}%
              </div>
              <div className="text-sm text-slate-400 mt-1">類似度</div>
            </div>
            <div className="text-center">
              <div className="text-4xl font-bold text-white">
                {results[0].confidence}%
              </div>
              <div className="text-sm text-slate-400 mt-1">信頼度</div>
            </div>
          </div>

          {results[0].features && results[0].features.length > 0 && (
            <div className="mb-8">
              <p className="text-base text-slate-300 mb-4 text-center font-medium">特徴的な要素:</p>
              <div className="flex flex-wrap justify-center gap-3">
                {results[0].features.map((feature, index) => (
                  <span
                    key={index}
                    className="px-4 py-2 bg-slate-700/50 text-slate-300 rounded-xl text-sm border border-slate-600/50 font-medium"
                  >
                    {feature}
                  </span>
                ))}
              </div>
            </div>
          )}

          <button
            onClick={handleShare}
            className="w-full bg-white hover:bg-slate-100 text-slate-900 font-semibold py-4 px-6 rounded-2xl transition-all duration-200 text-base"
          >
            結果をシェア
          </button>
        </div>
      )}

      {/* その他の結果 */}
      {results.slice(1).map((result, index) => (
        <div
          key={index}
          className="bg-slate-800/20 border border-slate-700/30 rounded-2xl p-6 hover:bg-slate-800/40 transition-all duration-200"
        >
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center space-x-4">
              <div className="text-3xl">{getCountryFlag(result.country)}</div>
              <div>
                <h4 className="font-semibold text-slate-200 text-lg">{result.country}</h4>
                <div className="flex items-center space-x-4 mt-2">
                  <span className={`px-3 py-1 rounded-xl text-sm font-semibold border ${getSimilarityColor(result.similarity)}`}>
                    類似度 {result.similarity}%
                  </span>
                  <span className="text-sm text-slate-400 font-medium">
                    信頼度 {result.confidence}%
                  </span>
                </div>
              </div>
            </div>
            <div className="text-right">
              <div className="text-xl font-bold text-slate-300">#{index + 2}</div>
            </div>
          </div>

          {result.features && result.features.length > 0 && (
            <div className="mb-4 pt-4 border-t border-slate-700/50">
              <p className="text-sm text-slate-400 mb-3 font-medium">特徴的な要素:</p>
              <div className="flex flex-wrap gap-2">
                {result.features.map((feature, featureIndex) => (
                  <span
                    key={featureIndex}
                    className="px-3 py-1 bg-slate-700/30 text-slate-400 rounded-lg text-sm"
                  >
                    {feature}
                  </span>
                ))}
              </div>
            </div>
          )}

          {/* プログレスバー */}
          <div className="w-full bg-slate-700/30 rounded-full h-3">
            <div
              className={`h-3 rounded-full transition-all duration-500 ${getProgressColor(result.similarity)}`}
              style={{ width: `${result.similarity}%` }}
            ></div>
          </div>
        </div>
      ))}

      {results.length === 0 && (
        <div className="text-center py-12">
          <div className="text-5xl mb-6">🤔</div>
          <p className="text-slate-400 text-lg">分析結果がありません</p>
        </div>
      )}
    </div>
  );
}

 