'use client';

import Image from 'next/image';

interface RankingItem {
  rank: number;
  country: string;
  score: number;
  country_code: string | null;
  representative_image_filename?: string;
}

interface ApiResponse {
  detected_gender: 'man' | 'woman';
  ranking: RankingItem[];
}

interface ResultsDisplayProps {
  results: ApiResponse;
  selectedImage: File | null;
}

export default function ResultsDisplay({ results, selectedImage }: ResultsDisplayProps) {
  const topResult = results.ranking[0];

  return (
    <div className="space-y-8">
      <h2 className="text-3xl font-bold text-slate-900 text-center mb-2">
        分析結果
      </h2>
      <p className="text-center text-slate-600 -mt-6">
        あなたの顔は<span className="font-bold text-teal-600 text-lg">{topResult.country}</span>の理想顔に最も近いようです！
      </p>

      {/* 1位の比較表示 */}
      {topResult.representative_image_filename && (
          <div className="flex flex-col sm:flex-row gap-4 items-stretch">
              <div className="w-full sm:w-1/2 bg-slate-100 rounded-lg p-3 border border-slate-200">
                  <p className="text-center font-bold text-slate-700 mb-2">1位の理想顔</p>
                  <div className="aspect-w-1 aspect-h-1">
                      <img
                          src={`http://localhost:8003/images/${topResult.representative_image_filename}`}
                          alt={`${topResult.country}の代表的な顔`}
                          className="object-cover w-full h-full rounded-md"
                      />
                  </div>
              </div>
              <div className="w-full sm:w-1/2 bg-slate-100 rounded-lg p-3 border border-slate-200">
                  <p className="text-center font-bold text-slate-700 mb-2">あなたの顔</p>
                   <div className="aspect-w-1 aspect-h-1">
                      {selectedImage && (
                          <img
                              src={URL.createObjectURL(selectedImage)}
                              alt="あなたの顔"
                              className="object-cover w-full h-full rounded-md"
                          />
                      )}
                  </div>
              </div>
          </div>
      )}

      {/* 2位以下のランキング */}
      <div className="space-y-3">
        {results.ranking.map((item) => (
          <div
            key={item.rank}
            className={`p-4 rounded-xl border transition-all duration-300 ${
              item.rank === 1
                ? 'bg-teal-500/10 border-teal-500/30 shadow-lg'
                : 'bg-white/80 border-slate-200'
            }`}
          >
            <div className="flex items-center space-x-4">
              <div className="text-2xl font-bold text-slate-400 w-8 text-center">{item.rank}</div>
              <div className="flex-shrink-0">
                {item.country_code ? (
                  <Image
                    src={`https://flagcdn.com/w40/${item.country_code.toLowerCase()}.png`}
                    alt={`${item.country}の国旗`}
                    width={40}
                    height={30}
                    className="rounded-md"
                  />
                ) : (
                  <div className="w-10 h-[30px] bg-slate-200 rounded-md"></div>
                )}
              </div>
              <div className="flex-1">
                <p className={`font-bold ${item.rank === 1 ? 'text-teal-800' : 'text-slate-800'}`}>
                  {item.country}
                </p>
              </div>
              <div className="text-right">
                <p className={`font-bold text-lg ${item.rank === 1 ? 'text-teal-600' : 'text-slate-700'}`}>
                  {Math.round(item.score * 100)}%
                </p>
                <p className="text-xs text-slate-500 -mt-1">類似度</p>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

 