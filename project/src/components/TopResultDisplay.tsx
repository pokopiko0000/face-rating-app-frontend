import React from 'react';
import { Link } from 'react-router-dom';
import { ExternalLink } from 'lucide-react';
import { CountryRanking } from '../types';
import { getCountryCodeFromDiagnosis } from '../utils/countryCodeMapping';

interface TopResultDisplayProps {
  topResult: CountryRanking;
  userImage: string;
  topCountryImage: string | null;
}

export default function TopResultDisplay({ topResult, userImage, topCountryImage }: TopResultDisplayProps) {
  const countryCode = getCountryCodeFromDiagnosis(topResult.country, topResult.country_code);

  return (
    <div className="bg-white rounded-3xl shadow-2xl overflow-hidden mb-8">
      <div className="bg-gradient-to-r from-purple-500 to-pink-500 text-white p-6 text-center">
        <div className="text-4xl font-bold mb-2 flex items-center justify-center gap-4">
          {topResult.country_code && (
            <img
              src={`https://flagcdn.com/w40/${topResult.country_code.toLowerCase()}.png`}
              alt={`${topResult.country}の国旗`}
              className="h-8 w-auto rounded"
            />
          )}
          <span>{topResult.country}</span>
        </div>
        <div className="text-3xl font-bold mb-2">{Math.round(topResult.similarity)}点</div>
        <div className="text-lg opacity-90">顔面相性スコア</div>
        
        {/* 国詳細ページへのリンク */}
        {countryCode && (
          <div className="mt-4">
            <Link
              to={`/country/${countryCode}`}
              className="inline-flex items-center gap-2 px-6 py-3 bg-white/20 hover:bg-white/30 text-white rounded-full font-medium transition-all duration-300 hover:scale-105"
            >
              <ExternalLink size={18} />
              {topResult.country}について詳しく見る
            </Link>
          </div>
        )}
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
                  <p className="text-sm text-gray-500">画像なし</p>
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
  );
}