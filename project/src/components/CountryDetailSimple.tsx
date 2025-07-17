import React, { useState } from 'react';
import { useParams, Link } from 'react-router-dom';
import { Sparkles } from 'lucide-react';
import CountryDetailNavigation from './CountryDetailNavigation';
import CountryDetailHeader from './CountryDetailHeader';
import CountryBasicInfo from './CountryBasicInfo';
import CountryHighlights from './CountryHighlights';
import { getCountryImage } from '../data/countryImages';
import { countryData } from '../data/countries';

export default function CountryDetailSimple() {
  const { countryCode } = useParams<{ countryCode: string }>();
  const country = countryData[countryCode?.toLowerCase() || ''];
  const [imageLoaded, setImageLoaded] = useState(false);

  if (!country) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-purple-50 via-pink-50 to-blue-50 py-8 px-4">
        <div className="max-w-4xl mx-auto text-center">
          <h1 className="text-2xl font-bold text-gray-800 mb-4">国の情報が見つかりません</h1>
          <Link to="/" className="text-purple-600 hover:text-purple-800">
            トップページに戻る
          </Link>
        </div>
      </div>
    );
  }

  const imageUrl = getCountryImage(countryCode || '');

  return (
    <div className="relative min-h-screen overflow-hidden">
      {/* Full Screen Background Image */}
      <div className="fixed inset-0 z-0">
        {/* Loading placeholder */}
        {!imageLoaded && (
          <div className="absolute inset-0 bg-gradient-to-br from-purple-200 to-pink-200 animate-pulse" />
        )}
        
        {/* Background image */}
        <img
          src={imageUrl}
          alt={`${country.name}の美しい風景`}
          className={`w-full h-full object-cover transition-opacity duration-700 ${imageLoaded ? 'opacity-100' : 'opacity-0'}`}
          crossOrigin="anonymous"
          onLoad={() => setImageLoaded(true)}
          onError={(e) => {
            const target = e.target as HTMLImageElement;
            if (!target.src.includes('photo-1469474968028')) {
              target.src = 'https://images.unsplash.com/photo-1469474968028-56623f02e42e?w=1200&h=800&fit=crop&q=80';
            }
            setImageLoaded(true);
          }}
        />
        
        {/* Overlay gradient for better readability */}
        <div className="absolute inset-0 bg-gradient-to-b from-black/40 via-black/50 to-black/60" />
      </div>

      {/* Content Container */}
      <div className="relative z-10 min-h-screen flex flex-col">
        {/* Navigation */}
        <CountryDetailNavigation />

        {/* Hero Section */}
        <div className="flex-1 flex items-center justify-center px-6 py-12">
          <div className="max-w-4xl w-full">
            {/* Country Header */}
            <CountryDetailHeader country={country} />

            {/* Content Cards */}
            <div className="space-y-6">
              {/* Map and Basic Info Section */}
              <CountryBasicInfo country={country} />

              {/* Highlights */}
              <CountryHighlights country={country} countryCode={countryCode || ''} />

              {/* Call to Action */}
              <div className="text-center">
                <Link
                  to="/"
                  className="inline-flex items-center gap-3 px-12 py-4 bg-gradient-to-r from-purple-500 to-pink-500 text-white rounded-full font-bold text-lg hover:from-purple-600 hover:to-pink-600 transition-all duration-300 shadow-2xl hover:shadow-3xl transform hover:scale-105 backdrop-blur-sm border border-white/30"
                >
                  <Sparkles className="w-6 h-6" />
                  自分も診断してみる
                </Link>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}