import React, { useState } from 'react';
import { useParams, Link, useNavigate } from 'react-router-dom';
import { ArrowLeft, Sparkles, MapPin, Users, Globe, Home } from 'lucide-react';
import CountryFlag from './CountryFlag';
import InteractiveWorldMap from './InteractiveWorldMap';
import { getCountryImage } from '../data/countryImages';
import { countryData } from '../data/countries';

export default function CountryDetailSimple() {
  const { countryCode } = useParams<{ countryCode: string }>();
  const navigate = useNavigate();
  const country = countryData[countryCode?.toLowerCase() || ''];
  const [imageLoaded, setImageLoaded] = useState(false);
  
  const handleBack = () => {
    // 診断結果から来た場合は戻る、そうでなければホームページに移動
    if (window.history.length > 1) {
      navigate(-1);
    } else {
      navigate('/');
    }
  };

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
        <div className="p-6 flex items-center justify-between">
          <button
            onClick={handleBack}
            className="inline-flex items-center gap-2 px-4 py-2 bg-white/20 backdrop-blur-md rounded-full shadow-lg hover:shadow-xl hover:bg-white/30 transition-all duration-300 text-white border border-white/30"
          >
            <ArrowLeft className="w-4 h-4" />
            戻る
          </button>
          
          <Link
            to="/"
            className="inline-flex items-center gap-2 px-4 py-2 bg-white/20 backdrop-blur-md rounded-full shadow-lg hover:shadow-xl hover:bg-white/30 transition-all duration-300 text-white border border-white/30"
          >
            <Home className="w-4 h-4" />
            ホーム
          </Link>
        </div>

        {/* Hero Section */}
        <div className="flex-1 flex items-center justify-center px-6 py-12">
          <div className="max-w-4xl w-full">
            {/* Country Header */}
            <div className="text-center mb-12">
              <div className="flex justify-center items-center gap-4 mb-4">
                <CountryFlag 
                  countryCode={country.code}
                  countryName={country.name}
                  size="large"
                />
              </div>
              <h1 className="text-6xl md:text-7xl font-bold mb-3 text-white drop-shadow-lg">
                {country.name}
              </h1>
              <p className="text-2xl md:text-3xl text-white/90">
                {country.nameEn}
              </p>
            </div>

            {/* Content Cards */}
            <div className="space-y-6">
              {/* Map and Basic Info Section */}
              <div className="bg-white/10 backdrop-blur-md rounded-3xl p-8 border border-white/20 shadow-2xl">
                <div className="grid md:grid-cols-2 gap-8">
                  {/* Left: World Map */}
                  <div>
                    <h3 className="text-xl font-bold text-white mb-4 flex items-center gap-2">
                      <Globe className="w-5 h-5 text-blue-300" />
                      世界地図での位置
                    </h3>
                    <InteractiveWorldMap 
                      countryName={country.name}
                      coordinates={country.coordinates}
                    />
                  </div>
                  
                  {/* Right: Basic Info */}
                  <div>
                    <h3 className="text-xl font-bold text-white mb-6">基本情報</h3>
                    <div className="space-y-4">
                      <div className="flex items-center gap-3 bg-white/10 rounded-lg p-4">
                        <MapPin className="w-5 h-5 text-purple-300 flex-shrink-0" />
                        <div>
                          <div className="text-sm text-white/70">首都</div>
                          <div className="font-semibold text-white text-lg">{country.basic.capital}</div>
                        </div>
                      </div>
                      <div className="flex items-center gap-3 bg-white/10 rounded-lg p-4">
                        <Users className="w-5 h-5 text-purple-300 flex-shrink-0" />
                        <div>
                          <div className="text-sm text-white/70">人口</div>
                          <div className="font-semibold text-white text-lg">{country.basic.population}</div>
                        </div>
                      </div>
                      <div className="flex items-center gap-3 bg-white/10 rounded-lg p-4">
                        <Globe className="w-5 h-5 text-purple-300 flex-shrink-0" />
                        <div>
                          <div className="text-sm text-white/70">言語</div>
                          <div className="font-semibold text-white text-lg">{country.basic.language}</div>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
                
                {/* Description */}
                <div className="mt-8 pt-8 border-t border-white/10">
                  <p className="text-lg text-white/90 leading-relaxed">
                    {country.description}
                  </p>
                </div>
              </div>


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