import React from 'react';
import { MapPin } from 'lucide-react';

interface WorldMapProps {
  countryName: string;
  coordinates: { lat: number; lng: number };
}

export default function WorldMap({ countryName, coordinates }: WorldMapProps) {
  // 緯度経度を地図上のパーセンテージ位置に変換
  const calculatePosition = (lat: number, lng: number) => {
    // 簡易的なメルカトル図法変換
    const x = (lng + 180) / 360 * 100;
    const y = (90 - lat) / 180 * 100;
    return { x, y };
  };

  const position = calculatePosition(coordinates.lat, coordinates.lng);

  return (
    <div className="relative w-full h-64 bg-white/5 backdrop-blur-sm rounded-2xl overflow-hidden border border-white/20">
      {/* 世界地図の背景 */}
      <div className="absolute inset-0 opacity-30">
        <svg viewBox="0 0 1000 500" className="w-full h-full">
          {/* 簡易的な世界地図の輪郭 */}
          <path
            d="M 200,250 Q 300,200 400,250 T 600,250 Q 700,200 800,250"
            stroke="currentColor"
            strokeWidth="1"
            fill="none"
            className="text-white/50"
          />
          {/* 大陸の輪郭を簡易的に表現 */}
          <circle cx="500" cy="250" r="200" fill="none" stroke="currentColor" strokeWidth="0.5" className="text-white/30" />
        </svg>
      </div>
      
      {/* 実際の世界地図画像 */}
      <img
        src="https://upload.wikimedia.org/wikipedia/commons/thumb/e/e3/World_map_blank_without_borders.svg/2560px-World_map_blank_without_borders.svg.png"
        alt="World Map"
        className="absolute inset-0 w-full h-full object-cover opacity-20"
      />
      
      {/* ピンマーカー */}
      <div
        className="absolute transform -translate-x-1/2 -translate-y-full animate-bounce"
        style={{ 
          left: `${position.x}%`, 
          top: `${position.y}%`,
          animation: 'float 3s ease-in-out infinite'
        }}
      >
        <div className="relative">
          <MapPin className="w-8 h-8 text-red-500 drop-shadow-lg" fill="currentColor" />
          <div className="absolute -top-1 left-1/2 transform -translate-x-1/2">
            <div className="w-3 h-3 bg-red-500 rounded-full animate-ping" />
          </div>
        </div>
      </div>
      
      {/* 国名ラベル */}
      <div className="absolute bottom-4 left-4 right-4">
        <p className="text-white/90 text-sm font-medium">
          📍 {countryName}の位置
        </p>
      </div>
      
      <style jsx>{`
        @keyframes float {
          0%, 100% {
            transform: translateX(-50%) translateY(-100%) translateY(0px);
          }
          50% {
            transform: translateX(-50%) translateY(-100%) translateY(-10px);
          }
        }
      `}</style>
    </div>
  );
}