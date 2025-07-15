import React from 'react';
import { MapPin } from 'lucide-react';

interface CountryMapProps {
  countryName: string;
  className?: string;
}

const CountryMap: React.FC<CountryMapProps> = ({ countryName, className = '' }) => {
  return (
    <div className={`bg-white rounded-lg overflow-hidden shadow-lg ${className}`}>
      <div className="bg-gradient-to-r from-blue-500 to-blue-600 text-white p-3">
        <div className="flex items-center gap-2">
          <MapPin className="w-5 h-5" />
          <span className="font-semibold">{countryName}の位置</span>
        </div>
      </div>
      
      {/* 簡易的な地図表示エリア */}
      <div className="h-48 bg-gradient-to-br from-blue-50 to-green-50 flex items-center justify-center relative">
        <div className="text-center">
          <div className="text-4xl mb-2">🗺️</div>
          <p className="text-sm text-gray-600 font-medium">{countryName}</p>
          <p className="text-xs text-gray-500">地図表示エリア</p>
        </div>
        
        {/* 装飾的な要素 */}
        <div className="absolute top-4 left-4 w-2 h-2 bg-red-500 rounded-full animate-pulse"></div>
        <div className="absolute bottom-4 right-4 w-1 h-1 bg-blue-500 rounded-full"></div>
        <div className="absolute top-8 right-6 w-1 h-1 bg-green-500 rounded-full"></div>
      </div>
      
      <div className="p-3 bg-gray-50">
        <p className="text-xs text-gray-500 text-center">
          📍 {countryName}の地理的位置
        </p>
      </div>
    </div>
  );
};

export default CountryMap;