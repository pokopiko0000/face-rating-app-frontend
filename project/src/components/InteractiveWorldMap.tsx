import React from 'react';
import { MapPin } from 'lucide-react';

interface InteractiveWorldMapProps {
  countryName: string;
  coordinates: { lat: number; lng: number };
}

export default function InteractiveWorldMap({ countryName, coordinates }: InteractiveWorldMapProps) {
  // 代替案：OpenStreetMap静的画像
  const osmMapUrl = `https://www.openstreetmap.org/export/embed.html?bbox=${coordinates.lng-10},${coordinates.lat-10},${coordinates.lng+10},${coordinates.lat+10}&layer=mapnik&marker=${coordinates.lat},${coordinates.lng}`;

  return (
    <div className="relative w-full h-80 bg-white/5 backdrop-blur-sm rounded-2xl overflow-hidden border border-white/20">
      {/* インタラクティブな地図 */}
      <iframe
        src={osmMapUrl}
        className="w-full h-full opacity-80"
        style={{ border: 0 }}
        allowFullScreen
        loading="lazy"
        title={`${countryName}の地図`}
      />
      
      {/* オーバーレイ情報 */}
      <div className="absolute top-4 left-4 bg-black/50 backdrop-blur-md rounded-lg px-4 py-2">
        <p className="text-white font-medium flex items-center gap-2">
          <MapPin className="w-4 h-4" />
          {countryName}
        </p>
        <p className="text-white/70 text-sm">
          緯度: {coordinates.lat.toFixed(2)}°, 経度: {coordinates.lng.toFixed(2)}°
        </p>
      </div>
      
      {/* ズームヒント */}
      <div className="absolute bottom-4 right-4 bg-black/50 backdrop-blur-md rounded-lg px-3 py-1">
        <p className="text-white/70 text-xs">
          🖱️ ドラッグで移動・スクロールでズーム
        </p>
      </div>
    </div>
  );
}