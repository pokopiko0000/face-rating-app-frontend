import React from 'react';
import { Sparkles } from 'lucide-react';
import { getHighlightImage } from '../data/countryHighlightImages';
import type { Country, CountryHighlight } from '../../../../shared/types';

interface CountryHighlightsProps {
  country: Country;
  countryCode: string;
}

export default function CountryHighlights({ country, countryCode }: CountryHighlightsProps) {
  return (
    <div className="bg-white/10 backdrop-blur-md rounded-3xl p-8 border border-white/20 shadow-2xl">
      <h3 className="text-xl font-bold text-white mb-6 flex items-center gap-2">
        <Sparkles className="w-5 h-5 text-purple-300" />
        {country.name}の魅力
      </h3>
      <div className="grid md:grid-cols-2 gap-4">
        {country.highlights.map((highlight: CountryHighlight, index: number) => (
          <div key={index} className="bg-white/10 backdrop-blur-sm rounded-xl border border-white/20 overflow-hidden hover:bg-white/15 transition-all duration-300">
            <div className="flex">
              {/* 左側：画像 */}
              <div className="w-40 h-32 flex-shrink-0 bg-white/10">
                <img 
                  src={getHighlightImage(highlight.title, countryCode)} 
                  alt={highlight.title}
                  className="w-full h-full object-cover"
                  crossOrigin="anonymous"
                  onError={(e) => {
                    const target = e.target as HTMLImageElement;
                    target.src = 'https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=400&h=300&fit=crop&q=80';
                  }}
                />
              </div>
              {/* 右側：テキスト */}
              <div className="flex-1 p-4">
                <h4 className="font-bold text-white mb-2">{highlight.title}</h4>
                <p className="text-white/80 text-sm leading-relaxed">{highlight.description}</p>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}