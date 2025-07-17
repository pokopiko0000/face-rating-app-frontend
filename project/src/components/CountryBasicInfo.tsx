import React from 'react';
import { MapPin, Users, Globe } from 'lucide-react';
import InteractiveWorldMap from './InteractiveWorldMap';
import type { Country } from '../../../../shared/types';

interface CountryBasicInfoProps {
  country: Country;
}

export default function CountryBasicInfo({ country }: CountryBasicInfoProps) {
  return (
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
  );
}