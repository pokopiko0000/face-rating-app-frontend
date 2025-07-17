import React from 'react';
import CountryFlag from './CountryFlag';
import type { Country } from '../../../../shared/types';

interface CountryDetailHeaderProps {
  country: Country;
}

export default function CountryDetailHeader({ country }: CountryDetailHeaderProps) {
  return (
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
  );
}