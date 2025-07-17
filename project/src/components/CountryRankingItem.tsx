import React from 'react';
import { Link } from 'react-router-dom';
import { ExternalLink } from 'lucide-react';
import { CountryRanking } from '../types';
import { getCountryCodeFromDiagnosis } from '../utils/countryCodeMapping';

interface CountryRankingItemProps {
  item: CountryRanking;
  rank: number;
}

export default function CountryRankingItem({ item, rank }: CountryRankingItemProps) {
  const countryCode = getCountryCodeFromDiagnosis(item.country, item.country_code);
  
  const flagElement = item.country_code ? (
    <img
      src={`https://flagcdn.com/w40/${item.country_code.toLowerCase()}.png`}
      alt={`${item.country}の国旗`}
      className="w-6 h-auto mr-3 rounded"
    />
  ) : (
    <span className="inline-block w-6 h-auto mr-3">🏳️</span>
  );

  const content = (
    <>
      <span className="text-lg font-bold text-gray-600 w-8">{rank}</span>
      {flagElement}
      <span className={`text-lg text-gray-800 font-medium flex-1 ${countryCode ? 'group-hover:text-purple-600 transition-colors' : ''}`}>
        {item.country}
      </span>
      <span className="text-lg font-bold text-purple-600">{Math.round(item.similarity)}点</span>
      {countryCode && (
        <ExternalLink size={16} className="ml-2 text-gray-400 group-hover:text-purple-600 transition-colors" />
      )}
    </>
  );

  return (
    <li className="group">
      {countryCode ? (
        <Link
          to={`/country/${countryCode}`}
          className="flex items-center p-3 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors duration-200 hover:shadow-md"
        >
          {content}
        </Link>
      ) : (
        <div className="flex items-center p-3 bg-gray-50 rounded-lg">
          {content}
        </div>
      )}
    </li>
  );
}