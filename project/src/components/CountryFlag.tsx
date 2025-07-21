import React from 'react';

interface CountryFlagProps {
  countryCode: string;
  countryName: string;
  size?: 'small' | 'medium' | 'large';
  className?: string;
}

const CountryFlag: React.FC<CountryFlagProps> = ({ 
  countryCode, 
  countryName, 
  size = 'medium',
  className = '' 
}) => {
  const getSizeClasses = () => {
    switch (size) {
      case 'small':
        return 'w-8 h-6';
      case 'large':
        return 'w-20 h-15';
      case 'medium':
      default:
        return 'w-12 h-9';
    }
  };

  // Flagpedia CDN を使用（信頼性が高い）
  const flagUrl = `https://flagcdn.com/w320/${countryCode.toLowerCase()}.png`;
  const fallbackFlagUrl = `https://flagcdn.com/${countryCode.toLowerCase()}.svg`;

  return (
    <div className={`inline-block ${className}`}>
      <img
        src={flagUrl}
        alt={`${countryName}の国旗`}
        className={`${getSizeClasses()} object-cover rounded border border-gray-200 shadow-sm`}
        onError={(e) => {
          // フォールバックでSVG版を試す
          const target = e.target as HTMLImageElement;
          if (target.src !== fallbackFlagUrl) {
            target.src = fallbackFlagUrl;
          } else {
            // それでも失敗した場合は国旗絵文字を表示する親要素を作る
            target.style.display = 'none';
            const parent = target.parentElement;
            if (parent) {
              parent.innerHTML = `<div class="${getSizeClasses()} bg-gray-100 border border-gray-200 rounded flex items-center justify-center text-lg">🏳️</div>`;
            }
          }
        }}
      />
    </div>
  );
};

export default CountryFlag;