import React from 'react';

interface GenderSelectorProps {
  selectedGender: 'male' | 'female';
  onGenderChange: (gender: 'male' | 'female') => void;
}

export default function GenderSelector({ selectedGender, onGenderChange }: GenderSelectorProps) {
  return (
    <div className="w-full max-w-md mx-auto mb-8">
      <h2 className="text-lg font-semibold text-gray-800 mb-4 text-center">
        比較対象の性別を選択してください
      </h2>
      <div className="relative bg-gray-100 rounded-full p-1 flex">
        <div
          className={`absolute top-1 bottom-1 w-1/2 bg-gradient-to-r from-purple-500 to-pink-500 rounded-full transition-all duration-300 ease-in-out ${
            selectedGender === 'male' ? 'left-1' : 'left-1/2'
          }`}
        />
        <button
          onClick={() => onGenderChange('male')}
          className={`relative z-10 flex-1 py-3 px-6 rounded-full font-medium transition-all duration-300 flex items-center justify-center gap-2 ${
            selectedGender === 'male'
              ? 'text-white'
              : 'text-gray-600 hover:text-gray-800'
          }`}
        >
          <span className="text-lg">👨</span>
          男性
        </button>
        <button
          onClick={() => onGenderChange('female')}
          className={`relative z-10 flex-1 py-3 px-6 rounded-full font-medium transition-all duration-300 flex items-center justify-center gap-2 ${
            selectedGender === 'female'
              ? 'text-white'
              : 'text-gray-600 hover:text-gray-800'
          }`}
        >
          <span className="text-lg">👩</span>
          女性
        </button>
      </div>
    </div>
  );
}