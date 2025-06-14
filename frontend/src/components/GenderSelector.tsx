'use client';

import React from 'react';

// コンポーネントが受け取るプロパティ（props）の型を定義
type Gender = 'male' | 'female';

interface GenderSelectorProps {
  selectedGender: Gender;
  onGenderChange: (gender: Gender) => void;
}

export default function GenderSelector({ selectedGender, onGenderChange }: GenderSelectorProps) {
  return (
    <div className="w-full">
      <div className="mb-6">
        <h3 className="text-xl font-semibold text-slate-200 text-center">
          性別を選択してください
        </h3>
      </div>
      
      <div className="grid grid-cols-2 gap-6">
        <button
          onClick={() => onGenderChange('male')}
          className={`relative p-8 rounded-2xl border-2 transition-all duration-200 ${
            selectedGender === 'male'
              ? 'border-blue-500 bg-blue-500/10 text-blue-300'
              : 'border-slate-700 bg-slate-800/20 text-slate-300 hover:border-slate-600 hover:bg-slate-800/40'
          }`}
        >
          <div className="flex flex-col items-center space-y-4">
            <div className="text-4xl">👨</div>
            <span className="font-semibold text-lg">男性</span>
          </div>
          {selectedGender === 'male' && (
            <div className="absolute top-3 right-3">
              <div className="w-4 h-4 bg-blue-500 rounded-full"></div>
            </div>
          )}
        </button>

        <button
          onClick={() => onGenderChange('female')}
          className={`relative p-8 rounded-2xl border-2 transition-all duration-200 ${
            selectedGender === 'female'
              ? 'border-pink-500 bg-pink-500/10 text-pink-300'
              : 'border-slate-700 bg-slate-800/20 text-slate-300 hover:border-slate-600 hover:bg-slate-800/40'
          }`}
        >
          <div className="flex flex-col items-center space-y-4">
            <div className="text-4xl">👩</div>
            <span className="font-semibold text-lg">女性</span>
          </div>
          {selectedGender === 'female' && (
            <div className="absolute top-3 right-3">
              <div className="w-4 h-4 bg-pink-500 rounded-full"></div>
            </div>
          )}
        </button>
      </div>
    </div>
  );
} 