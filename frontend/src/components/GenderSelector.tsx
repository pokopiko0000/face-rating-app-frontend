'use client';

import React from 'react';

// コンポーネントが受け取るプロパティ（props）の型を定義
type Gender = 'man' | 'woman';

interface GenderSelectorProps {
  selectedGender: Gender;
  onGenderChange: (gender: Gender) => void;
}

export default function GenderSelector({ selectedGender, onGenderChange }: GenderSelectorProps) {
  const genders: { id: Gender; label: string; icon: string; colors: string }[] = [
    { id: 'man', label: '男性', icon: '♂', colors: 'border-blue-500 bg-blue-50 text-blue-600' },
    { id: 'woman', label: '女性', icon: '♀', colors: 'border-pink-500 bg-pink-50 text-pink-600' },
  ];

  return (
    <div className="w-full">
      <p className="text-center text-slate-600 mb-4 font-medium">あなたの性別を選択してください</p>
      <div className="grid grid-cols-2 gap-4">
        {genders.map((gender) => (
          <button
            key={gender.id}
            onClick={() => onGenderChange(gender.id)}
            className={`relative p-6 rounded-xl border-2 transition-all duration-200 text-center ${
              selectedGender === gender.id
                ? gender.colors
                : 'border-slate-300 bg-white text-slate-500 hover:border-slate-400 hover:bg-slate-50'
            }`}
          >
            <div className="text-3xl mb-2">{gender.icon}</div>
            <span className="font-semibold text-base">{gender.label}</span>
            {selectedGender === gender.id && (
              <div className="absolute top-2 right-2 w-5 h-5 bg-white rounded-full flex items-center justify-center">
                  <div className={`w-3 h-3 ${selectedGender === 'man' ? 'bg-blue-500' : 'bg-pink-500'} rounded-full`}></div>
              </div>
            )}
          </button>
        ))}
      </div>
    </div>
  );
} 