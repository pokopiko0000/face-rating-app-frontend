import React, { useState, useEffect } from 'react';
import { Sparkles, Brain, Eye } from 'lucide-react';
import AdBanner from './AdBanner';
import { ADS_CONFIG } from '../config/ads';

const messages = [
  "AIがあなたの顔の特徴をスキャンしています…",
  "世界中の膨大な顔データとあなたを照合中…",
  "あなたと最も相性の良い国を探しています！",
  "あなたが行くとモテモテな国が分かります！",
  "あなたの顔が最も輝く国を探しています…",
];

export default function LoadingScreen() {
  const [messageIndex, setMessageIndex] = useState(0);

  useEffect(() => {
    const intervalId = setInterval(() => {
      setMessageIndex(prevIndex => (prevIndex + 1) % messages.length);
    }, 2500); // 2.5秒ごとにインデックスを更新

    return () => clearInterval(intervalId);
  }, []);

  return (
    <div className="fixed inset-0 bg-gradient-to-br from-purple-600 via-pink-600 to-blue-600 flex items-center justify-center z-50 p-4">
      <div className="max-w-4xl w-full">
        {/* Main Loading Content */}
        <div className="text-center text-white mb-8">
          <div className="relative mb-8">
            <div className="w-32 h-32 mx-auto relative">
              {/* Outer rotating ring */}
              <div className="absolute inset-0 border-4 border-white/20 rounded-full"></div>
              <div className="absolute inset-0 border-4 border-transparent border-t-white rounded-full animate-spin"></div>
              
              {/* Inner pulsing circle */}
              <div className="absolute inset-4 bg-white/10 rounded-full flex items-center justify-center backdrop-blur-sm">
                <Brain className="w-12 h-12 text-white animate-pulse" />
              </div>
              
              {/* Floating icons */}
              <div className="absolute -top-2 -right-2 animate-bounce">
                <Sparkles className="w-6 h-6 text-yellow-300" />
              </div>
              <div className="absolute -bottom-2 -left-2 animate-bounce" style={{ animationDelay: '0.5s' }}>
                <Eye className="w-6 h-6 text-blue-300" />
              </div>
            </div>
          </div>
          
          <div className="space-y-4">
            <h2 className="text-2xl font-bold mb-4">AI分析中...</h2>
            
            <div className="h-12 flex items-center justify-center">
              <p 
                key={messageIndex}
                className="text-lg text-white/90 animate-fade-in"
              >
                {messages[messageIndex]}
              </p>
            </div>
            
            {/* Progress bar */}
            <div className="w-64 h-2 bg-white/20 rounded-full mx-auto overflow-hidden">
              <div className="h-full bg-gradient-to-r from-yellow-300 to-pink-400 animate-pulse-slow w-full"></div>
            </div>
          </div>
        </div>

        {/* Advertisement during loading */}
        <div className="bg-white/10 backdrop-blur-sm rounded-2xl p-4 mx-4">
          <AdBanner 
            adSlot={ADS_CONFIG.SLOTS.LOADING}
            adFormat="rectangle"
            className="mx-auto"
            style={{ minHeight: '250px', maxWidth: '300px' }}
          />
        </div>
      </div>
    </div>
  );
}