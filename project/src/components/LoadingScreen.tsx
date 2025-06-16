import React from 'react';
import { Sparkles, Brain, Eye } from 'lucide-react';

export default function LoadingScreen() {
  return (
    <div className="fixed inset-0 bg-gradient-to-br from-purple-600 via-pink-600 to-blue-600 flex items-center justify-center z-50">
      <div className="text-center text-white">
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
          <h2 className="text-2xl font-bold">AI分析中...</h2>
          <div className="space-y-2">
            <div className="flex items-center justify-center gap-2 text-white/80">
              <div className="w-2 h-2 bg-white rounded-full animate-pulse"></div>
              <span>顔の特徴を検出しています</span>
            </div>
            <div className="flex items-center justify-center gap-2 text-white/80" style={{ animationDelay: '1s' }}>
              <div className="w-2 h-2 bg-white rounded-full animate-pulse"></div>
              <span>有名人データベースと照合中</span>
            </div>
            <div className="flex items-center justify-center gap-2 text-white/80" style={{ animationDelay: '2s' }}>
              <div className="w-2 h-2 bg-white rounded-full animate-pulse"></div>
              <span>類似度を計算しています</span>
            </div>
          </div>
          
          {/* Progress bar */}
          <div className="w-64 h-2 bg-white/20 rounded-full mx-auto overflow-hidden">
            <div className="h-full bg-gradient-to-r from-yellow-400 to-pink-400 rounded-full animate-pulse"></div>
          </div>
        </div>
      </div>
    </div>
  );
}