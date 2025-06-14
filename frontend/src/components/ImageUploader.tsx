'use client';

import { useState, useRef } from 'react';

interface ImageUploaderProps {
  onImageUpload: (file: File) => void;
}

export default function ImageUploader({ onImageUpload }: ImageUploaderProps) {
  const [dragActive, setDragActive] = useState(false);
  const [preview, setPreview] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFile = (file: File) => {
    if (file && file.type.startsWith('image/')) {
      onImageUpload(file);
      
      // プレビュー画像を生成
      const reader = new FileReader();
      reader.onload = (e) => {
        setPreview(e.target?.result as string);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    e.preventDefault();
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0]);
    }
  };

  const openFileSelector = () => {
    fileInputRef.current?.click();
  };

  return (
    <div className="w-full">
      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        onChange={handleChange}
        className="hidden"
      />
      
      {preview ? (
        <div className="relative">
          <div className="relative overflow-hidden rounded-2xl bg-slate-800/50 border border-slate-700/50">
            <img
              src={preview}
              alt="アップロード画像"
              className="w-full h-72 object-cover"
            />
            <div className="absolute inset-0 bg-gradient-to-t from-black/60 to-transparent"></div>
          </div>
          <button
            onClick={() => {
              setPreview(null);
              openFileSelector();
            }}
            className="absolute top-4 right-4 bg-slate-900/80 hover:bg-slate-800/80 text-white p-3 rounded-xl transition-colors duration-200 backdrop-blur-sm border border-slate-700/50"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
            </svg>
          </button>
          <div className="absolute bottom-4 left-4 right-4">
            <p className="text-white text-base font-semibold">
              画像がアップロードされました
            </p>
          </div>
        </div>
      ) : (
        <div
          className={`relative border-2 border-dashed rounded-2xl p-12 text-center cursor-pointer transition-all duration-200 ${
            dragActive
              ? 'border-slate-400 bg-slate-800/50'
              : 'border-slate-700 bg-slate-800/20 hover:border-slate-600 hover:bg-slate-800/40'
          }`}
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
          onClick={openFileSelector}
        >
          <div className="space-y-6">
            <div className="text-5xl">
              {dragActive ? '📤' : '📷'}
            </div>
            <div>
              <p className="text-xl font-semibold text-slate-200 mb-3">
                {dragActive ? 'ここにドロップ' : '顔写真をアップロード'}
              </p>
              <p className="text-base text-slate-400">
                ドラッグ&ドロップ または クリックして選択
              </p>
            </div>
            <div className="flex items-center justify-center space-x-3 text-sm text-slate-500">
              <span className="px-3 py-1 bg-slate-800/50 rounded-lg">JPG</span>
              <span className="px-3 py-1 bg-slate-800/50 rounded-lg">PNG</span>
              <span className="px-3 py-1 bg-slate-800/50 rounded-lg">WEBP</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
} 