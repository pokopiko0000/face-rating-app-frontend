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
        <div className="relative group">
          <div className="relative overflow-hidden rounded-xl bg-slate-100 border border-slate-200">
            <img
              src={preview}
              alt="アップロード画像"
              className="w-full h-72 object-cover"
            />
             <div className="absolute inset-0 bg-gradient-to-t from-black/50 to-transparent"></div>
          </div>
          <button
            onClick={openFileSelector}
            className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 bg-white/80 hover:bg-white text-slate-800 font-bold py-3 px-6 rounded-lg transition-all duration-300 backdrop-blur-sm border border-slate-200/50 opacity-0 group-hover:opacity-100 transform group-hover:scale-100 scale-95"
          >
            画像を変更
          </button>
          <div className="absolute bottom-4 left-4 right-4 text-white text-base font-semibold">
            <p>画像が選択されました</p>
          </div>
        </div>
      ) : (
        <div
          className={`relative border-2 border-dashed rounded-xl p-12 text-center cursor-pointer transition-all duration-200 ${
            dragActive
              ? 'border-teal-500 bg-teal-50'
              : 'border-slate-300 bg-white hover:border-slate-400 hover:bg-slate-50'
          }`}
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
          onClick={openFileSelector}
        >
          <div className="space-y-4">
            <div className="flex justify-center text-4xl text-slate-500">
             <svg xmlns="http://www.w3.org/2000/svg" className="h-12 w-12" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
              </svg>
            </div>
            <div>
              <p className="text-lg font-semibold text-slate-800 mb-1">
                {dragActive ? 'ここにドロップして開始' : 'クリックして画像を選択'}
              </p>
              <p className="text-sm text-slate-500">
                または、ファイルをドラッグ＆ドロップ
              </p>
            </div>
            <div className="flex items-center justify-center space-x-2 text-xs text-slate-400 pt-2">
              <span className="px-2 py-1 bg-slate-100 rounded-md">JPG</span>
              <span className="px-2 py-1 bg-slate-100 rounded-md">PNG</span>
              <span className="px-2 py-1 bg-slate-100 rounded-md">WEBP</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
} 