import React, { useCallback, useState } from 'react';
import { Upload, X, Camera, Shield } from 'lucide-react';

interface ImageUploadProps {
  onImageSelect: (file: File) => void;
  selectedImage: File | null;
  onImageRemove: () => void;
}

export default function ImageUpload({ onImageSelect, selectedImage, onImageRemove }: ImageUploadProps) {
  const [isDragOver, setIsDragOver] = useState(false);
  const [imagePreview, setImagePreview] = useState<string | null>(null);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    
    const files = Array.from(e.dataTransfer.files);
    const imageFile = files.find(file => file.type.startsWith('image/'));
    
    if (imageFile) {
      onImageSelect(imageFile);
      const reader = new FileReader();
      reader.onload = (e) => {
        setImagePreview(e.target?.result as string);
      };
      reader.readAsDataURL(imageFile);
    }
  }, [onImageSelect]);

  const handleFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file && file.type.startsWith('image/')) {
      onImageSelect(file);
      const reader = new FileReader();
      reader.onload = (e) => {
        setImagePreview(e.target?.result as string);
      };
      reader.readAsDataURL(file);
    }
  }, [onImageSelect]);

  const handleRemoveImage = useCallback(() => {
    onImageRemove();
    setImagePreview(null);
  }, [onImageRemove]);

  React.useEffect(() => {
    if (!selectedImage) {
      setImagePreview(null);
    }
  }, [selectedImage]);

  return (
    <div className="w-full max-w-md mx-auto mb-8">
      <h2 className="text-lg font-semibold text-gray-800 mb-4 text-center">
        顔写真をアップロードしてください
      </h2>
      
      {!selectedImage ? (
        <div
          className={`relative border-2 border-dashed rounded-2xl p-8 text-center transition-all duration-300 cursor-pointer ${
            isDragOver
              ? 'border-purple-400 bg-purple-50 scale-105'
              : 'border-gray-300 hover:border-purple-400 hover:bg-gray-50'
          }`}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={() => document.getElementById('fileInput')?.click()}
        >
          <input
            id="fileInput"
            type="file"
            accept="image/*"
            onChange={handleFileSelect}
            className="hidden"
          />
          
          <div className="flex flex-col items-center gap-4">
            <div className={`p-4 rounded-full transition-all duration-300 ${
              isDragOver ? 'bg-purple-100' : 'bg-gray-100'
            }`}>
              <Camera className={`w-8 h-8 transition-colors duration-300 ${
                isDragOver ? 'text-purple-600' : 'text-gray-400'
              }`} />
            </div>
            
            <div>
              <p className="text-lg font-medium text-gray-700 mb-2">
                写真をドラッグ&ドロップ
              </p>
              <p className="text-sm text-gray-500 mb-4">
                またはクリックしてファイルを選択
              </p>
              <div className="inline-flex items-center gap-2 px-4 py-2 bg-gradient-to-r from-purple-500 to-pink-500 text-white rounded-full font-medium hover:from-purple-600 hover:to-pink-600 transition-all duration-300">
                <Upload size={16} />
                ファイルを選択
              </div>
            </div>
          </div>
        </div>
      ) : (
        <div className="relative">
          <div className="relative overflow-hidden rounded-2xl bg-gray-100 aspect-square">
            {imagePreview && (
              <img
                src={imagePreview}
                alt="アップロードされた画像"
                className="w-full h-full object-cover"
              />
            )}
            <button
              onClick={handleRemoveImage}
              className="absolute top-3 right-3 p-2 bg-red-500 text-white rounded-full hover:bg-red-600 transition-colors duration-200 shadow-lg"
            >
              <X size={16} />
            </button>
          </div>
          <p className="text-sm text-gray-600 text-center mt-3">
            画像が選択されました
          </p>
        </div>
      )}
      
      {/* プライバシー安心メッセージ */}
      <div className="mt-4 p-3 bg-blue-50 border border-blue-200 rounded-lg">
        <div className="flex items-center gap-2 mb-1">
          <Shield className="w-4 h-4 text-blue-600" />
          <span className="text-sm font-medium text-blue-800">プライバシー保護</span>
        </div>
        <p className="text-xs text-blue-700">
          アップロードされた写真は分析後すぐに削除され、データベースには一切保存されません。
        </p>
      </div>
    </div>
  );
}