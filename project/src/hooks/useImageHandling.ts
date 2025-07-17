import { useState, useCallback } from 'react';

export interface ImageHandlingOptions {
  maxSize?: number; // in bytes
  allowedTypes?: string[];
  onError?: (error: string) => void;
}

export function useImageHandling(options: ImageHandlingOptions = {}) {
  const [isLoading, setIsLoading] = useState(false);
  const [imagePreview, setImagePreview] = useState<string | null>(null);

  const {
    maxSize = 10 * 1024 * 1024, // 10MB default
    allowedTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/gif', 'image/webp'],
    onError
  } = options;

  const validateImage = useCallback((file: File): boolean => {
    // File type validation
    if (!allowedTypes.includes(file.type)) {
      onError?.('サポートされていない画像形式です。JPEG, PNG, GIF, WebP形式をご利用ください。');
      return false;
    }

    // File size validation
    if (file.size > maxSize) {
      const maxSizeMB = (maxSize / (1024 * 1024)).toFixed(1);
      onError?.(`ファイルサイズが上限(${maxSizeMB}MB)を超えています。`);
      return false;
    }

    return true;
  }, [allowedTypes, maxSize, onError]);

  const processImage = useCallback(async (file: File): Promise<string | null> => {
    if (!validateImage(file)) {
      return null;
    }

    setIsLoading(true);

    try {
      // Create image preview
      const reader = new FileReader();
      const imageUrl = await new Promise<string>((resolve, reject) => {
        reader.onload = (e) => {
          if (e.target?.result) {
            resolve(e.target.result as string);
          } else {
            reject(new Error('画像の読み込みに失敗しました'));
          }
        };
        reader.onerror = () => reject(new Error('画像の読み込みに失敗しました'));
        reader.readAsDataURL(file);
      });

      // Validate image by creating Image object
      await new Promise<void>((resolve, reject) => {
        const img = new Image();
        img.onload = () => resolve();
        img.onerror = () => reject(new Error('無効な画像ファイルです'));
        img.src = imageUrl;
      });

      setImagePreview(imageUrl);
      return imageUrl;
    } catch (error) {
      onError?.(error instanceof Error ? error.message : '画像の処理に失敗しました');
      return null;
    } finally {
      setIsLoading(false);
    }
  }, [validateImage, onError]);

  const handleImageUpload = useCallback(async (files: FileList): Promise<File | null> => {
    if (files.length === 0) {
      return null;
    }

    const file = files[0];
    const imageUrl = await processImage(file);
    
    return imageUrl ? file : null;
  }, [processImage]);

  const clearImage = useCallback(() => {
    setImagePreview(null);
  }, []);

  const resetState = useCallback(() => {
    setIsLoading(false);
    setImagePreview(null);
  }, []);

  return {
    isLoading,
    imagePreview,
    processImage,
    handleImageUpload,
    clearImage,
    resetState,
    validateImage
  };
}