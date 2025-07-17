import { useState, useCallback, DragEvent } from 'react';

export interface DragAndDropOptions {
  onDrop: (files: FileList) => void;
  accept?: string[];
  maxFiles?: number;
  maxSize?: number; // in bytes
  onError?: (error: string) => void;
}

export function useDragAndDrop(options: DragAndDropOptions) {
  const [isDragging, setIsDragging] = useState(false);
  const [dragCounter, setDragCounter] = useState(0);

  const validateFiles = useCallback((files: FileList): boolean => {
    if (options.maxFiles && files.length > options.maxFiles) {
      options.onError?.(`最大${options.maxFiles}個のファイルまで選択可能です`);
      return false;
    }

    for (let i = 0; i < files.length; i++) {
      const file = files[i];
      
      // File type validation
      if (options.accept && !options.accept.some(type => file.type.includes(type))) {
        options.onError?.(`サポートされていないファイル形式です: ${file.name}`);
        return false;
      }

      // File size validation
      if (options.maxSize && file.size > options.maxSize) {
        const maxSizeMB = (options.maxSize / (1024 * 1024)).toFixed(1);
        options.onError?.(`ファイルサイズが上限(${maxSizeMB}MB)を超えています: ${file.name}`);
        return false;
      }
    }

    return true;
  }, [options]);

  const handleDragEnter = useCallback((e: DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragCounter(prev => prev + 1);
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragCounter(prev => {
      const newCount = prev - 1;
      if (newCount === 0) {
        setIsDragging(false);
      }
      return newCount;
    });
  }, []);

  const handleDragOver = useCallback((e: DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
  }, []);

  const handleDrop = useCallback((e: DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    
    setIsDragging(false);
    setDragCounter(0);

    const files = e.dataTransfer.files;
    if (files.length > 0 && validateFiles(files)) {
      options.onDrop(files);
    }
  }, [options, validateFiles]);

  const resetDragState = useCallback(() => {
    setIsDragging(false);
    setDragCounter(0);
  }, []);

  return {
    isDragging,
    dragHandlers: {
      onDragEnter: handleDragEnter,
      onDragLeave: handleDragLeave,
      onDragOver: handleDragOver,
      onDrop: handleDrop,
    },
    resetDragState,
  };
}