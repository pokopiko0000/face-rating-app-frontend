import React, { useState, useEffect } from 'react';
import { UnsplashImage, fetchImagesByKeyword } from '../services/imageService';

interface CountryImageProps {
  keyword: string;
  alt: string;
  className?: string;
  size?: 'small' | 'medium' | 'large';
}

const CountryImage: React.FC<CountryImageProps> = ({ 
  keyword, 
  alt, 
  className = '', 
  size = 'medium' 
}) => {
  const [image, setImage] = useState<UnsplashImage | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);

  useEffect(() => {
    const loadImage = async () => {
      try {
        setLoading(true);
        setError(false);
        const images = await fetchImagesByKeyword(keyword, 1);
        if (images.length > 0) {
          setImage(images[0]);
        } else {
          setError(true);
        }
      } catch (err) {
        console.error('Failed to load image:', err);
        setError(true);
      } finally {
        setLoading(false);
      }
    };

    loadImage();
  }, [keyword]);

  const getSizeClasses = () => {
    switch (size) {
      case 'small':
        return 'h-32 w-full';
      case 'large':
        return 'h-64 w-full';
      case 'medium':
      default:
        return 'h-48 w-full';
    }
  };

  if (loading) {
    return (
      <div className={`${getSizeClasses()} bg-gray-200 animate-pulse rounded-lg flex items-center justify-center ${className}`}>
        <div className="text-gray-400 text-sm">読み込み中...</div>
      </div>
    );
  }

  if (error || !image) {
    return (
      <div className={`${getSizeClasses()} bg-gradient-to-br from-purple-100 to-pink-100 rounded-lg flex items-center justify-center ${className}`}>
        <div className="text-center text-gray-500">
          <div className="text-2xl mb-2">🏞️</div>
          <div className="text-sm">{alt}</div>
        </div>
      </div>
    );
  }

  return (
    <div className={`${getSizeClasses()} relative overflow-hidden rounded-lg ${className}`}>
      <img
        src={image.url}
        alt={image.alt || alt}
        className="w-full h-full object-cover transition-transform duration-300 hover:scale-105"
        onError={() => setError(true)}
      />
      <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/50 to-transparent p-2">
        <p className="text-white text-xs opacity-80">
          Photo by{' '}
          <a 
            href={image.photographerUrl} 
            target="_blank" 
            rel="noopener noreferrer"
            className="underline hover:no-underline"
          >
            {image.photographer}
          </a>
          {' '}on Unsplash
        </p>
      </div>
    </div>
  );
};

export default CountryImage;