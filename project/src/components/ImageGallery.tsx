import React, { useState, useEffect } from 'react';
import { UnsplashImage, fetchCountryImages } from '../services/imageService';

interface ImageGalleryProps {
  countryCode: string;
  countryName: string;
  imageCount?: number;
  className?: string;
}

const ImageGallery: React.FC<ImageGalleryProps> = ({ 
  countryCode, 
  countryName, 
  imageCount = 3,
  className = '' 
}) => {
  const [images, setImages] = useState<UnsplashImage[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const loadImages = async () => {
      try {
        setLoading(true);
        const fetchedImages = await fetchCountryImages(countryCode, countryName, imageCount);
        setImages(fetchedImages);
      } catch (error) {
        console.error('Failed to load gallery images:', error);
      } finally {
        setLoading(false);
      }
    };

    loadImages();
  }, [countryCode, countryName, imageCount]);

  if (loading) {
    return (
      <div className={`grid grid-cols-1 md:grid-cols-3 gap-4 ${className}`}>
        {Array.from({ length: imageCount }, (_, index) => (
          <div 
            key={index}
            className="h-48 bg-gray-200 animate-pulse rounded-lg"
          />
        ))}
      </div>
    );
  }

  if (images.length === 0) {
    return (
      <div className={`bg-gradient-to-br from-purple-100 to-pink-100 rounded-lg p-8 text-center ${className}`}>
        <div className="text-4xl mb-4">📸</div>
        <p className="text-gray-600">画像を読み込めませんでした</p>
      </div>
    );
  }

  return (
    <div className={`grid grid-cols-1 md:grid-cols-${Math.min(images.length, 3)} gap-4 ${className}`}>
      {images.map((image) => (
        <div key={image.id} className="relative group">
          <div className="h-48 overflow-hidden rounded-lg">
            <img
              src={image.url}
              alt={image.alt}
              className="w-full h-full object-cover transition-transform duration-300 group-hover:scale-105"
            />
          </div>
          <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/70 to-transparent p-2 opacity-0 group-hover:opacity-100 transition-opacity duration-300">
            <p className="text-white text-xs">
              Photo by{' '}
              <a 
                href={image.photographerUrl} 
                target="_blank" 
                rel="noopener noreferrer"
                className="underline hover:no-underline"
              >
                {image.photographer}
              </a>
            </p>
          </div>
        </div>
      ))}
    </div>
  );
};

export default ImageGallery;