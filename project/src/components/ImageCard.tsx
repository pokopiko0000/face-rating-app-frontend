import React, { useState } from 'react';

interface ImageCardProps {
  src: string;
  alt: string;
  fallbackSrc?: string;
  className?: string;
  children?: React.ReactNode;
}

export default function ImageCard({ 
  src, 
  alt, 
  fallbackSrc = 'https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=400&h=300&fit=crop&q=80',
  className = 'w-48 h-48 mx-auto rounded-2xl object-cover shadow-lg',
  children 
}: ImageCardProps) {
  const [imageLoaded, setImageLoaded] = useState(false);

  return (
    <div className="relative">
      {/* Loading placeholder */}
      {!imageLoaded && (
        <div className={`${className} bg-gray-200 animate-pulse`} />
      )}
      
      {/* Image */}
      <img
        src={src}
        alt={alt}
        className={`${className} transition-opacity duration-300 ${imageLoaded ? 'opacity-100' : 'opacity-0'}`}
        crossOrigin="anonymous"
        onLoad={() => setImageLoaded(true)}
        onError={(e) => {
          const target = e.target as HTMLImageElement;
          if (target.src !== fallbackSrc) {
            target.src = fallbackSrc;
          }
          setImageLoaded(true);
        }}
      />
      
      {/* Overlay content */}
      {children && (
        <div className="absolute -bottom-3 left-1/2 transform -translate-x-1/2">
          {children}
        </div>
      )}
    </div>
  );
}