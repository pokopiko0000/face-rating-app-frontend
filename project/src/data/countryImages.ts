import { getCountryImageWithFallback } from '../services/unsplashService';
import { countryData } from './countries';

// Backup static images for critical countries (used as fallback if Unsplash fails)
export const staticCountryImages: Record<string, string> = {
  'korea': 'https://images.unsplash.com/photo-1517154421773-0529f29ea451?w=1200&h=800&fit=crop&q=80',
  'japan': 'https://images.unsplash.com/photo-1490650034439-fd184c3c86a5?w=1200&h=800&fit=crop&q=80',
  'usa': 'https://images.unsplash.com/photo-1485738422979-f5c462d49f74?w=1200&h=800&fit=crop&q=80',
  'france': 'https://images.unsplash.com/photo-1502602898536-47ad22581b52?w=1200&h=800&fit=crop&q=80',
  'italy': 'https://images.unsplash.com/photo-1515542622106-78bda8ba0e5b?w=1200&h=800&fit=crop&q=80',
  'china': 'https://images.unsplash.com/photo-1508804185872-d7badad00f7d?w=1200&h=800&fit=crop&q=80',
  'germany': 'https://images.unsplash.com/photo-1467269204594-9661b134dd2b?w=1200&h=800&fit=crop&q=80',
  'uk': 'https://images.unsplash.com/photo-1513635269975-59663e0ac1ad?w=1200&h=800&fit=crop&q=80',
};

// フォールバック画像（国の画像が見つからない場合）
export const getFallbackImage = (): string => {
  return `https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=1200&h=800&fit=crop&q=80`;
};

// 画像取得関数 - 動的にUnsplashから国別画像を取得
export const getCountryImage = (countryCode: string): string => {
  try {
    // まず国データから国名を取得
    const country = countryData[countryCode.toLowerCase()];
    if (!country) {
      return getFallbackImage();
    }

    // 静的画像が存在する場合はそれを優先
    const staticImage = staticCountryImages[countryCode.toLowerCase()];
    if (staticImage) {
      return staticImage;
    }

    // Unsplashサービスを使用して動的に画像を取得
    return getCountryImageWithFallback(country.nameEn || country.name);
  } catch (error) {
    console.warn(`Failed to get image for country ${countryCode}:`, error);
    return getFallbackImage();
  }
};