// 広告設定
export const ADS_CONFIG = {
  // Google AdSense設定
  ADSENSE_CLIENT_ID: import.meta.env.VITE_ADSENSE_CLIENT_ID || 'ca-pub-XXXXXXXXXXXXXXXXX',
  
  // 広告スロットID
  SLOTS: {
    FOOTER: import.meta.env.VITE_ADSENSE_SLOT_FOOTER || '1234567890',
    LOADING: import.meta.env.VITE_ADSENSE_SLOT_LOADING || '0987654321',
    RESULT: import.meta.env.VITE_ADSENSE_SLOT_RESULT || '1122334455',
  },
  
  // Google Analytics設定
  GA_MEASUREMENT_ID: import.meta.env.VITE_GA_MEASUREMENT_ID || 'GA_MEASUREMENT_ID',
};

// 広告が有効かどうかを判定
export const isAdsEnabled = () => {
  return ADS_CONFIG.ADSENSE_CLIENT_ID !== 'ca-pub-XXXXXXXXXXXXXXXXX';
}; 