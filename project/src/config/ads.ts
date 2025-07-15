// 広告設定
export const ADS_CONFIG = {
  // Google AdSense設定
  ADSENSE_CLIENT_ID: import.meta.env.VITE_ADSENSE_CLIENT_ID || '',
  
  // 広告スロットID
  SLOTS: {
    FOOTER: import.meta.env.VITE_ADSENSE_SLOT_FOOTER || '',
    LOADING: import.meta.env.VITE_ADSENSE_SLOT_LOADING || '',
    RESULT: import.meta.env.VITE_ADSENSE_SLOT_RESULT || '',
  },
  
  // Google Analytics設定
  GA_MEASUREMENT_ID: import.meta.env.VITE_GA_MEASUREMENT_ID || '',
};

// 広告が有効かどうかを判定
export const isAdsEnabled = () => {
  return ADS_CONFIG.ADSENSE_CLIENT_ID && 
         ADS_CONFIG.ADSENSE_CLIENT_ID !== '' &&
         ADS_CONFIG.ADSENSE_CLIENT_ID.startsWith('ca-pub-');
}; 