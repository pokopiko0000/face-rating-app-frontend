// Google Analytics utility functions
declare global {
  interface Window {
    gtag?: (
      command: string,
      trackingId: string,
      config?: Record<string, unknown>
    ) => void;
  }
}

// export const initializeAnalytics = () => {
//   // Initialize Google Analytics if needed
//   if (typeof window !== 'undefined' && window.gtag) {
//     // Analytics initialization code would go here
//   }
// };

export const trackEvent = (eventName: string, parameters?: Record<string, unknown>) => {
  if (typeof window !== 'undefined' && window.gtag) {
    window.gtag('event', eventName, parameters);
  }
};

