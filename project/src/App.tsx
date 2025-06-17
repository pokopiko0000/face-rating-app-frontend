import React, { useState } from 'react';
import { Sparkles } from 'lucide-react';
import GenderSelector from './components/GenderSelector';
import ImageUpload from './components/ImageUpload';
import LoadingScreen from './components/LoadingScreen';
import ResultDisplay from './components/ResultDisplay';
import ErrorMessage from './components/ErrorMessage';
import PrivacyPolicy from './components/PrivacyPolicy';
import Contact from './components/Contact';
import TermsOfService from './components/TermsOfService';
import AdBanner from './components/AdBanner';
import { ADS_CONFIG } from './config/ads';
import { useDiagnosis } from './hooks/useDiagnosis';

type PageType = 'home' | 'privacy' | 'contact' | 'terms';

function App() {
  const [currentPage, setCurrentPage] = useState<PageType>('home');
  const [selectedGender, setSelectedGender] = useState<'male' | 'female'>('female');
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  
  const { isLoading, result, error, diagnose, reset } = useDiagnosis();

  const handleImageSelect = (file: File) => {
    setSelectedImage(file);
    const reader = new FileReader();
    reader.onload = (e) => {
      setImagePreview(e.target?.result as string);
    };
    reader.readAsDataURL(file);
  };

  const handleImageRemove = () => {
    setSelectedImage(null);
    setImagePreview(null);
  };

  const handleDiagnose = async () => {
    if (!selectedImage) {
      alert('画像を選択してください');
      return;
    }

    await diagnose({
      image: selectedImage,
      gender: selectedGender
    });
  };

  const handleReset = () => {
    reset();
    setSelectedImage(null);
    setImagePreview(null);
    setCurrentPage('home');
  };

  const handleBackToHome = () => {
    setCurrentPage('home');
  };

  // Page routing
  if (currentPage === 'privacy') {
    return <PrivacyPolicy onBack={handleBackToHome} />;
  }

  if (currentPage === 'contact') {
    return <Contact onBack={handleBackToHome} />;
  }

  if (currentPage === 'terms') {
    return <TermsOfService onBack={handleBackToHome} />;
  }

  if (isLoading) {
    return <LoadingScreen />;
  }

  if (result && imagePreview) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-purple-50 via-pink-50 to-blue-50 py-8 px-4">
        <ResultDisplay
          result={result}
          userImage={imagePreview}
          onReset={handleReset}
          gender={selectedGender}
        />
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-50 via-pink-50 to-blue-50">
      {/* Header */}
      <div className="text-center pt-8 pb-12 px-4">
        <div className="inline-flex items-center gap-2 px-4 py-2 bg-white/80 backdrop-blur-sm rounded-full shadow-lg mb-6">
          <Sparkles className="w-6 h-6 text-purple-600" />
          <span className="font-semibold text-gray-800">AI顔診断</span>
        </div>
        
        <h1 className="text-4xl md:text-5xl font-bold text-gray-800 mb-4">
          あなたの顔はどこの国で人気？
          <br />
          <span className="bg-gradient-to-r from-purple-600 to-pink-600 bg-clip-text text-transparent">
            あなたがモテる国をAIが診断
          </span>
        </h1>
        
        <p className="text-lg text-gray-600 max-w-2xl mx-auto">
          最新AIがあなたの顔をグローバル基準で分析！
          <br />
          世界の中で、あなたの魅力が最も輝く国はどこ？ 驚きの診断結果が、新しい出会いのきっかけになるかも。
        </p>
      </div>

      {/* Main Content */}
      <div className="max-w-4xl mx-auto px-4 pb-12">
        <div className="bg-white/80 backdrop-blur-sm rounded-3xl shadow-2xl p-8 md:p-12">
          {error ? (
            <ErrorMessage message={error} onRetry={handleReset} />
          ) : (
            <>
              <GenderSelector
                selectedGender={selectedGender}
                onGenderChange={setSelectedGender}
              />
              
              <ImageUpload
                onImageSelect={handleImageSelect}
                selectedImage={selectedImage}
                onImageRemove={handleImageRemove}
              />
              
              <div className="text-center">
                <button
                  onClick={handleDiagnose}
                  disabled={!selectedImage}
                  className={`px-12 py-4 rounded-2xl font-bold text-lg transition-all duration-300 transform ${
                    selectedImage
                      ? 'bg-gradient-to-r from-purple-600 to-pink-600 text-white hover:from-purple-700 hover:to-pink-700 hover:scale-105 shadow-lg hover:shadow-xl'
                      : 'bg-gray-200 text-gray-400 cursor-not-allowed'
                  }`}
                >
                  <span className="flex items-center gap-2">
                    <Sparkles className="w-5 h-5" />
                    AI診断を開始
                  </span>
                </button>
                
                {!selectedImage && (
                  <p className="text-sm text-gray-500 mt-3">
                    ※ 画像を選択してから診断ボタンを押してください
                  </p>
                )}
              </div>
            </>
          )}
        </div>
      </div>

      {/* Advertisement Banner */}
      <div className="max-w-4xl mx-auto px-4 mb-8">
        <AdBanner 
          adSlot={ADS_CONFIG.SLOTS.FOOTER}
          adFormat="horizontal"
          className="text-center"
          style={{ minHeight: '100px' }}
        />
      </div>

      {/* Footer */}
      <div className="text-center pb-8 px-4">
        <p className="text-sm text-gray-500 mb-4">
          ※ この診断は娯楽目的です。実際の類似性を保証するものではありません。
        </p>
        
        {/* Footer Links */}
        <div className="flex flex-wrap justify-center gap-4 mb-4">
          <button
            onClick={() => setCurrentPage('privacy')}
            className="text-sm text-gray-400 hover:text-purple-600 transition-colors duration-300"
          >
            プライバシーポリシー
          </button>
          <span className="text-gray-300">|</span>
          <button
            onClick={() => setCurrentPage('terms')}
            className="text-sm text-gray-400 hover:text-purple-600 transition-colors duration-300"
          >
            利用規約
          </button>
          <span className="text-gray-300">|</span>
          <button
            onClick={() => setCurrentPage('contact')}
            className="text-sm text-gray-400 hover:text-purple-600 transition-colors duration-300"
          >
            お問い合わせ
          </button>
        </div>
        
        <p className="text-xs text-gray-400">
          © 2025 AI顔診断. All rights reserved.
        </p>
      </div>
    </div>
  );
}

export default App;