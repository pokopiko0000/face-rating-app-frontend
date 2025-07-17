import React from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { ArrowLeft, Home } from 'lucide-react';

export default function CountryDetailNavigation() {
  const navigate = useNavigate();
  
  const handleBack = () => {
    // 診断結果から来た場合は戻る、そうでなければホームページに移動
    if (window.history.length > 1) {
      navigate(-1);
    } else {
      navigate('/');
    }
  };

  return (
    <div className="p-6 flex items-center justify-between">
      <button
        onClick={handleBack}
        className="inline-flex items-center gap-2 px-4 py-2 bg-white/20 backdrop-blur-md rounded-full shadow-lg hover:shadow-xl hover:bg-white/30 transition-all duration-300 text-white border border-white/30"
      >
        <ArrowLeft className="w-4 h-4" />
        戻る
      </button>
      
      <Link
        to="/"
        className="inline-flex items-center gap-2 px-4 py-2 bg-white/20 backdrop-blur-md rounded-full shadow-lg hover:shadow-xl hover:bg-white/30 transition-all duration-300 text-white border border-white/30"
      >
        <Home className="w-4 h-4" />
        ホーム
      </Link>
    </div>
  );
}