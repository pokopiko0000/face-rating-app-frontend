import React from 'react';
import { Link } from 'react-router-dom';
import { ArrowLeft, Mail } from 'lucide-react';

export default function ContactHeader() {
  return (
    <div className="mb-8">
      <Link
        to="/"
        className="inline-flex items-center gap-2 px-4 py-2 bg-white/80 backdrop-blur-sm rounded-full shadow-lg hover:shadow-xl transition-all duration-300 mb-6"
      >
        <ArrowLeft className="w-4 h-4" />
        戻る
      </Link>
      
      <div className="text-center">
        <div className="inline-flex items-center gap-2 px-4 py-2 bg-white/80 backdrop-blur-sm rounded-full shadow-lg mb-6">
          <Mail className="w-6 h-6 text-purple-600" />
          <span className="font-semibold text-gray-800">お問い合わせ</span>
        </div>
        <h1 className="text-3xl md:text-4xl font-bold text-gray-800 mb-4">
          お問い合わせ
        </h1>
        <p className="text-lg text-gray-600 max-w-2xl mx-auto">
          ご質問、ご意見、不具合の報告など、お気軽にお問い合わせください。
          <br />
          できる限り迅速にご対応いたします。
        </p>
      </div>
    </div>
  );
}