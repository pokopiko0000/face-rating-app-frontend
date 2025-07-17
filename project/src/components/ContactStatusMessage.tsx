import React from 'react';
import { MessageCircle } from 'lucide-react';
import { SubmitStatus } from '../hooks/useContactForm';

interface ContactStatusMessageProps {
  status: SubmitStatus;
  onReset: () => void;
}

export default function ContactStatusMessage({ status, onReset }: ContactStatusMessageProps) {
  if (status === 'success') {
    return (
      <div className="text-center py-12">
        <div className="inline-flex items-center justify-center w-16 h-16 bg-green-100 rounded-full mb-6">
          <MessageCircle className="w-8 h-8 text-green-600" />
        </div>
        <h2 className="text-2xl font-bold text-gray-800 mb-4">
          お問い合わせありがとうございました
        </h2>
        <p className="text-gray-600 mb-8">
          お問い合わせを受け付けました。内容を確認の上、可能な限り迅速にご返信いたします。
        </p>
        <button
          onClick={onReset}
          className="px-6 py-3 bg-gradient-to-r from-purple-600 to-pink-600 text-white rounded-full font-medium hover:from-purple-700 hover:to-pink-700 transition-all duration-300"
        >
          新しいお問い合わせ
        </button>
      </div>
    );
  }

  if (status === 'error') {
    return (
      <div className="text-center py-12">
        <div className="inline-flex items-center justify-center w-16 h-16 bg-red-100 rounded-full mb-6">
          <MessageCircle className="w-8 h-8 text-red-600" />
        </div>
        <h2 className="text-2xl font-bold text-gray-800 mb-4">
          送信に失敗しました
        </h2>
        <p className="text-gray-600 mb-8">
          申し訳ございません。お問い合わせの送信に失敗しました。<br />
          しばらく時間をおいてから再度お試しください。
        </p>
        <button
          onClick={onReset}
          className="px-6 py-3 bg-gradient-to-r from-purple-600 to-pink-600 text-white rounded-full font-medium hover:from-purple-700 hover:to-pink-700 transition-all duration-300"
        >
          再度お問い合わせ
        </button>
      </div>
    );
  }

  return null;
}