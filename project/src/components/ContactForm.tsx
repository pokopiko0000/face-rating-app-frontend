import React from 'react';
import { Send } from 'lucide-react';
import { ContactFormData } from '../hooks/useContactForm';

interface ContactFormProps {
  formData: ContactFormData;
  isSubmitting: boolean;
  onInputChange: (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement>) => void;
  onSubmit: (e: React.FormEvent) => void;
}

export default function ContactForm({ formData, isSubmitting, onInputChange, onSubmit }: ContactFormProps) {
  return (
    <form onSubmit={onSubmit} className="space-y-6">
      <div className="grid md:grid-cols-2 gap-6">
        <div>
          <label htmlFor="name" className="block text-sm font-medium text-gray-700 mb-2">
            お名前 <span className="text-red-500">*</span>
          </label>
          <input
            type="text"
            id="name"
            name="name"
            required
            value={formData.name}
            onChange={onInputChange}
            className="w-full px-4 py-3 border border-gray-300 rounded-xl focus:ring-2 focus:ring-purple-500 focus:border-transparent transition-all duration-300"
            placeholder="山田太郎"
          />
        </div>
        
        <div>
          <label htmlFor="email" className="block text-sm font-medium text-gray-700 mb-2">
            メールアドレス <span className="text-red-500">*</span>
          </label>
          <input
            type="email"
            id="email"
            name="email"
            required
            value={formData.email}
            onChange={onInputChange}
            className="w-full px-4 py-3 border border-gray-300 rounded-xl focus:ring-2 focus:ring-purple-500 focus:border-transparent transition-all duration-300"
            placeholder="example@email.com"
          />
        </div>
      </div>

      <div>
        <label htmlFor="subject" className="block text-sm font-medium text-gray-700 mb-2">
          お問い合わせ種別 <span className="text-red-500">*</span>
        </label>
        <select
          id="subject"
          name="subject"
          required
          value={formData.subject}
          onChange={onInputChange}
          className="w-full px-4 py-3 border border-gray-300 rounded-xl focus:ring-2 focus:ring-purple-500 focus:border-transparent transition-all duration-300"
        >
          <option value="">選択してください</option>
          <option value="general">一般的なお問い合わせ</option>
          <option value="bug">不具合の報告</option>
          <option value="feature">機能の要望</option>
          <option value="privacy">プライバシーに関するお問い合わせ</option>
          <option value="business">ビジネス・提携に関するお問い合わせ</option>
          <option value="other">その他</option>
        </select>
      </div>

      <div>
        <label htmlFor="message" className="block text-sm font-medium text-gray-700 mb-2">
          お問い合わせ内容 <span className="text-red-500">*</span>
        </label>
        <textarea
          id="message"
          name="message"
          required
          rows={6}
          value={formData.message}
          onChange={onInputChange}
          className="w-full px-4 py-3 border border-gray-300 rounded-xl focus:ring-2 focus:ring-purple-500 focus:border-transparent transition-all duration-300 resize-vertical"
          placeholder="お問い合わせ内容を詳しくご記入ください..."
        />
      </div>

      <div className="bg-blue-50 border border-blue-200 rounded-xl p-4">
        <p className="text-sm text-blue-700">
          <strong>ご注意：</strong> 
          お問い合わせの内容によっては、ご返信にお時間をいただく場合があります。
          また、内容によってはご返信できない場合もございますので、予めご了承ください。
        </p>
      </div>

      <div className="text-center">
        <button
          type="submit"
          disabled={isSubmitting}
          className={`px-8 py-4 rounded-2xl font-bold text-lg transition-all duration-300 transform ${
            isSubmitting
              ? 'bg-gray-400 text-gray-200 cursor-not-allowed'
              : 'bg-gradient-to-r from-purple-600 to-pink-600 text-white hover:from-purple-700 hover:to-pink-700 hover:scale-105 shadow-lg hover:shadow-xl'
          }`}
        >
          <span className="flex items-center gap-2">
            <Send className="w-5 h-5" />
            {isSubmitting ? '送信中...' : 'お問い合わせを送信'}
          </span>
        </button>
      </div>
    </form>
  );
}