import React from 'react';

export default function ContactFAQ() {
  return (
    <div className="mt-8 text-center">
      <div className="bg-white/80 backdrop-blur-sm rounded-2xl shadow-lg p-6">
        <h3 className="text-lg font-semibold text-gray-800 mb-3">よくあるご質問</h3>
        <div className="text-sm text-gray-600 space-y-2">
          <p><strong>Q: 診断結果は正確ですか？</strong></p>
          <p>A: 当サイトの診断は娯楽目的であり、実際の類似性を保証するものではありません。</p>
          
          <p className="mt-4"><strong>Q: アップロードした画像はどうなりますか？</strong></p>
          <p>A: 診断処理後、即座にサーバーから削除され、保存されることはありません。</p>
        </div>
      </div>
    </div>
  );
}