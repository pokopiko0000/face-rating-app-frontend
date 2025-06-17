import React from 'react';
import { ArrowLeft, Shield } from 'lucide-react';

interface PrivacyPolicyProps {
  onBack: () => void;
}

export default function PrivacyPolicy({ onBack }: PrivacyPolicyProps) {
  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-50 via-pink-50 to-blue-50 py-8 px-4">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <button
            onClick={onBack}
            className="inline-flex items-center gap-2 px-4 py-2 bg-white/80 backdrop-blur-sm rounded-full shadow-lg hover:shadow-xl transition-all duration-300 mb-6"
          >
            <ArrowLeft className="w-4 h-4" />
            戻る
          </button>
          
          <div className="text-center">
            <div className="inline-flex items-center gap-2 px-4 py-2 bg-white/80 backdrop-blur-sm rounded-full shadow-lg mb-6">
              <Shield className="w-6 h-6 text-purple-600" />
              <span className="font-semibold text-gray-800">プライバシーポリシー</span>
            </div>
            <h1 className="text-3xl md:text-4xl font-bold text-gray-800 mb-4">
              プライバシーポリシー
            </h1>
          </div>
        </div>

        {/* Content */}
        <div className="bg-white/80 backdrop-blur-sm rounded-3xl shadow-2xl p-8 md:p-12">
          <div className="prose prose-lg max-w-none">
            <p className="text-gray-600 mb-6">
              当サイト「AI顔診断」（以下「当サイト」）は、ユーザーの皆様の個人情報の保護に関して、以下のとおりプライバシーポリシーを定めます。
            </p>

            <section className="mb-8">
              <h2 className="text-2xl font-bold text-gray-800 mb-4">1. 収集する情報</h2>
              <div className="text-gray-700 space-y-3">
                <p>当サイトでは、以下の情報を収集する場合があります：</p>
                <ul className="list-disc list-inside space-y-2 ml-4">
                  <li>診断のためにアップロードされた画像データ（分析後即座に削除）</li>
                  <li>アクセス解析のための匿名化された統計情報</li>
                  <li>Cookie及び類似の技術による情報</li>
                </ul>
              </div>
            </section>

            <section className="mb-8">
              <h2 className="text-2xl font-bold text-gray-800 mb-4">2. 画像データの取り扱い</h2>
              <div className="text-gray-700 space-y-3">
                <p>ユーザーがアップロードした画像について：</p>
                <ul className="list-disc list-inside space-y-2 ml-4">
                  <li>画像は診断処理のためのみに使用されます</li>
                  <li>診断完了後、サーバーから即座に削除されます</li>
                  <li>画像データはデータベースに保存されません</li>
                  <li>第三者への提供は一切行いません</li>
                </ul>
              </div>
            </section>

            <section className="mb-8">
              <h2 className="text-2xl font-bold text-gray-800 mb-4">3. Cookieの使用について</h2>
              <div className="text-gray-700 space-y-3">
                <p>当サイトでは、以下の目的でCookieを使用します：</p>
                <ul className="list-disc list-inside space-y-2 ml-4">
                  <li>サイトの利用状況の分析</li>
                  <li>ユーザー体験の向上</li>
                  <li>広告の配信及び効果測定</li>
                </ul>
                <p>Cookieの無効化をご希望の場合は、ブラウザの設定から変更できます。</p>
              </div>
            </section>

            <section className="mb-8">
              <h2 className="text-2xl font-bold text-gray-800 mb-4">4. 第三者配信の広告サービスについて</h2>
              <div className="text-gray-700 space-y-3">
                <p>当サイトでは、第三者配信の広告サービス（Google AdSense等）を利用しています：</p>
                <ul className="list-disc list-inside space-y-2 ml-4">
                  <li>これらの広告配信事業者は、ユーザーの興味に応じた商品やサービスの広告を表示するため、当サイトや他のサイトへのアクセスに関する情報（氏名、住所、メール アドレス、電話番号は含まれません）を使用することがあります</li>
                  <li>このプロセスの詳細やこのような情報が広告配信事業者に使用されないようにする方法については、<a href="https://policies.google.com/technologies/ads" target="_blank" rel="noopener noreferrer" className="text-purple-600 hover:text-purple-800 underline">Googleの広告設定</a>をご覧ください</li>
                </ul>
              </div>
            </section>

            <section className="mb-8">
              <h2 className="text-2xl font-bold text-gray-800 mb-4">5. アクセス解析ツールについて</h2>
              <div className="text-gray-700 space-y-3">
                <p>当サイトでは、Googleによるアクセス解析ツール「Google Analytics」を利用しています。このGoogle Analyticsはトラフィックデータの収集のためにCookieを使用しています。このトラフィックデータは匿名で収集されており、個人を特定するものではありません。</p>
              </div>
            </section>

            <section className="mb-8">
              <h2 className="text-2xl font-bold text-gray-800 mb-4">6. 免責事項</h2>
              <div className="text-gray-700 space-y-3">
                <ul className="list-disc list-inside space-y-2 ml-4">
                  <li>当サイトの診断結果は娯楽目的であり、実際の類似性や適合性を保証するものではありません</li>
                  <li>診断結果による如何なる損害についても、当サイトは責任を負いません</li>
                  <li>当サイトのサービス内容は予告なく変更される場合があります</li>
                </ul>
              </div>
            </section>

            <section className="mb-8">
              <h2 className="text-2xl font-bold text-gray-800 mb-4">7. プライバシーポリシーの変更</h2>
              <div className="text-gray-700 space-y-3">
                <p>当サイトは、個人情報に関して適用される日本の法令を遵守するとともに、本ポリシーの内容を適宜見直しその改善に努めます。修正された最新のプライバシーポリシーは常に本ページにて開示されます。</p>
              </div>
            </section>

            <section className="mb-8">
              <h2 className="text-2xl font-bold text-gray-800 mb-4">8. お問い合わせ</h2>
              <div className="text-gray-700 space-y-3">
                <p>本ポリシーに関するお問い合わせは、当サイトのお問い合わせページよりご連絡ください。</p>
              </div>
            </section>

            <div className="text-center text-gray-500 text-sm mt-12 pt-8 border-t border-gray-200">
              <p>制定日：2025年6月</p>
              <p>最終更新日：2025年6月</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
} 