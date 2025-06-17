import React from 'react';
import { ArrowLeft, FileText } from 'lucide-react';

interface TermsOfServiceProps {
  onBack: () => void;
}

export default function TermsOfService({ onBack }: TermsOfServiceProps) {
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
              <FileText className="w-6 h-6 text-purple-600" />
              <span className="font-semibold text-gray-800">利用規約</span>
            </div>
            <h1 className="text-3xl md:text-4xl font-bold text-gray-800 mb-4">
              利用規約
            </h1>
          </div>
        </div>

        {/* Content */}
        <div className="bg-white/80 backdrop-blur-sm rounded-3xl shadow-2xl p-8 md:p-12">
          <div className="prose prose-lg max-w-none">
            <p className="text-gray-600 mb-6">
              当サイト「AI顔診断」（以下「当サービス」）をご利用いただく前に、以下の利用規約をお読みください。当サービスをご利用いただくことで、本規約に同意したものとみなします。
            </p>

            <section className="mb-8">
              <h2 className="text-2xl font-bold text-gray-800 mb-4">第1条（適用）</h2>
              <div className="text-gray-700 space-y-3">
                <p>本規約は、当サービスの利用に関して、当サービス提供者（以下「当方」）とユーザーとの間に適用されるものとします。</p>
                <p>本規約に加えて、当サービスの利用に関する個別の規定がある場合、それらも本規約の一部を構成するものとします。</p>
              </div>
            </section>

            <section className="mb-8">
              <h2 className="text-2xl font-bold text-gray-800 mb-4">第2条（サービスの内容）</h2>
              <div className="text-gray-700 space-y-3">
                <p>当サービスは、AI技術を用いて顔写真を分析し、娯楽目的として「どの国で魅力的とされるか」を診断するサービスです。</p>
                <p>診断結果は統計的データに基づく推測であり、実際の類似性や適合性を保証するものではありません。</p>
                <p>当サービスは無料で提供されますが、広告が表示される場合があります。</p>
              </div>
            </section>

            <section className="mb-8">
              <h2 className="text-2xl font-bold text-gray-800 mb-4">第3条（利用の条件）</h2>
              <div className="text-gray-700 space-y-3">
                <p>ユーザーは、以下の条件を満たす場合にのみ当サービスを利用できます：</p>
                <ul className="list-disc list-inside space-y-2 ml-4">
                  <li>本規約に同意していること</li>
                  <li>法的に有効な同意を行う能力を有していること</li>
                  <li>当サービスを適切な目的で利用すること</li>
                </ul>
              </div>
            </section>

            <section className="mb-8">
              <h2 className="text-2xl font-bold text-gray-800 mb-4">第4条（禁止事項）</h2>
              <div className="text-gray-700 space-y-3">
                <p>ユーザーは、当サービスの利用にあたり、以下の行為を行ってはなりません：</p>
                <ul className="list-disc list-inside space-y-2 ml-4">
                  <li>法令または公序良俗に反する行為</li>
                  <li>犯罪行為に関連する行為</li>
                  <li>他人の著作権、肖像権、プライバシー権、その他の権利を侵害する行為</li>
                  <li>他人になりすます行為</li>
                  <li>未成年者の写真を無断でアップロードする行為</li>
                  <li>当サービスの運営を妨害する行為</li>
                  <li>コンピュータウイルス等の有害なプログラムを送信する行為</li>
                  <li>その他、当方が不適切と判断する行為</li>
                </ul>
              </div>
            </section>

            <section className="mb-8">
              <h2 className="text-2xl font-bold text-gray-800 mb-4">第5条（画像の取り扱い）</h2>
              <div className="text-gray-700 space-y-3">
                <p>アップロードされた画像について：</p>
                <ul className="list-disc list-inside space-y-2 ml-4">
                  <li>画像は診断処理のためのみに使用されます</li>
                  <li>診断完了後、サーバーから自動的に削除されます</li>
                  <li>画像データはデータベースに保存されません</li>
                  <li>第三者への提供は一切行いません</li>
                  <li>ユーザーは、アップロードする画像について適切な権利を有している必要があります</li>
                </ul>
              </div>
            </section>

            <section className="mb-8">
              <h2 className="text-2xl font-bold text-gray-800 mb-4">第6条（免責事項）</h2>
              <div className="text-gray-700 space-y-3">
                <ul className="list-disc list-inside space-y-2 ml-4">
                  <li>当サービスの診断結果は娯楽目的であり、実際の類似性、適合性、将来の成果等を保証するものではありません</li>
                  <li>診断結果に基づく行動により生じた損害について、当方は一切の責任を負いません</li>
                  <li>当サービスの利用により生じた直接的または間接的な損害について、当方は責任を負いません</li>
                  <li>当サービスは現状有姿で提供され、可用性、正確性、安全性等について保証しません</li>
                  <li>技術的な問題により一時的にサービスが利用できない場合があります</li>
                </ul>
              </div>
            </section>

            <section className="mb-8">
              <h2 className="text-2xl font-bold text-gray-800 mb-4">第7条（サービスの変更・中止）</h2>
              <div className="text-gray-700 space-y-3">
                <p>当方は、ユーザーに事前に通知することなく、当サービスの内容を変更し、または当サービスの提供を中止することができるものとします。</p>
                <p>これらの変更または中止により生じた損害について、当方は一切の責任を負いません。</p>
              </div>
            </section>

            <section className="mb-8">
              <h2 className="text-2xl font-bold text-gray-800 mb-4">第8条（利用規約の変更）</h2>
              <div className="text-gray-700 space-y-3">
                <p>当方は、必要と判断した場合には、ユーザーに通知することなく、いつでも本規約を変更することができるものとします。</p>
                <p>変更後の利用規約は、当サイトに掲載したときから効力を生じるものとします。</p>
              </div>
            </section>

            <section className="mb-8">
              <h2 className="text-2xl font-bold text-gray-800 mb-4">第9条（個人情報の保護）</h2>
              <div className="text-gray-700 space-y-3">
                <p>当方は、当サービスの利用によって取得する個人情報については、当方のプライバシーポリシーに従い適切に取り扱うものとします。</p>
              </div>
            </section>

            <section className="mb-8">
              <h2 className="text-2xl font-bold text-gray-800 mb-4">第10条（準拠法・裁判管轄）</h2>
              <div className="text-gray-700 space-y-3">
                <p>本規約の解釈にあたっては、日本法を準拠法とします。</p>
                <p>当サービスに関して紛争が生じた場合には、当方の所在地を管轄する裁判所を専属的合意管轄とします。</p>
              </div>
            </section>

            <div className="bg-yellow-50 border border-yellow-200 rounded-xl p-6 mt-8">
              <h3 className="text-lg font-semibold text-yellow-800 mb-3">重要なお知らせ</h3>
              <div className="text-sm text-yellow-700 space-y-2">
                <p>• 当サービスは娯楽目的のツールです。診断結果を過度に信頼せず、楽しみの範囲でご利用ください。</p>
                <p>• 他人の写真を無断でアップロードすることは禁止されています。</p>
                <p>• 未成年者の方は、保護者の同意を得てからご利用ください。</p>
              </div>
            </div>

            <div className="text-center text-gray-500 text-sm mt-12 pt-8 border-t border-gray-200">
              <p>制定日：2024年12月</p>
              <p>最終更新日：2024年12月</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
} 