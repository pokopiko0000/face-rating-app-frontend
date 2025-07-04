import React, { useState } from 'react';
import { useParams, Link } from 'react-router-dom';
import { ArrowLeft, Sparkles, MapPin, Users, Globe, Heart } from 'lucide-react';
import AdBanner from './AdBanner';
import CountryFlag from './CountryFlag';
import { ADS_CONFIG } from '../config/ads';
import { getCountryImage } from '../data/countryImages';

// シンプル化された国別データ
const countryData: Record<string, any> = {
  'korea': {
    name: '韓国',
    nameEn: 'South Korea',
    flag: '🇰🇷',
    code: 'kr',
    basic: { capital: 'ソウル', population: '5,100万人', language: '韓国語' },
    description: 'K-POPとK-ドラマで世界を魅了する韓国。美容大国として知られ、「Glass Skin」発祥の地。ソウルの夜景は息をのむ美しさです。',
    highlights: [
      '世界最先端の美容・スキンケア文化',
      'K-POP・K-ドラマの聖地', 
      '24時間眠らない活気あふれるソウル',
      '済州島の美しい自然とリゾート'
    ],
    whyVisit: '最新トレンドと伝統が共存する、エネルギッシュな国'
  },
  
  'finland': {
    name: 'フィンランド',
    nameEn: 'Finland',
    flag: '🇫🇮', 
    code: 'fi',
    basic: { capital: 'ヘルシンキ', population: '550万人', language: 'フィンランド語' },
    description: '「森と湖の国」フィンランドは、オーロラが舞う神秘の北欧。サウナ発祥の地で、世界一幸福な国民が暮らしています。',
    highlights: [
      '神秘的なオーロラ観測体験',
      'サウナ文化の本場',
      'ムーミンの故郷',
      '18万8千の美しい湖'
    ],
    whyVisit: '自然と共に生きる幸福な暮らしを体験できる国'
  },

  'bhutan': {
    name: 'ブータン',
    nameEn: 'Bhutan',
    flag: '🇧🇹',
    code: 'bt', 
    basic: { capital: 'ティンプー', population: '77万人', language: 'ゾンカ語' },
    description: '「最後のシャングリラ」と呼ばれるヒマラヤの小さな王国。GDP（国内総生産）ではなくGNH（国民総幸福量）を重視する世界で唯一の国。',
    highlights: [
      '世界で唯一の炭素マイナス国',
      '国民の97%が幸せと答える国',
      'タイガーズネスト僧院の絶景',
      '全国民が民族衣装を着用'
    ],
    whyVisit: '物質的豊かさとは違う、本当の幸せを見つけられる国'
  },

  'japan': {
    name: '日本',
    nameEn: 'Japan',
    flag: '🇯🇵',
    code: 'jp',
    basic: { capital: '東京', population: '1億2,500万人', language: '日本語' },
    description: '伝統と革新が見事に調和した島国。四季の美しさ、おもてなしの心、そして世界に誇る技術力で多くの人を魅了し続けています。',
    highlights: [
      '富士山と桜の絶景',
      '世界最高峰の和食文化',
      '最先端技術と伝統工芸の融合',
      '温泉文化と癒しの時間'
    ],
    whyVisit: '古き良き伝統と最新技術が共存する唯一無二の国'
  },

  'maldives': {
    name: 'モルディブ',
    nameEn: 'Maldives', 
    flag: '🇲🇻',
    code: 'mv',
    basic: { capital: 'マレ', population: '54万人', language: 'ディベヒ語' },
    description: 'インド洋に浮かぶ1,200の珊瑚島からなる楽園。透明度抜群のエメラルドブルーの海と、一島一リゾートの贅沢な時間が待っています。',
    highlights: [
      '世界最高レベルの透明度の海',
      '一島一リゾートの究極プライベート',
      '色とりどりのサンゴ礁とダイビング',
      '水上ヴィラでの非日常体験'
    ],
    whyVisit: '地上最後の楽園で究極のリゾート体験ができる国'
  },

  'iceland': {
    name: 'アイスランド',
    nameEn: 'Iceland',
    flag: '🇮🇸',
    code: 'is',
    basic: { capital: 'レイキャビク', population: '37万人', language: 'アイスランド語' },
    description: '「火と氷の国」アイスランドは、大自然のエネルギーを肌で感じられる神秘の島。オーロラ、氷河、間欠泉が織りなす絶景の宝庫です。',
    highlights: [
      '世界最大の露天温泉ブルーラグーン',
      '迫力満点のガイザー間欠泉',
      '幻想的なオーロラ観測',
      '氷河と火山の雄大な絶景'
    ],
    whyVisit: '地球の原始の力を体感できる、他では味わえない大自然の国'
  }
};

export default function CountryDetailSimple() {
  const { countryCode } = useParams<{ countryCode: string }>();
  const country = countryData[countryCode?.toLowerCase() || ''];
  const [imageLoaded, setImageLoaded] = useState(false);

  if (!country) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-purple-50 via-pink-50 to-blue-50 py-8 px-4">
        <div className="max-w-4xl mx-auto text-center">
          <h1 className="text-2xl font-bold text-gray-800 mb-4">国の情報が見つかりません</h1>
          <Link to="/" className="text-purple-600 hover:text-purple-800">
            トップページに戻る
          </Link>
        </div>
      </div>
    );
  }

  const imageUrl = getCountryImage(countryCode || '', country.name);

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-50 via-pink-50 to-blue-50">
      {/* Navigation */}
      <div className="relative z-10 p-6">
        <Link
          to="/"
          className="inline-flex items-center gap-2 px-4 py-2 bg-white/90 backdrop-blur-sm rounded-full shadow-lg hover:shadow-xl transition-all duration-300"
        >
          <ArrowLeft className="w-4 h-4" />
          戻る
        </Link>
      </div>

      {/* Hero Image Section */}
      <div className="relative h-[70vh] overflow-hidden">
        {/* Loading placeholder */}
        {!imageLoaded && (
          <div className="absolute inset-0 bg-gradient-to-br from-purple-200 to-pink-200 animate-pulse flex items-center justify-center">
            <div className="text-6xl">{country.flag}</div>
          </div>
        )}
        
        {/* Main image */}
        <img
          src={imageUrl}
          alt={`${country.name}の美しい風景`}
          className={`w-full h-full object-cover transition-opacity duration-700 ${
            imageLoaded ? 'opacity-100' : 'opacity-0'
          }`}
          onLoad={() => setImageLoaded(true)}
          onError={() => setImageLoaded(true)}
        />
        
        {/* Overlay gradient */}
        <div className="absolute inset-0 bg-gradient-to-t from-black/60 via-black/20 to-transparent" />
        
        {/* Country info overlay */}
        <div className="absolute bottom-0 left-0 right-0 p-8 text-white">
          <div className="max-w-4xl mx-auto">
            <div className="flex items-center gap-4 mb-4">
              <div className="text-5xl">{country.flag}</div>
              <CountryFlag 
                countryCode={country.code}
                countryName={country.name}
                size="large"
              />
            </div>
            <h1 className="text-5xl md:text-6xl font-bold mb-3">
              {country.name}
            </h1>
            <p className="text-xl md:text-2xl opacity-90">
              {country.nameEn}
            </p>
          </div>
        </div>
      </div>

      {/* Content Section */}
      <div className="max-w-4xl mx-auto px-6 py-12">
        {/* Basic Info */}
        <div className="bg-white/80 backdrop-blur-sm rounded-3xl shadow-xl p-8 mb-8">
          <div className="grid md:grid-cols-3 gap-6 mb-8">
            <div className="flex items-center gap-3">
              <MapPin className="w-5 h-5 text-purple-600" />
              <div>
                <div className="text-sm text-gray-500">首都</div>
                <div className="font-semibold">{country.basic.capital}</div>
              </div>
            </div>
            <div className="flex items-center gap-3">
              <Users className="w-5 h-5 text-purple-600" />
              <div>
                <div className="text-sm text-gray-500">人口</div>
                <div className="font-semibold">{country.basic.population}</div>
              </div>
            </div>
            <div className="flex items-center gap-3">
              <Globe className="w-5 h-5 text-purple-600" />
              <div>
                <div className="text-sm text-gray-500">言語</div>
                <div className="font-semibold">{country.basic.language}</div>
              </div>
            </div>
          </div>

          {/* Description */}
          <div className="mb-8">
            <p className="text-lg text-gray-700 leading-relaxed">
              {country.description}
            </p>
          </div>

          {/* Why Visit */}
          <div className="bg-gradient-to-r from-purple-50 to-pink-50 rounded-2xl p-6 mb-8">
            <h3 className="text-xl font-bold text-gray-800 mb-3 flex items-center gap-2">
              <Heart className="w-5 h-5 text-red-500" />
              なぜ{country.name}なのか
            </h3>
            <p className="text-gray-700 font-medium">
              {country.whyVisit}
            </p>
          </div>

          {/* Highlights */}
          <div>
            <h3 className="text-xl font-bold text-gray-800 mb-4 flex items-center gap-2">
              <Sparkles className="w-5 h-5 text-purple-600" />
              {country.name}の魅力
            </h3>
            <div className="grid md:grid-cols-2 gap-3">
              {country.highlights.map((highlight: string, index: number) => (
                <div key={index} className="flex items-start gap-3 p-4 bg-blue-50 rounded-xl">
                  <span className="text-blue-500 font-bold">✨</span>
                  <span className="text-gray-700">{highlight}</span>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Ad Banner */}
        <div className="mb-8">
          <AdBanner 
            adSlot={ADS_CONFIG.SLOTS.RESULT}
            adFormat="horizontal"
            className="text-center"
            style={{ minHeight: '100px' }}
          />
        </div>

        {/* Call to Action */}
        <div className="text-center">
          <Link
            to="/"
            className="inline-flex items-center gap-3 px-12 py-4 bg-gradient-to-r from-purple-600 to-pink-600 text-white rounded-full font-bold text-lg hover:from-purple-700 hover:to-pink-700 transition-all duration-300 shadow-lg hover:shadow-xl transform hover:scale-105"
          >
            <Sparkles className="w-6 h-6" />
            自分も診断してみる
          </Link>
        </div>
      </div>
    </div>
  );
}