import React, { useState } from 'react';
import { useParams, Link } from 'react-router-dom';
import { ArrowLeft, Sparkles, MapPin, Users, Globe } from 'lucide-react';
import AdBanner from './AdBanner';
import CountryFlag from './CountryFlag';
import InteractiveWorldMap from './InteractiveWorldMap';
import { ADS_CONFIG } from '../config/ads';
import { getCountryImage } from '../data/countryImages';

// 画像マッピング関数
const getHighlightImage = (title: string, countryCode: string): string => {
  const imageMap: Record<string, Record<string, string>> = {
    'finland': {
      '神秘的なオーロラ観測体験': 'https://images.unsplash.com/photo-1531366936337-7c912a4589a7?w=400&h=300&fit=crop&q=80', // オーロラの実際の画像
      'サウナ文化の本場': 'https://images.unsplash.com/photo-1571902943202-507ec2618e8f?w=400&h=300&fit=crop&q=80', // フィンランドサウナの実際の画像
      'ムーミンの故郷': 'https://images.unsplash.com/photo-1578925518681-39e3ded07b3a?w=400&h=300&fit=crop&q=80', // ムーミン関連・フィンランドの童話的風景
      '18万8千の美しい湖': 'https://images.unsplash.com/photo-1527004013197-933c4bb611b3?w=400&h=300&fit=crop&q=80' // 湖の画像
    },
    'bhutan': {
      '世界で唯一の炭素マイナス国': 'https://images.unsplash.com/photo-1441974231531-c6227db76b6e?w=400&h=300&fit=crop&q=80', // 深い森林
      'タイガーズネスト僧院の絶景': 'https://images.unsplash.com/photo-1605649487212-47bdab064df0?w=400&h=300&fit=crop&q=80', // 山の僧院
      '国民総幸福量（GNH）の哲学': 'https://images.unsplash.com/photo-1529156069898-49953e39b3ac?w=400&h=300&fit=crop&q=80', // 幸せそうな人々
      '伝統的な民族衣装文化': 'https://images.unsplash.com/photo-1544112190-6d6c1e9efed8?w=400&h=300&fit=crop&q=80' // ブータンの伝統的な民族衣装
    }
  };
  
  const fallbackImage = 'https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=400&h=300&fit=crop&q=80';
  return imageMap[countryCode]?.[title] || fallbackImage;
};

// シンプル化された国別データ
const countryData: Record<string, any> = {
  'korea': {
    name: '韓国',
    nameEn: 'South Korea',
    flag: '🇰🇷',
    code: 'kr',
    basic: { capital: 'ソウル', population: '5,100万人', language: '韓国語' },
    coordinates: { lat: 37.5665, lng: 126.9780 },
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
    coordinates: { lat: 60.1699, lng: 24.9384 },
    description: '「森と湖の国」フィンランドは、オーロラが舞う神秘の北欧。サウナ発祥の地で、世界一幸福な国民が暮らしています。',
    highlights: [
      {
        title: '神秘的なオーロラ観測体験',
        description: 'ラップランド地方では年間200日以上オーロラが出現。天空に揺らめく緑色のカーテンは、一生の思い出に。'
      },
      {
        title: 'サウナ文化の本場',
        description: '人口1人あたり0.5棟という世界最高密度のサウナ大国。湖畔の伝統的なスモークサウナで心身をリフレッシュ。'
      },
      {
        title: 'ムーミンの故郷',
        description: 'ナーンタリにはムーミン博物館があり、ムーミン谷の世界を再現。作者トーヴェ・ヤンソンの生家も訪れることができます。'
      },
      {
        title: '18万8千の美しい湖',
        description: '「千湖の国」と呼ばれるフィンランド。夏は白夜で輝く湖、冬は凍った湖上でアイスフィッシングを楽しめます。'
      }
    ],
    whyVisit: '自然と共に生きる幸福な暮らしを体験できる国'
  },

  'bhutan': {
    name: 'ブータン',
    nameEn: 'Bhutan',
    flag: '🇧🇹',
    code: 'bt', 
    basic: { capital: 'ティンプー', population: '77万人', language: 'ゾンカ語' },
    coordinates: { lat: 27.5142, lng: 90.4336 },
    description: '「最後のシャングリラ」と呼ばれるヒマラヤの小さな王国。GDP（国内総生産）ではなくGNH（国民総幸福量）を重視する世界で唯一の国。',
    highlights: [
      {
        title: '世界で唯一の炭素マイナス国',
        description: '国土の72%が森林で覆われ、CO2吸収量が排出量を上回る環境先進国。憲法で森林保護が義務付けられています。'
      },
      {
        title: 'タイガーズネスト僧院の絶景',
        description: '断崖絶壁に建つ神秘的な僧院。標高3,000mの聖地で、グル・リンポチェが瞑想した場所として崇められています。'
      },
      {
        title: '国民総幸福量（GNH）の哲学',
        description: '経済発展よりも国民の幸福を重視する独自の国家理念。国民の97%が「幸せ」と答える驚異的な満足度。'
      },
      {
        title: '伝統的な民族衣装文化',
        description: '男性はゴ、女性はキラという民族衣装の着用が法律で定められ、美しい伝統文化が日常に息づいています。'
      }
    ],
    whyVisit: '物質的豊かさとは違う、本当の幸せを見つけられる国'
  },

  'japan': {
    name: '日本',
    nameEn: 'Japan',
    flag: '🇯🇵',
    code: 'jp',
    basic: { capital: '東京', population: '1億2,500万人', language: '日本語' },
    coordinates: { lat: 35.6762, lng: 139.6503 },
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
    coordinates: { lat: 4.1755, lng: 73.5093 },
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
    coordinates: { lat: 64.1466, lng: -21.9426 },
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
    <div className="relative min-h-screen overflow-hidden">
      {/* Full Screen Background Image */}
      <div className="fixed inset-0 z-0">
        {/* Loading placeholder */}
        {!imageLoaded && (
          <div className="absolute inset-0 bg-gradient-to-br from-purple-200 to-pink-200 animate-pulse" />
        )}
        
        {/* Background image */}
        <img
          src={imageUrl}
          alt={`${country.name}の美しい風景`}
          className={`w-full h-full object-cover transition-opacity duration-700 ${imageLoaded ? 'opacity-100' : 'opacity-0'}`}
          crossOrigin="anonymous"
          onLoad={() => setImageLoaded(true)}
          onError={(e) => {
            const target = e.target as HTMLImageElement;
            if (!target.src.includes('photo-1469474968028')) {
              target.src = 'https://images.unsplash.com/photo-1469474968028-56623f02e42e?w=1200&h=800&fit=crop&q=80';
            }
            setImageLoaded(true);
          }}
        />
        
        {/* Overlay gradient for better readability */}
        <div className="absolute inset-0 bg-gradient-to-b from-black/40 via-black/50 to-black/60" />
      </div>

      {/* Content Container */}
      <div className="relative z-10 min-h-screen flex flex-col">
        {/* Navigation */}
        <div className="p-6">
          <Link
            to="/"
            className="inline-flex items-center gap-2 px-4 py-2 bg-white/20 backdrop-blur-md rounded-full shadow-lg hover:shadow-xl hover:bg-white/30 transition-all duration-300 text-white border border-white/30"
          >
            <ArrowLeft className="w-4 h-4" />
            戻る
          </Link>
        </div>

        {/* Hero Section */}
        <div className="flex-1 flex items-center justify-center px-6 py-12">
          <div className="max-w-4xl w-full">
            {/* Country Header */}
            <div className="text-center mb-12">
              <div className="flex justify-center items-center gap-4 mb-4">
                <CountryFlag 
                  countryCode={country.code}
                  countryName={country.name}
                  size="large"
                />
              </div>
              <h1 className="text-6xl md:text-7xl font-bold mb-3 text-white drop-shadow-lg">
                {country.name}
              </h1>
              <p className="text-2xl md:text-3xl text-white/90">
                {country.nameEn}
              </p>
            </div>

            {/* Content Cards */}
            <div className="space-y-6">
              {/* Map and Basic Info Section */}
              <div className="bg-white/10 backdrop-blur-md rounded-3xl p-8 border border-white/20 shadow-2xl">
                <div className="grid md:grid-cols-2 gap-8">
                  {/* Left: World Map */}
                  <div>
                    <h3 className="text-xl font-bold text-white mb-4 flex items-center gap-2">
                      <Globe className="w-5 h-5 text-blue-300" />
                      世界地図での位置
                    </h3>
                    <InteractiveWorldMap 
                      countryName={country.name}
                      countryCode={country.code}
                      coordinates={country.coordinates}
                    />
                  </div>
                  
                  {/* Right: Basic Info */}
                  <div>
                    <h3 className="text-xl font-bold text-white mb-6">基本情報</h3>
                    <div className="space-y-4">
                      <div className="flex items-center gap-3 bg-white/10 rounded-lg p-4">
                        <MapPin className="w-5 h-5 text-purple-300 flex-shrink-0" />
                        <div>
                          <div className="text-sm text-white/70">首都</div>
                          <div className="font-semibold text-white text-lg">{country.basic.capital}</div>
                        </div>
                      </div>
                      <div className="flex items-center gap-3 bg-white/10 rounded-lg p-4">
                        <Users className="w-5 h-5 text-purple-300 flex-shrink-0" />
                        <div>
                          <div className="text-sm text-white/70">人口</div>
                          <div className="font-semibold text-white text-lg">{country.basic.population}</div>
                        </div>
                      </div>
                      <div className="flex items-center gap-3 bg-white/10 rounded-lg p-4">
                        <Globe className="w-5 h-5 text-purple-300 flex-shrink-0" />
                        <div>
                          <div className="text-sm text-white/70">言語</div>
                          <div className="font-semibold text-white text-lg">{country.basic.language}</div>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
                
                {/* Description */}
                <div className="mt-8 pt-8 border-t border-white/10">
                  <p className="text-lg text-white/90 leading-relaxed">
                    {country.description}
                  </p>
                </div>
              </div>

              {/* Highlights */}
              <div className="bg-white/10 backdrop-blur-md rounded-3xl p-8 border border-white/20 shadow-2xl">
                <h3 className="text-xl font-bold text-white mb-6 flex items-center gap-2">
                  <Sparkles className="w-5 h-5 text-purple-300" />
                  {country.name}の魅力
                </h3>
                <div className="grid md:grid-cols-2 gap-4">
                  {country.highlights.map((highlight: any, index: number) => (
                    <div key={index} className="bg-white/10 backdrop-blur-sm rounded-xl border border-white/20 overflow-hidden hover:bg-white/15 transition-all duration-300">
                      <div className="flex">
                        {/* 左側：画像 */}
                        <div className="w-40 h-32 flex-shrink-0 bg-white/10">
                          <img 
                            src={getHighlightImage(highlight.title, countryCode)} 
                            alt={highlight.title}
                            className="w-full h-full object-cover"
                            crossOrigin="anonymous"
                            onError={(e) => {
                              const target = e.target as HTMLImageElement;
                              target.src = 'https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=400&h=300&fit=crop&q=80';
                            }}
                          />
                        </div>
                        {/* 右側：テキスト */}
                        <div className="flex-1 p-4">
                          <h4 className="font-bold text-white mb-2">{highlight.title}</h4>
                          <p className="text-white/80 text-sm leading-relaxed">{highlight.description}</p>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Call to Action */}
              <div className="text-center">
                <Link
                  to="/"
                  className="inline-flex items-center gap-3 px-12 py-4 bg-gradient-to-r from-purple-500 to-pink-500 text-white rounded-full font-bold text-lg hover:from-purple-600 hover:to-pink-600 transition-all duration-300 shadow-2xl hover:shadow-3xl transform hover:scale-105 backdrop-blur-sm border border-white/30"
                >
                  <Sparkles className="w-6 h-6" />
                  自分も診断してみる
                </Link>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}