import React from 'react';
import { useParams, Link } from 'react-router-dom';
import { ArrowLeft, MapPin, Users, Globe, Sparkles, Camera, Utensils, Heart, Plane, Star } from 'lucide-react';
import AdBanner from './AdBanner';
import CountryImage from './CountryImage';
import ImageGallery from './ImageGallery';
import CountryFlag from './CountryFlag';
import CountryMap from './CountryMap';
import { ADS_CONFIG } from '../config/ads';

// 国別データ（簡略版 - 主要3カ国のみ）
const countryData: Record<string, any> = {
  'korea': {
    name: '韓国',
    nameEn: 'South Korea',
    flag: '🇰🇷',
    code: 'kr',
    type: 'major',
    basic: {
      capital: 'ソウル',
      population: '5,100万人',
      language: '韓国語',
      continent: 'アジア'
    },
    beautyTrends: {
      title: '韓国の最新美容トレンド',
      content: 'K-POPとK-ドラマの影響で、韓国の美容文化は世界中に広まっています。特に「글래스 스킨（グラス肌）」と呼ばれる透明感のある肌作りが人気。',
      trends: [
        { name: 'グラス肌', description: '陶器のように滑らかで透明感のある肌。多段階スキンケアが基本' },
        { name: '韓国式眉毛', description: 'ストレート眉で自然な印象を演出。アーチは控えめに' },
        { name: 'グラデーションリップ', description: '唇の内側から外側に向かって色を薄くするテクニック' },
        { name: 'クッションファンデ', description: '韓国発祥の革新的なファンデーション。ツヤ肌を演出' }
      ]
    },
    attractions: {
      title: '必見観光スポット',
      spots: [
        { name: 'ソウル', description: '現代的な首都' },
        { name: '釜山', description: '港町とビーチ' },
        { name: '済州島', description: '自然豊かなリゾート' },
        { name: '慶州', description: '歴史的な古都' },
        { name: '明洞', description: 'ショッピングの中心地' }
      ]
    }
  },
  'finland': {
    name: 'フィンランド',
    nameEn: 'Finland',
    flag: '🇫🇮',
    code: 'fi',
    type: 'medium',
    basic: {
      capital: 'ヘルシンキ',
      population: '550万人',
      language: 'フィンランド語・スウェーデン語',
      continent: 'ヨーロッパ（北欧）',
      location: '北欧の国。ロシア、スウェーデン、ノルウェーと国境を接する'
    },
    charm: {
      title: 'フィンランドの知られざる魅力',
      content: '「森と湖の国」と呼ばれるフィンランドは、サウナ発祥の地として有名。実は世界一幸福度の高い国の常連でもあります。',
      points: ['オーロラ観察の穴場スポット', 'ムーミンの故郷', 'サウナ文化の本場', '世界最高の教育システム', 'デザイン大国（マリメッコ、イッタラ等）']
    },
    attractions: {
      title: '必見観光スポット',
      spots: [
        { name: 'ヘルシンキ大聖堂', description: '白い美しい建物が印象的なヘルシンキのシンボル' },
        { name: 'サンタクロース村', description: 'ラップランドにある本物のサンタの村' },
        { name: 'スオメンリンナ要塞', description: '世界遺産の海上要塞。ヘルシンキからフェリーで行ける' },
        { name: 'ラップランド', description: '冬はオーロラ、夏は白夜が楽しめる神秘的な土地' },
        { name: '湖水地方', description: '18万8千の湖がある美しい自然スポット' }
      ]
    }
  },
  'bhutan': {
    name: 'ブータン',
    nameEn: 'Bhutan',
    flag: '🇧🇹',
    code: 'bt',
    type: 'minor',
    basic: {
      capital: 'ティンプー',
      population: '77万人',
      language: 'ゾンカ語',
      continent: 'アジア（ヒマラヤ）',
      location: '中国とインドに囲まれたヒマラヤの小国'
    },
    introduction: {
      title: 'ブータンってどんな国？',
      content: '「最後のシャングリラ」と呼ばれるヒマラヤの小さな王国。GDP（国内総生産）ではなくGNH（国民総幸福量）を重視する世界で唯一の国として有名です。',
      features: ['世界で唯一の炭素マイナス国', '国民の97%が幸せと答える国', '森林面積が国土の70%以上', 'テレビ放送開始は1999年', '国民全員が民族衣装を着用']
    },
    uniqueExperiences: {
      title: 'ブータンでしか体験できないこと',
      experiences: [
        { name: 'タイガーズネスト僧院', description: '断崖絶壁に建つ幻想的な僧院。ブータン最大の聖地' },
        { name: 'ツェチュ祭り', description: '仮面舞踊などの伝統芸能が披露される宗教祭り' },
        { name: 'ヒマラヤトレッキング', description: '手つかずの自然の中をゆっくりと歩く贅沢な時間' },
        { name: '伝統的なファームステイ', description: '農家に泊まって現地の生活を体験' }
      ]
    }
  }
};

export default function CountryDetailImproved() {
  const { countryCode } = useParams<{ countryCode: string }>();
  const country = countryData[countryCode?.toLowerCase() || ''];

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

  const Section: React.FC<{ title: string; icon: React.ReactNode; imageKeyword?: string; children: React.ReactNode }> = 
    ({ title, icon, imageKeyword, children }) => (
    <div className="bg-white/80 backdrop-blur-sm rounded-3xl shadow-xl overflow-hidden mb-8">
      <div className="grid lg:grid-cols-2 gap-0">
        {/* 左側：画像 */}
        <div className="order-2 lg:order-1">
          {imageKeyword ? (
            <CountryImage 
              keyword={imageKeyword}
              alt={`${country.name} ${title}`}
              size="large"
              className="h-full min-h-[300px] rounded-none"
            />
          ) : (
            <ImageGallery 
              countryCode={countryCode || ''}
              countryName={country.name}
              imageCount={1}
              className="h-full min-h-[300px]"
            />
          )}
        </div>
        
        {/* 右側：コンテンツ */}
        <div className="order-1 lg:order-2 p-8">
          <h2 className="text-2xl font-bold text-gray-800 mb-6 flex items-center gap-2">
            {icon}
            {title}
          </h2>
          {children}
        </div>
      </div>
    </div>
  );

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-50 via-pink-50 to-blue-50 py-8 px-4">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <Link
            to="/"
            className="inline-flex items-center gap-2 px-4 py-2 bg-white/80 backdrop-blur-sm rounded-full shadow-lg hover:shadow-xl transition-all duration-300 mb-6"
          >
            <ArrowLeft className="w-4 h-4" />
            戻る
          </Link>
          
          <div className="text-center mb-8">
            <div className="flex items-center justify-center gap-4 mb-4">
              <div className="text-6xl">{country.flag}</div>
              <CountryFlag 
                countryCode={country.code}
                countryName={country.name}
                size="large"
              />
            </div>
            <h1 className="text-4xl md:text-5xl font-bold text-gray-800 mb-2">
              {country.name}
            </h1>
            <p className="text-xl text-gray-600">{country.nameEn}</p>
          </div>
        </div>

        {/* Basic Info with Map */}
        <div className="grid lg:grid-cols-3 gap-6 mb-8">
          <div className="lg:col-span-2">
            <div className="bg-white/80 backdrop-blur-sm rounded-3xl shadow-xl p-8">
              <h2 className="text-2xl font-bold text-gray-800 mb-6 flex items-center gap-2">
                <Globe className="w-6 h-6 text-purple-600" />
                基本情報
              </h2>
              <div className="grid md:grid-cols-2 gap-4">
                <div className="flex items-center gap-3">
                  <MapPin className="w-5 h-5 text-gray-500" />
                  <span className="text-gray-700">首都: {country.basic.capital}</span>
                </div>
                <div className="flex items-center gap-3">
                  <Users className="w-5 h-5 text-gray-500" />
                  <span className="text-gray-700">人口: {country.basic.population}</span>
                </div>
                <div className="flex items-center gap-3">
                  <Globe className="w-5 h-5 text-gray-500" />
                  <span className="text-gray-700">言語: {country.basic.language}</span>
                </div>
                <div className="flex items-center gap-3">
                  <Globe className="w-5 h-5 text-gray-500" />
                  <span className="text-gray-700">大陸: {country.basic.continent}</span>
                </div>
              </div>
            </div>
          </div>
          
          {/* Map */}
          <div>
            <CountryMap 
              countryName={country.name}
              countryCode={country.code}
              className="h-full min-h-[200px]"
            />
          </div>
        </div>

        {/* Content Sections */}
        {country.type === 'major' && country.beautyTrends && (
          <Section 
            title={country.beautyTrends.title}
            icon={<Sparkles className="w-6 h-6 text-purple-600" />}
            imageKeyword="korean beauty skincare"
          >
            <p className="text-gray-700 mb-6">{country.beautyTrends.content}</p>
            <div className="space-y-3">
              {country.beautyTrends.trends.map((trend: any, index: number) => (
                <div key={index} className="bg-purple-50 rounded-lg p-4">
                  <h3 className="font-semibold text-purple-800 mb-2">{trend.name}</h3>
                  <p className="text-sm text-gray-700">{trend.description}</p>
                </div>
              ))}
            </div>
          </Section>
        )}

        {country.type === 'medium' && country.charm && (
          <Section 
            title={country.charm.title}
            icon={<Heart className="w-6 h-6 text-purple-600" />}
            imageKeyword="finland aurora landscape"
          >
            <p className="text-gray-700 mb-6">{country.charm.content}</p>
            <div className="flex flex-wrap gap-2">
              {country.charm.points.map((point: string, index: number) => (
                <span key={index} className="px-3 py-1 bg-blue-100 text-blue-700 rounded-full text-sm">
                  {point}
                </span>
              ))}
            </div>
          </Section>
        )}

        {country.type === 'minor' && country.introduction && (
          <Section 
            title={country.introduction.title}
            icon={<Globe className="w-6 h-6 text-purple-600" />}
            imageKeyword="bhutan himalaya monastery"
          >
            <p className="text-gray-700 mb-6">{country.introduction.content}</p>
            <div className="space-y-2">
              {country.introduction.features.map((feature: string, index: number) => (
                <div key={index} className="flex items-start gap-2">
                  <span className="text-blue-600">🌟</span>
                  <span className="text-gray-700">{feature}</span>
                </div>
              ))}
            </div>
          </Section>
        )}

        {/* Attractions */}
        {country.attractions && (
          <Section 
            title={country.attractions.title}
            icon={<Camera className="w-6 h-6 text-purple-600" />}
          >
            <div className="space-y-3">
              {country.attractions.spots.map((spot: any, index: number) => (
                <div key={index} className="flex items-start gap-3">
                  <span className="text-purple-600 font-bold">{index + 1}.</span>
                  <div>
                    <h3 className="font-semibold text-gray-800">{spot.name}</h3>
                    <p className="text-sm text-gray-600">{spot.description}</p>
                  </div>
                </div>
              ))}
            </div>
          </Section>
        )}

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
        <div className="text-center mb-8">
          <Link
            to="/"
            className="inline-flex items-center gap-2 px-8 py-4 bg-gradient-to-r from-purple-600 to-pink-600 text-white rounded-full font-bold text-lg hover:from-purple-700 hover:to-pink-700 transition-all duration-300 shadow-lg hover:shadow-xl"
          >
            <Sparkles className="w-5 h-5" />
            自分も診断してみる
          </Link>
        </div>
      </div>
    </div>
  );
}