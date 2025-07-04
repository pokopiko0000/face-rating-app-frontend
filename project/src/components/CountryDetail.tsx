import React from 'react';
import { useParams, Link } from 'react-router-dom';
import { ArrowLeft, MapPin, Users, Globe, Sparkles, Camera, Utensils, Heart, Plane, Star } from 'lucide-react';
import AdBanner from './AdBanner';
import CountryImage from './CountryImage';
import ImageGallery from './ImageGallery';
import { ADS_CONFIG } from '../config/ads';

// 国別データ
const countryData: Record<string, any> = {
  'korea': {
    name: '韓国',
    nameEn: 'South Korea',
    flag: '🇰🇷',
    type: 'major', // major, medium, minor
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
        {
          name: 'グラス肌',
          description: '陶器のように滑らかで透明感のある肌。多段階スキンケアが基本'
        },
        {
          name: '韓国式眉毛',
          description: 'ストレート眉で自然な印象を演出。アーチは控えめに'
        },
        {
          name: 'グラデーションリップ',
          description: '唇の内側から外側に向かって色を薄くするテクニック'
        },
        {
          name: 'クッションファンデ',
          description: '韓国発祥の革新的なファンデーション。ツヤ肌を演出'
        }
      ]
    },
    regionalDifferences: {
      title: '地域別の特色',
      regions: [
        {
          name: 'ソウル',
          description: '最新トレンドの発信地。江南エリアは特に美容意識が高い'
        },
        {
          name: '釜山',
          description: '海沿いの都市らしく、日焼け止めやスキンケアへの意識が高い'
        },
        {
          name: '済州島',
          description: '自然派化粧品の原料となる植物が豊富。オーガニック美容が人気'
        }
      ]
    },
    hiddenSpots: {
      title: '現地人おすすめスポット',
      spots: [
        {
          name: '弘大の美容室街',
          description: 'K-POPアイドルも通う隠れた名店が集まるエリア'
        },
        {
          name: '清潭洞の美容皮膚科',
          description: '芸能人御用達の美容クリニックが密集'
        },
        {
          name: '東大門の夜市場',
          description: '深夜まで営業する美容グッズ専門店が豊富'
        }
      ]
    },
    surprisingFacts: [
      '韓国は世界で最も美容整形が一般的な国の一つ',
      'スキンケアステップは平均10段階以上',
      '「リップティント」は韓国で生まれた美容用語',
      '男性の化粧品使用率が世界トップクラス'
    ]
  },
  
  'finland': {
    name: 'フィンランド',
    nameEn: 'Finland',
    flag: '🇫🇮',
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
      points: [
        'オーロラ観察の穴場スポット',
        'ムーミンの故郷',
        'サウナ文化の本場',
        '世界最高の教育システム',
        'デザイン大国（マリメッコ、イッタラ等）'
      ]
    },
    beautyStandards: {
      title: 'フィンランドの美の基準',
      content: '北欧らしい自然体の美しさを重視。環境への配慮から、ナチュラルでサステナブルな美容法が好まれます。',
      styles: ['ナチュラルメイク', 'エコ美容', 'サウナ美容法', 'ミニマリストファッション']
    },
    attractions: {
      title: '必見観光スポット',
      spots: [
        {
          name: 'ヘルシンキ大聖堂',
          description: '白い美しい建物が印象的なヘルシンキのシンボル'
        },
        {
          name: 'サンタクロース村',
          description: 'ラップランドにある本物のサンタの村'
        },
        {
          name: 'スオメンリンナ要塞',
          description: '世界遺産の海上要塞。ヘルシンキからフェリーで行ける'
        },
        {
          name: 'ラップランド',
          description: '冬はオーロラ、夏は白夜が楽しめる神秘的な土地'
        },
        {
          name: '湖水地方',
          description: '18万8千の湖がある美しい自然スポット'
        }
      ]
    },
    cuisine: {
      title: 'フィンランドグルメ',
      dishes: [
        { name: 'サーモンスープ', description: '国民的料理の温かいスープ' },
        { name: 'カレリアパイ', description: '米のお粥を包んだ伝統的なパイ' },
        { name: 'トナカイ肉', description: 'ラップランド地方の名物' },
        { name: 'ベリー類', description: 'ブルーベリーやクラウドベリーが豊富' }
      ]
    },
    travelTips: {
      title: '旅行お役立ち情報',
      tips: [
        '夏の白夜シーズン（6-7月）がおすすめ',
        '冬のオーロラシーズンは9月-3月',
        '英語がよく通じる',
        'サウナ体験は必須',
        'キャッシュレス決済が普及'
      ]
    }
  },

  'bhutan': {
    name: 'ブータン',
    nameEn: 'Bhutan',
    flag: '🇧🇹',
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
      features: [
        '世界で唯一の炭素マイナス国',
        '国民の97%が幸せと答える国',
        '森林面積が国土の70%以上',
        'テレビ放送開始は1999年',
        '国民全員が民族衣装を着用'
      ]
    },
    surprisingCulture: {
      title: '驚きの文化・習慣',
      items: [
        {
          title: '幸福度を数値化',
          description: '政府が国民の幸福度を定期的に調査し、政策に反映'
        },
        {
          title: '全国民が民族衣装',
          description: '男性は「ゴ」、女性は「キラ」という民族衣装を日常的に着用'
        },
        {
          title: '観光税制度',
          description: '1日200ドルの観光税により、質の高い観光を維持'
        },
        {
          title: '信号機がない',
          description: '首都でも信号機はなく、警察官が手信号で交通整理'
        }
      ]
    },
    uniqueExperiences: {
      title: 'ブータンでしか体験できないこと',
      experiences: [
        {
          name: 'タイガーズネスト僧院',
          description: '断崖絶壁に建つ幻想的な僧院。ブータン最大の聖地'
        },
        {
          name: 'ツェチュ祭り',
          description: '仮面舞踊などの伝統芸能が披露される宗教祭り'
        },
        {
          name: 'ヒマラヤトレッキング',
          description: '手つかずの自然の中をゆっくりと歩く贅沢な時間'
        },
        {
          name: '伝統的なファームステイ',
          description: '農家に泊まって現地の生活を体験'
        }
      ]
    },
    whyVisit: {
      title: 'なぜブータンに行くべきか',
      reasons: [
        '忙しい現代社会で忘れがちな「幸せの本質」を考えられる',
        'デジタルデトックスで心の平穏を取り戻せる',
        '手つかずの大自然とヒマラヤの絶景',
        '他では体験できない独特の仏教文化',
        '物質的豊かさとは違う価値観に触れられる'
      ]
    },
    accessInfo: {
      title: '日本からのアクセス',
      info: [
        'バンコクまたはデリー経由が一般的',
        'ドゥルク航空が唯一のブータン国営航空',
        '入国にはビザが必要',
        '旅行会社を通したツアー参加が基本',
        '所要時間：約10-15時間（乗継込み）'
      ]
    }
  }
};

export default function CountryDetail() {
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

  const renderMajorCountry = () => (
    <>
      {/* Basic Info - Compact */}
      <div className="bg-white/80 backdrop-blur-sm rounded-2xl shadow-lg p-4 mb-6">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
          <div><span className="text-gray-500">首都:</span> {country.basic.capital}</div>
          <div><span className="text-gray-500">人口:</span> {country.basic.population}</div>
          <div><span className="text-gray-500">言語:</span> {country.basic.language}</div>
          <div><span className="text-gray-500">大陸:</span> {country.basic.continent}</div>
        </div>
      </div>

      {/* Beauty Trends - Detailed */}
      <div className="bg-white/80 backdrop-blur-sm rounded-3xl shadow-xl p-8 mb-8">
        <h2 className="text-2xl font-bold text-gray-800 mb-6 flex items-center gap-2">
          <Sparkles className="w-6 h-6 text-purple-600" />
          {country.beautyTrends.title}
        </h2>
        
        {/* Hero Image */}
        <div className="mb-6">
          <CountryImage 
            keyword="korean beauty skincare"
            alt="Korean beauty and skincare"
            size="large"
            className="mb-4"
          />
        </div>
        
        <p className="text-gray-700 mb-6">{country.beautyTrends.content}</p>
        <div className="grid md:grid-cols-2 gap-4">
          {country.beautyTrends.trends.map((trend: any, index: number) => (
            <div key={index} className="bg-purple-50 rounded-xl p-4">
              <h3 className="font-semibold text-purple-800 mb-2">{trend.name}</h3>
              <p className="text-sm text-gray-700">{trend.description}</p>
            </div>
          ))}
        </div>
      </div>

      {/* Regional Differences */}
      <div className="bg-white/80 backdrop-blur-sm rounded-3xl shadow-xl p-8 mb-8">
        <h2 className="text-2xl font-bold text-gray-800 mb-6 flex items-center gap-2">
          <MapPin className="w-6 h-6 text-purple-600" />
          {country.regionalDifferences.title}
        </h2>
        <div className="space-y-4">
          {country.regionalDifferences.regions.map((region: any, index: number) => (
            <div key={index} className="border-l-4 border-purple-300 pl-4">
              <h3 className="font-semibold text-gray-800">{region.name}</h3>
              <p className="text-gray-600">{region.description}</p>
            </div>
          ))}
        </div>
      </div>

      {/* Hidden Spots */}
      <div className="bg-white/80 backdrop-blur-sm rounded-3xl shadow-xl p-8 mb-8">
        <h2 className="text-2xl font-bold text-gray-800 mb-6 flex items-center gap-2">
          <Star className="w-6 h-6 text-purple-600" />
          {country.hiddenSpots.title}
        </h2>
        <div className="space-y-3">
          {country.hiddenSpots.spots.map((spot: any, index: number) => (
            <div key={index} className="flex items-start gap-3">
              <span className="text-purple-600 font-bold">{index + 1}.</span>
              <div>
                <h3 className="font-semibold text-gray-800">{spot.name}</h3>
                <p className="text-sm text-gray-600">{spot.description}</p>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Surprising Facts */}
      <div className="bg-white/80 backdrop-blur-sm rounded-3xl shadow-xl p-8 mb-8">
        <h2 className="text-2xl font-bold text-gray-800 mb-6">意外な事実</h2>
        <ul className="space-y-2">
          {country.surprisingFacts.map((fact: string, index: number) => (
            <li key={index} className="flex items-start gap-2">
              <span className="text-purple-600">•</span>
              <span className="text-gray-700">{fact}</span>
            </li>
          ))}
        </ul>
      </div>
    </>
  );

  const renderMediumCountry = () => (
    <>
      {/* Basic Info */}
      <div className="bg-white/80 backdrop-blur-sm rounded-3xl shadow-xl p-8 mb-8">
        <h2 className="text-2xl font-bold text-gray-800 mb-6 flex items-center gap-2">
          <Globe className="w-6 h-6 text-purple-600" />
          基本情報
        </h2>
        <div className="grid md:grid-cols-2 gap-4 mb-4">
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
            <span className="text-gray-700">場所: {country.basic.location}</span>
          </div>
        </div>
      </div>

      {/* Charm */}
      <div className="bg-white/80 backdrop-blur-sm rounded-3xl shadow-xl p-8 mb-8">
        <h2 className="text-2xl font-bold text-gray-800 mb-6 flex items-center gap-2">
          <Heart className="w-6 h-6 text-purple-600" />
          {country.charm.title}
        </h2>
        <p className="text-gray-700 mb-6">{country.charm.content}</p>
        <div className="flex flex-wrap gap-2">
          {country.charm.points.map((point: string, index: number) => (
            <span key={index} className="px-3 py-1 bg-blue-100 text-blue-700 rounded-full text-sm">
              {point}
            </span>
          ))}
        </div>
      </div>

      {/* Beauty Standards */}
      <div className="bg-white/80 backdrop-blur-sm rounded-3xl shadow-xl p-8 mb-8">
        <h2 className="text-2xl font-bold text-gray-800 mb-6 flex items-center gap-2">
          <Sparkles className="w-6 h-6 text-purple-600" />
          {country.beautyStandards.title}
        </h2>
        <p className="text-gray-700 mb-6">{country.beautyStandards.content}</p>
        <div className="flex flex-wrap gap-2">
          {country.beautyStandards.styles.map((style: string, index: number) => (
            <span key={index} className="px-3 py-1 bg-purple-100 text-purple-700 rounded-full text-sm">
              {style}
            </span>
          ))}
        </div>
      </div>

      {/* Attractions */}
      <div className="bg-white/80 backdrop-blur-sm rounded-3xl shadow-xl p-8 mb-8">
        <h2 className="text-2xl font-bold text-gray-800 mb-6 flex items-center gap-2">
          <Camera className="w-6 h-6 text-purple-600" />
          {country.attractions.title}
        </h2>
        
        {/* Image Gallery */}
        <div className="mb-6">
          <ImageGallery 
            countryCode={countryCode}
            countryName={country.name}
            imageCount={3}
            className="mb-6"
          />
        </div>
        
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
      </div>

      {/* Cuisine */}
      <div className="bg-white/80 backdrop-blur-sm rounded-3xl shadow-xl p-8 mb-8">
        <h2 className="text-2xl font-bold text-gray-800 mb-6 flex items-center gap-2">
          <Utensils className="w-6 h-6 text-purple-600" />
          {country.cuisine.title}
        </h2>
        <div className="grid md:grid-cols-2 gap-4">
          {country.cuisine.dishes.map((dish: any, index: number) => (
            <div key={index} className="bg-orange-50 rounded-lg p-3">
              <h3 className="font-semibold text-orange-800">{dish.name}</h3>
              <p className="text-sm text-gray-700">{dish.description}</p>
            </div>
          ))}
        </div>
      </div>

      {/* Travel Tips */}
      <div className="bg-white/80 backdrop-blur-sm rounded-3xl shadow-xl p-8 mb-8">
        <h2 className="text-2xl font-bold text-gray-800 mb-6 flex items-center gap-2">
          <Plane className="w-6 h-6 text-purple-600" />
          {country.travelTips.title}
        </h2>
        <ul className="space-y-2">
          {country.travelTips.tips.map((tip: string, index: number) => (
            <li key={index} className="flex items-start gap-2">
              <span className="text-green-600">✓</span>
              <span className="text-gray-700">{tip}</span>
            </li>
          ))}
        </ul>
      </div>
    </>
  );

  const renderMinorCountry = () => (
    <>
      {/* Where is it? */}
      <div className="bg-white/80 backdrop-blur-sm rounded-3xl shadow-xl p-8 mb-8">
        <h2 className="text-2xl font-bold text-gray-800 mb-6 flex items-center gap-2">
          <MapPin className="w-6 h-6 text-purple-600" />
          どこにある？
        </h2>
        <div className="bg-blue-50 rounded-lg p-4 mb-4">
          <p className="text-gray-700 font-medium">{country.basic.location}</p>
        </div>
        <div className="grid md:grid-cols-2 gap-4">
          <div><span className="text-gray-500">首都:</span> {country.basic.capital}</div>
          <div><span className="text-gray-500">人口:</span> {country.basic.population}</div>
          <div><span className="text-gray-500">言語:</span> {country.basic.language}</div>
          <div><span className="text-gray-500">大陸:</span> {country.basic.continent}</div>
        </div>
      </div>

      {/* Introduction */}
      <div className="bg-white/80 backdrop-blur-sm rounded-3xl shadow-xl p-8 mb-8">
        <h2 className="text-2xl font-bold text-gray-800 mb-6 flex items-center gap-2">
          <Globe className="w-6 h-6 text-purple-600" />
          {country.introduction.title}
        </h2>
        <p className="text-gray-700 mb-6">{country.introduction.content}</p>
        <div className="space-y-2">
          {country.introduction.features.map((feature: string, index: number) => (
            <div key={index} className="flex items-start gap-2">
              <span className="text-blue-600">🌟</span>
              <span className="text-gray-700">{feature}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Surprising Culture */}
      <div className="bg-white/80 backdrop-blur-sm rounded-3xl shadow-xl p-8 mb-8">
        <h2 className="text-2xl font-bold text-gray-800 mb-6 flex items-center gap-2">
          <Star className="w-6 h-6 text-purple-600" />
          {country.surprisingCulture.title}
        </h2>
        <div className="space-y-4">
          {country.surprisingCulture.items.map((item: any, index: number) => (
            <div key={index} className="bg-yellow-50 rounded-lg p-4">
              <h3 className="font-semibold text-yellow-800 mb-2">{item.title}</h3>
              <p className="text-gray-700">{item.description}</p>
            </div>
          ))}
        </div>
      </div>

      {/* Unique Experiences */}
      <div className="bg-white/80 backdrop-blur-sm rounded-3xl shadow-xl p-8 mb-8">
        <h2 className="text-2xl font-bold text-gray-800 mb-6 flex items-center gap-2">
          <Camera className="w-6 h-6 text-purple-600" />
          {country.uniqueExperiences.title}
        </h2>
        
        {/* Hero Image */}
        <div className="mb-6">
          <CountryImage 
            keyword="tigers nest monastery bhutan"
            alt="Tigers Nest Monastery, Bhutan"
            size="large"
            className="mb-4"
          />
        </div>
        
        <div className="grid md:grid-cols-2 gap-4">
          {country.uniqueExperiences.experiences.map((exp: any, index: number) => (
            <div key={index} className="bg-green-50 rounded-lg p-4">
              <h3 className="font-semibold text-green-800 mb-2">{exp.name}</h3>
              <p className="text-sm text-gray-700">{exp.description}</p>
            </div>
          ))}
        </div>
      </div>

      {/* Why Visit */}
      <div className="bg-white/80 backdrop-blur-sm rounded-3xl shadow-xl p-8 mb-8">
        <h2 className="text-2xl font-bold text-gray-800 mb-6 flex items-center gap-2">
          <Heart className="w-6 h-6 text-purple-600" />
          {country.whyVisit.title}
        </h2>
        <ul className="space-y-3">
          {country.whyVisit.reasons.map((reason: string, index: number) => (
            <li key={index} className="flex items-start gap-3">
              <span className="text-red-500 text-lg">♥</span>
              <span className="text-gray-700">{reason}</span>
            </li>
          ))}
        </ul>
      </div>

      {/* Access Info */}
      <div className="bg-white/80 backdrop-blur-sm rounded-3xl shadow-xl p-8 mb-8">
        <h2 className="text-2xl font-bold text-gray-800 mb-6 flex items-center gap-2">
          <Plane className="w-6 h-6 text-purple-600" />
          {country.accessInfo.title}
        </h2>
        <ul className="space-y-2">
          {country.accessInfo.info.map((info: string, index: number) => (
            <li key={index} className="flex items-start gap-2">
              <span className="text-blue-600">✈️</span>
              <span className="text-gray-700">{info}</span>
            </li>
          ))}
        </ul>
      </div>
    </>
  );

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-50 via-pink-50 to-blue-50 py-8 px-4">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <Link
            to="/"
            className="inline-flex items-center gap-2 px-4 py-2 bg-white/80 backdrop-blur-sm rounded-full shadow-lg hover:shadow-xl transition-all duration-300 mb-6"
          >
            <ArrowLeft className="w-4 h-4" />
            戻る
          </Link>
          
          <div className="text-center">
            <div className="text-6xl mb-4">{country.flag}</div>
            <h1 className="text-4xl md:text-5xl font-bold text-gray-800 mb-2">
              {country.name}
            </h1>
            <p className="text-xl text-gray-600">{country.nameEn}</p>
          </div>
        </div>

        {/* Content based on country type */}
        {country.type === 'major' && renderMajorCountry()}
        {country.type === 'medium' && renderMediumCountry()}
        {country.type === 'minor' && renderMinorCountry()}

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