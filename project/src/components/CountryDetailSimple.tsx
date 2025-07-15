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
  },

  // 新規追加: 50カ国の特設ページ
  'afghanistan': {
    name: 'アフガニスタン',
    nameEn: 'Afghanistan',
    flag: '🇦🇫',
    code: 'af',
    basic: { capital: 'カブール', population: '3,900万人', language: 'ダリ語・パシュトー語' },
    coordinates: { lat: 34.5553, lng: 69.2075 },
    description: 'シルクロードの要衝として栄えた歴史ある国。バーミヤン石仏やヒンドゥークシュ山脈の雄大な自然が魅力的です。',
    highlights: [
      { title: 'バーミヤン渓谷の古代遺跡', description: 'シルクロード時代の仏教石窟群。ユネスコ世界遺産に登録された歴史的価値の高い遺跡群です。' },
      { title: 'ヒンドゥークシュ山脈の絶景', description: '7,000m級の峰々が連なる壮大な山脈。トレッキングや登山愛好家の憧れの地です。' },
      { title: '豊かなペルシャ文化', description: 'ペルシャ帝国の影響を受けた独特の文化と芸術。美しい絨毯や工芸品で知られています。' },
      { title: '伝統的なバザール文化', description: 'カブールやヘラートの賑やかなバザール。スパイスや手工芸品が並ぶ活気ある市場です。' }
    ],
    whyVisit: 'シルクロードの歴史とペルシャ文化を感じられる神秘的な国'
  },

  'albania': {
    name: 'アルバニア',
    nameEn: 'Albania',
    flag: '🇦🇱',
    code: 'al',
    basic: { capital: 'ティラナ', population: '290万人', language: 'アルバニア語' },
    coordinates: { lat: 41.3275, lng: 19.8187 },
    description: 'バルカン半島の隠れた宝石。アドリア海とイオニア海に面し、美しいビーチと山々が共存する多様な自然が魅力です。',
    highlights: [
      { title: 'アルバニアンリビエラの絶景', description: 'エメラルドグリーンの海と白い砂浜。まだ観光地化されていない手つかずのビーチリゾートです。' },
      { title: 'ベラト世界遺産の街', description: '「千の窓の街」と呼ばれる美しい古都。オスマン帝国時代の建築が保存された歴史的な街並みです。' },
      { title: 'アルプス山脈の大自然', description: 'アルバニアンアルプスの壮大な山岳風景。ハイキングや自然観察に最適な未開拓の山々です。' },
      { title: '多様な文化の融合', description: 'イスラム、正教会、カトリックが共存する寛容な文化。東西文明の交差点ならではの多様性です。' }
    ],
    whyVisit: 'まだ知られていないヨーロッパの秘境を発見できる国'
  },

  'algeria': {
    name: 'アルジェリア',
    nameEn: 'Algeria',
    flag: '🇩🇿',
    code: 'dz',
    basic: { capital: 'アルジェ', population: '4,400万人', language: 'アラビア語・ベルベル語' },
    coordinates: { lat: 36.7538, lng: 3.0588 },
    description: 'アフリカ最大の国土を誇る北アフリカの大国。サハラ砂漠の玄関口として、古代ローマ遺跡とイスラム文化が織りなす魅力的な国です。',
    highlights: [
      { title: 'サハラ砂漠の壮大な景観', description: '世界最大の砂漠の一部を占める圧倒的な砂丘群。星空観測や砂漠ツアーで非日常体験ができます。' },
      { title: 'ティムガッド古代ローマ遺跡', description: '保存状態抜群のローマ時代の都市遺跡。完璧な格子状の街並みが2000年前の姿を留めています。' },
      { title: 'カスバの迷宮都市アルジェ', description: 'オスマン帝国時代の城塞都市。迷路のような細い路地と白い家々が美しいユネスコ世界遺産です。' },
      { title: '地中海沿岸の美しいコースト', description: 'エメラルドブルーの地中海に面した美しい海岸線。リゾート地としても人気の地域です。' }
    ],
    whyVisit: 'サハラ砂漠と古代文明の神秘を体験できる壮大な国'
  },

  'andorra': {
    name: 'アンドラ',
    nameEn: 'Andorra',
    flag: '🇦🇩',
    code: 'ad',
    basic: { capital: 'アンドラ・ラ・ベリャ', population: '8万人', language: 'カタルーニャ語' },
    coordinates: { lat: 42.5063, lng: 1.5218 },
    description: 'ピレネー山脈に囲まれた小さな公国。フランスとスペインの共同統治下にある、スキーとショッピングの楽園です。',
    highlights: [
      { title: '世界屈指のスキーリゾート', description: 'グランバリラやソルデウなど、雪質抜群のスキー場。ヨーロッパ屈指のウィンタースポーツの聖地です。' },
      { title: 'タックスフリーショッピング', description: '消費税がない免税の買い物天国。高級ブランドから日用品まで、お得に購入できる魅力的な国です。' },
      { title: 'ピレネー山脈の大自然', description: '標高2,000m級の美しい山々に囲まれた自然豊かな環境。ハイキングや登山を楽しめます。' },
      { title: 'ロマネスク様式の教会群', description: '小さな国に数多く残る美しい石造りの教会。中世ヨーロッパの建築美を堪能できます。' }
    ],
    whyVisit: 'ヨーロッパの隠れた山岳リゾートで非日常を満喫できる国'
  },

  'angola': {
    name: 'アンゴラ',
    nameEn: 'Angola',
    flag: '🇦🇴',
    code: 'ao',
    basic: { capital: 'ルアンダ', population: '3,300万人', language: 'ポルトガル語' },
    coordinates: { lat: -8.8390, lng: 13.2894 },
    description: '南部アフリカの大西洋沿岸に位置する国。豊富な石油とダイヤモンド資源を持ち、急速に発展している新興国です。',
    highlights: [
      { title: 'カランドゥラの滝の絶景', description: 'アフリカで2番目に高い落差105mの壮大な滝。雨季には水量が増し、圧倒的な迫力を見せます。' },
      { title: 'ルアンダの近代的な街並み', description: '石油収入で急発展した首都の摩天楼。アフリカとは思えないモダンな都市景観が印象的です。' },
      { title: 'ナミブ砂漠の南端部', description: '世界最古の砂漠の一部。オレンジ色の砂丘と大西洋が出会う神秘的な風景を楽しめます。' },
      { title: 'ポルトガル植民地時代の遺産', description: 'コロニアル建築や文化が残る歴史的な街並み。アフリカとヨーロッパが融合した独特の文化です。' }
    ],
    whyVisit: 'アフリカの新たな成長の可能性と豊かな自然を感じられる国'
  },

  'argentina': {
    name: 'アルゼンチン',
    nameEn: 'Argentina',
    flag: '🇦🇷',
    code: 'ar',
    basic: { capital: 'ブエノスアイレス', population: '4,500万人', language: 'スペイン語' },
    coordinates: { lat: -34.6118, lng: -58.3960 },
    description: '南米第2の国土を誇る多様性の国。タンゴ発祥の地として知られ、パタゴニアの大自然からブエノスアイレスの洗練された都市文化まで魅力満載です。',
    highlights: [
      { title: 'イグアスの滝の圧倒的迫力', description: '世界最大級の滝群の一つ。「悪魔の喉笛」と呼ばれる大瀑布は圧巻の自然スペクタクルです。' },
      { title: 'タンゴ発祥の地ブエノスアイレス', description: '情熱的なタンゴが生まれた街。ボカ地区のカラフルな街並みと本場のタンゴショーを楽しめます。' },
      { title: 'パタゴニアの果てしない大地', description: '地球の果てと呼ばれる壮大な自然。氷河、湖、山々が織りなす手つかずの絶景が広がります。' },
      { title: '世界最高品質の牛肉文化', description: 'アルゼンチン牛とマルベックワインの本場。本格的なアサードで極上のグルメ体験ができます。' }
    ],
    whyVisit: '情熱のタンゴと壮大な自然が共存する魅力的な南米の国'
  },

  'armenia': {
    name: 'アルメニア',
    nameEn: 'Armenia',
    flag: '🇦🇲',
    code: 'am',
    basic: { capital: 'エレバン', population: '300万人', language: 'アルメニア語' },
    coordinates: { lat: 40.0691, lng: 44.5147 },
    description: '世界最古のキリスト教国として知られる歴史深い国。アララト山を望む美しい風景と古代文明の遺跡が魅力です。',
    highlights: [
      { title: '世界最古のキリスト教国', description: '301年に世界で初めてキリスト教を国教とした国。古い修道院や教会が数多く残されています。' },
      { title: 'ゲガルド修道院の神秘', description: '岩を刻んで作られた13世紀の修道院。洞窟内に響く美しい聖歌は心を浄化してくれます。' },
      { title: 'アララト山の象徴的な景観', description: 'ノアの方舟が漂着したとされる聖なる山。エレバンから見る雄大な姿は国のシンボルです。' },
      { title: '3000年の歴史を持つ文字', description: '独自のアルメニア文字は5世紀に創られた文化遺産。古代から続く豊かな文学と芸術の伝統があります。' }
    ],
    whyVisit: '古代キリスト教文明と壮大な山岳風景に触れられる神聖な国'
  },

  'australia': {
    name: 'オーストラリア',
    nameEn: 'Australia',
    flag: '🇦🇺',
    code: 'au',
    basic: { capital: 'キャンベラ', population: '2,600万人', language: '英語' },
    coordinates: { lat: -35.2820, lng: 149.1286 },
    description: '大陸全体が一つの国である広大な島国。グレートバリアリーフやウルル、シドニーオペラハウスなど世界的な観光名所が豊富です。',
    highlights: [
      { title: 'グレートバリアリーフの海洋生態系', description: '世界最大のサンゴ礁群。2,300kmに渡って広がる海洋の楽園で、ダイビングやシュノーケリングを楽しめます。' },
      { title: 'ウルル（エアーズロック）の神秘', description: '先住民アボリジニの聖地である巨大な一枚岩。夕日に染まる赤い岩肌は一生忘れられない感動を与えます。' },
      { title: 'シドニーハーバーの美景', description: 'オペラハウスとハーバーブリッジが織りなす世界で最も美しい港の風景。近代建築と自然が調和した絶景です。' },
      { title: '固有種動物との出会い', description: 'カンガルー、コアラ、ウォンバットなど、ここでしか見られない可愛い有袋類動物たちに出会えます。' }
    ],
    whyVisit: '大自然と近代都市が共存する、地球上で最もユニークな大陸'
  },

  'austria': {
    name: 'オーストリア',
    nameEn: 'Austria',
    flag: '🇦🇹',
    code: 'at',
    basic: { capital: 'ウィーン', population: '900万人', language: 'ドイツ語' },
    coordinates: { lat: 48.2082, lng: 16.3738 },
    description: '音楽の都として世界に知られる中欧の美しい国。モーツァルトやベートーベンを輩出し、ハプスブルク帝国の栄華を今に伝えます。',
    highlights: [
      { title: '音楽の都ウィーンの文化', description: 'ウィーン国立歌劇場やウィーン・フィルハーモニー管弦楽団。クラシック音楽の最高峰を体験できます。' },
      { title: 'ザルツブルクのモーツァルト', description: 'モーツァルト生誕の地。バロック建築の美しい旧市街でクラシック音楽の歴史を感じられます。' },
      { title: 'アルプスの雄大な山岳風景', description: 'ハルシュタットやザルツカンマーグートの湖水地方。絵画のような美しい山と湖の風景が広がります。' },
      { title: 'ハプスブルク帝国の栄華', description: 'シェーンブルン宮殿やホーフブルク宮殿。ヨーロッパを支配した帝国の豪華絢爛な文化遺産です。' }
    ],
    whyVisit: 'クラシック音楽と帝国の歴史、アルプスの自然美を満喫できる国'
  },

  'azerbaijan': {
    name: 'アゼルバイジャン',
    nameEn: 'Azerbaijan',
    flag: '🇦🇿',
    code: 'az',
    basic: { capital: 'バクー', population: '1,000万人', language: 'アゼルバイジャン語' },
    coordinates: { lat: 40.4093, lng: 49.8671 },
    description: 'カスピ海に面する「火の国」として知られる国。石油資源で栄え、古代ペルシャとロシア・ソビエト文化が交差する興味深い国です。',
    highlights: [
      { title: 'バクー旧市街の歴史', description: 'シルクロード時代から続く城壁都市。ペルシャ・アラブ・ロシアの文化が融合した独特の建築群です。' },
      { title: '炎の塔フレームタワーズ', description: 'バクーのランドマークとなった近代的な超高層ビル。夜間のライトアップは炎をイメージした美しい演出です。' },
      { title: 'ゴブスタン岩絵国立公園', description: '4万年前の岩絵が残る考古学的遺跡。人類最古のアートが刻まれた貴重な文化遺産です。' },
      { title: '天然ガス噴出地ヤナルダグ', description: '3000年間燃え続ける自然の炎。拝火教の聖地としても崇められてきた神秘的な自然現象です。' }
    ],
    whyVisit: '石油文明と古代文化が交錯する、エネルギッシュな火の国'
  },

  'bahamas': {
    name: 'バハマ',
    nameEn: 'Bahamas',
    flag: '🇧🇸',
    code: 'bs',
    basic: { capital: 'ナッソー', population: '40万人', language: '英語' },
    coordinates: { lat: 25.0443, lng: -77.3504 },
    description: 'カリブ海に浮かぶ700の島々からなる楽園。透明度抜群のターコイズブルーの海と白砂のビーチで有名なリゾート天国です。',
    highlights: [
      { title: 'エグズーマ諸島の泳ぐ豚', description: '世界唯一の泳ぐ豚で有名な無人島。可愛い豚たちと一緒に海で泳げるユニークな体験ができます。' },
      { title: 'ナッソーのピンクサンドビーチ', description: 'ピンク色に輝く美しい砂浜。珊瑚のかけらが混じった独特の色合いは世界でも珍しい絶景です。' },
      { title: 'アトランティス・パラダイス・リゾート', description: '巨大なウォーターパークと水族館を併設した世界的リゾート。失われた大陸をテーマにした夢の世界です。' },
      { title: 'ブルーホールの神秘', description: '深海へと続く円形の穴。ダイビングスポットとして世界的に有名な海洋の神秘を探検できます。' }
    ],
    whyVisit: 'カリブ海の究極の楽園で非日常のリゾート体験ができる国'
  },

  'bahrain': {
    name: 'バーレーン',
    nameEn: 'Bahrain',
    flag: '🇧🇭',
    code: 'bh',
    basic: { capital: 'マナーマ', population: '170万人', language: 'アラビア語' },
    coordinates: { lat: 26.0667, lng: 50.5577 },
    description: 'ペルシャ湾に浮かぶ島国で、中東の金融ハブとして発展。古代ディルムン文明の遺跡と近代的な都市が共存する国です。',
    highlights: [
      { title: 'バーレーン・フォーミュラ1', description: '砂漠の中のサーキットで開催されるF1バーレーンGP。中東で最初のF1開催国として世界的に有名です。' },
      { title: 'バーレーン国立博物館', description: '6000年の歴史を誇るディルムン文明の貴重な遺物。古代メソポタミアとインダス文明を結ぶ海洋文明の中心地でした。' },
      { title: 'マナーマの近代的スカイライン', description: 'ペルシャ湾金融センターとして発展した首都の摩天楼。伝統的なアラブ建築と超近代建築が調和した景観です。' },
      { title: '生命の木ツリー・オブ・ライフ', description: '砂漠の中で400年間生き続ける奇跡の木。水源のない乾燥地帯で緑を保つ神秘的な自然の驚異です。' }
    ],
    whyVisit: '古代文明の遺産と現代金融都市が融合した中東の小さな宝石'
  },

  'bangladesh': {
    name: 'バングラデシュ',
    nameEn: 'Bangladesh',
    flag: '🇧🇩',
    code: 'bd',
    basic: { capital: 'ダッカ', population: '1億6,500万人', language: 'ベンガル語' },
    coordinates: { lat: 23.8103, lng: 90.4125 },
    description: 'ベンガル湾に面した水と緑の国。世界最大のマングローブ林シュンドルボンと、豊かなベンガル文化で知られています。',
    highlights: [
      { title: 'シュンドルボン国立公園', description: '世界最大のマングローブ林。ロイヤルベンガルタイガーの最後の生息地として、生物多様性の宝庫です。' },
      { title: '60万のモスクの国', description: 'イスラム建築の美しいモスクが点在。600年の歴史を持つバゲルハット・モスクシティは世界遺産です。' },
      { title: 'サイクロンによる独特の地形', description: '無数の川と中州が作り出すデルタ地帯。船でしか行けない村々と水上生活の独特な文化があります。' },
      { title: 'ベンガル織物の伝統', description: 'モスリンやジャムダニなど世界最高品質の手織り織物。繊細な技術は1000年以上受け継がれています。' }
    ],
    whyVisit: '豊かな水郷文化と世界最大のマングローブ林を体験できる国'
  },

  'barbados': {
    name: 'バルバドス',
    nameEn: 'Barbados',
    flag: '🇧🇧',
    code: 'bb',
    basic: { capital: 'ブリッジタウン', population: '29万人', language: '英語' },
    coordinates: { lat: 13.1939, lng: -59.5432 },
    description: 'カリブ海東端の美しい島国。ラム酒発祥の地として知られ、英国文化とカリブ文化が融合した独特の魅力があります。',
    highlights: [
      { title: 'ラム酒発祥の地', description: 'マウントゲイ蒸留所は世界最古のラム酒ブランド。サトウキビから作られる琥珀色の極上ラムを味わえます。' },
      { title: 'ブリッジタウンの世界遺産', description: '17世紀からの英国植民地建築が残る歴史的な港町。カリブ海貿易の中心地として栄えた街並みです。' },
      { title: '東海岸の大西洋の荒波', description: 'サーフィンの世界的メッカ。強い貿易風と大西洋の大波が作り出すパーフェクトなサーフブレイクです。' },
      { title: 'フライングフィッシュ料理', description: '国の象徴でもあるトビウオを使った伝統料理。新鮮な魚介類とスパイシーなクレオール料理を楽しめます。' }
    ],
    whyVisit: 'ラム酒と美しいビーチ、英国・カリブ文化の融合を楽しめる島'
  },

  'belarus': {
    name: 'ベラルーシ',
    nameEn: 'Belarus',
    flag: '🇧🇾',
    code: 'by',
    basic: { capital: 'ミンスク', population: '950万人', language: 'ベラルーシ語・ロシア語' },
    coordinates: { lat: 53.9045, lng: 27.5615 },
    description: '東欧の内陸国で「ヨーロッパの肺」と呼ばれる森林国。原生林と湿地帯が残る自然豊かな国です。',
    highlights: [
      { title: 'ビャウォヴィエジャ原生林', description: 'ヨーロッパ最後の原生林。野生のヨーロッパバイソンが生息する1万年前からの手つかずの森です。' },
      { title: 'ミール城の中世建築', description: '16世紀の美しい要塞宮殿。ルネサンス様式とゴシック様式が融合した建築の傑作です。' },
      { title: 'ミンスクの社会主義建築', description: 'ソビエト時代の計画都市として整備された首都。壮大なスターリン建築と広大な広場が印象的です。' },
      { title: 'ナショナル・プリピャチ', description: 'チェルノブイリ原発事故の影響を受けた地域。現在は野生動物の楽園として回復している貴重なエリアです。' }
    ],
    whyVisit: 'ヨーロッパ最後の原生林と社会主義時代の歴史を体験できる国'
  },

  'belgium': {
    name: 'ベルギー',
    nameEn: 'Belgium',
    flag: '🇧🇪',
    code: 'be',
    basic: { capital: 'ブリュッセル', population: '1,150万人', language: 'オランダ語・フランス語・ドイツ語' },
    coordinates: { lat: 50.8503, lng: 4.3517 },
    description: 'ヨーロッパの心臓部に位置する小さな王国。チョコレート、ワッフル、ビールで有名で、中世の美しい街並みが残る文化国家です。',
    highlights: [
      { title: '世界最高品質のチョコレート', description: 'ゴディバやレオニダスなど世界的ブランド発祥の地。繊細で上品な味わいのプラリネは芸術品レベルです。' },
      { title: 'ブルージュの中世の街並み', description: '「北のヴェネツィア」と呼ばれる運河の街。13世紀から変わらない石畳の路地と美しい建築群です。' },
      { title: '1500種類以上のベルギービール', description: 'トラピストビールやランビックなど独特の醸造法。修道院で作られる神聖なビール文化があります。' },
      { title: 'ブリュッセルのEU本部', description: 'ヨーロッパ連合の政治的中心地。現代ヨーロッパ統合の象徴的な都市として重要な役割を果たしています。' }
    ],
    whyVisit: '美食と芸術、ヨーロッパ文化の精髄を味わえる洗練された国'
  },

  'belize': {
    name: 'ベリーズ',
    nameEn: 'Belize',
    flag: '🇧🇿',
    code: 'bz',
    basic: { capital: 'ベルモパン', population: '40万人', language: '英語' },
    coordinates: { lat: 17.2510, lng: -88.7590 },
    description: '中央アメリカの小さな英語圏の国。世界第2位のバリアリーフと古代マヤ文明の遺跡で知られる自然と歴史の宝庫です。',
    highlights: [
      { title: 'ベリーズバリアリーフ', description: '世界第2位の長さを誇るサンゴ礁。グレートブルーホールを含む海洋生態系は世界遺産です。' },
      { title: 'カラコル古代マヤ遺跡', description: 'ジャングルに眠る古代マヤ最大級の都市遺跡。カナア神殿は40mの高さでマヤ文明の壮大さを物語ります。' },
      { title: 'ジャガー保護区', description: '世界初のジャガー保護区コクスコーム。中央アメリカ最大の野生動物保護区でエコツーリズムを楽しめます。' },
      { title: '多民族文化の融合', description: 'マヤ、ガリフナ、メスティソ、クレオールの多様な文化。カリブ海文化とラテンアメリカ文化が共存しています。' }
    ],
    whyVisit: '古代マヤ文明と世界屈指のサンゴ礁を同時に楽しめる国'
  },

  'benin': {
    name: 'ベナン',
    nameEn: 'Benin',
    flag: '🇧🇯',
    code: 'bj',
    basic: { capital: 'ポルトノボ', population: '1,200万人', language: 'フランス語' },
    coordinates: { lat: 6.4969, lng: 2.6283 },
    description: '西アフリカの小国で、かつてのダホメ王国として栄えた歴史ある国。ブードゥー教発祥の地として神秘的な文化を持ちます。',
    highlights: [
      { title: 'ブードゥー教発祥の地', description: '世界中に広がったブードゥー教の起源地。神秘的な儀式と精霊信仰の本場でスピリチュアルな体験ができます。' },
      { title: 'アボメイ王宮群', description: 'ダホメ王国の宮殿群。17-19世紀の西アフリカ王朝文化を物語る貴重な歴史遺産がユネスコ世界遺産です。' },
      { title: 'ガンビエ水上村落', description: 'ノコエ湖の水上に浮かぶ村。竹馬の家々と伝統的な漁業文化が残る「アフリカのヴェネツィア」です。' },
      { title: '奴隷貿易の歴史', description: '奴隷海岸として知られた悲しい歴史。ウィダの奴隷博物館で人類の歴史を深く学ぶことができます。' }
    ],
    whyVisit: 'アフリカの精神世界と歴史の深淵に触れられる神秘的な国'
  },

  'bolivia': {
    name: 'ボリビア',
    nameEn: 'Bolivia',
    flag: '🇧🇴',
    code: 'bo',
    basic: { capital: 'スクレ', population: '1,200万人', language: 'スペイン語・ケチュア語・アイマラ語' },
    coordinates: { lat: -19.0196, lng: -65.2619 },
    description: '南米の高地に位置する内陸国。ウユニ塩湖の絶景と豊かな先住民文化、アンデス山脈の雄大な自然で知られています。',
    highlights: [
      { title: 'ウユニ塩湖の天空の鏡', description: '世界最大の塩湖が作り出す奇跡の絶景。雨季には完璧な鏡面となり、空と大地の境界が消える幻想的な体験ができます。' },
      { title: '首都ラパスの標高3600m', description: '世界最高所の首都での生活体験。アンデス山脈に囲まれた盆地にある活気ある市場と先住民文化の中心地です。' },
      { title: 'インカ帝国以前の遺跡群', description: 'ティワナク遺跡やイスカンワヤなど、インカよりも古い高度な文明の痕跡。神秘的な巨石建築の謎に迫れます。' },
      { title: 'アイマラ・ケチュア文化', description: '5000年続く先住民文化が現代も息づく国。カラフルな民族衣装と伝統的な生活様式を体験できます。' }
    ],
    whyVisit: '地球上で最も美しい絶景と古代アンデス文明の神秘を体験できる国'
  },

  'bosnia and herzegovina': {
    name: 'ボスニア・ヘルツェゴビナ',
    nameEn: 'Bosnia and Herzegovina',
    flag: '🇧🇦',
    code: 'ba',
    basic: { capital: 'サラエボ', population: '330万人', language: 'ボスニア語・クロアチア語・セルビア語' },
    coordinates: { lat: 43.8563, lng: 18.4131 },
    description: 'バルカン半島の中心に位置する多民族国家。東西文明の交差点として、オスマン帝国とヨーロッパ文化が融合した独特の魅力があります。',
    highlights: [
      { title: 'サラエボの多文化都市', description: 'イスラム、正教、カトリック、ユダヤ教が共存する稀有な都市。「ヨーロッパのエルサレム」と呼ばれる宗教的寛容の象徴です。' },
      { title: 'モスタルの古い橋', description: '16世紀オスマン朝の美しい石橋。破壊と再建の歴史を持つ平和と和解のシンボルです。' },
      { title: 'ドリナ川の美しい渓谷', description: 'エメラルドグリーンの清流が刻む深い渓谷。ラフティングや自然散策で手つかずの自然を満喫できます。' },
      { title: 'ボスニアコーヒー文化', description: 'オスマン帝国時代から続く独特のコーヒー文化。銅製のジェズベで淹れる伝統的なコーヒーでおもてなしを受けられます。' }
    ],
    whyVisit: '東西文明の交差点で多様な文化と歴史の深さを体験できる国'
  },

  'botswana': {
    name: 'ボツワナ',
    nameEn: 'Botswana',
    flag: '🇧🇼',
    code: 'bw',
    basic: { capital: 'ハボローネ', population: '240万人', language: '英語・ツワナ語' },
    coordinates: { lat: -24.6282, lng: 25.9231 },
    description: '南部アフリカの内陸国で、カラハリ砂漠とオカバンゴ・デルタの大自然で有名。アフリカ有数の政治的安定と野生動物保護で知られています。',
    highlights: [
      { title: 'オカバンゴ・デルタの奇跡', description: '砂漠の中のオアシス。世界最大の内陸デルタで、ゾウ、ライオン、カバなど豊富な野生動物を観察できます。' },
      { title: 'チョベ国立公園のゾウ', description: 'アフリカ最大のゾウの群れが生息。12万頭のゾウが河で水浴びする光景は圧巻のサファリ体験です。' },
      { title: 'カラハリ砂漠の文化', description: 'サン族（ブッシュマン）の伝統的な狩猟採集文化。数万年続く人類最古の生活様式を学ぶことができます。' },
      { title: 'マカディカディ塩湖', description: '雨季に現れる巨大な塩湖。フラミンゴの大群が飛来し、ピンク色に染まる湖面は幻想的な美しさです。' }
    ],
    whyVisit: 'アフリカ最高峰のサファリ体験と人類最古の文化に出会える国'
  },

  'brazil': {
    name: 'ブラジル',
    nameEn: 'Brazil',
    flag: '🇧🇷',
    code: 'br',
    basic: { capital: 'ブラジリア', population: '2億1,500万人', language: 'ポルトガル語' },
    coordinates: { lat: -15.8267, lng: -47.9218 },
    description: '南米最大の国で、アマゾン熱帯雨林からリオのカーニバルまで、地球上で最も多様性に富んだ国の一つです。',
    highlights: [
      { title: 'アマゾン熱帯雨林の生命力', description: '地球の肺と呼ばれる世界最大の熱帯雨林。生物多様性の宝庫で、地球上の動植物種の10%が生息しています。' },
      { title: 'リオデジャネイロのカーニバル', description: '世界最大規模のカーニバル。サンバのリズムと華やかな衣装で繰り広げられる地球上で最も情熱的な祭典です。' },
      { title: 'イグアスの滝の大迫力', description: '悪魔の喉笛と呼ばれる滝群。毎秒1,750トンの水が落下する世界最大級の滝の迫力を間近で体感できます。' },
      { title: 'ブラジリアの未来都市', description: '1960年に建設された計画都市。UFOのような近未来建築群は世界遺産に登録された都市計画の傑作です。' }
    ],
    whyVisit: '地球最大の自然と最も情熱的な文化を同時に体験できる壮大な国'
  },

  'brunei': {
    name: 'ブルネイ',
    nameEn: 'Brunei',
    flag: '🇧🇳',
    code: 'bn',
    basic: { capital: 'バンダルスリブガワン', population: '44万人', language: 'マレー語' },
    coordinates: { lat: 4.5353, lng: 114.7277 },
    description: 'ボルネオ島の小さな王国で、石油資源により世界有数の富裕国。イスラム文化と豊かな熱帯雨林が共存する国です。',
    highlights: [
      { title: 'スルタン・オマール・アリ・サイフディン・モスク', description: '純金のドームが輝く美しいモスク。ブルネイ川に浮かぶように建つ東南アジア屈指のイスラム建築です。' },
      { title: 'カンポン・アイールの水上村', description: '600年の歴史を持つ世界最大の水上集落。「東洋のヴェネツィア」と呼ばれる伝統的な水上生活を見学できます。' },
      { title: 'ウル・テンブロン国立公園', description: '手つかずのボルネオ熱帯雨林。樹冠ウォークで熱帯雨林の生態系を空中から観察できる貴重な体験です。' },
      { title: '世界一豪華な王宮', description: 'イスタナ・ヌルル・イマン宮殿は部屋数1,788室の世界最大の王宮。スルタンの絶対的な権力と富を象徴しています。' }
    ],
    whyVisit: '石油王国の豪華さと手つかずの熱帯雨林を体験できる東南アジアの宝石'
  },

  'bulgaria': {
    name: 'ブルガリア',
    nameEn: 'Bulgaria',
    flag: '🇧🇬',
    code: 'bg',
    basic: { capital: 'ソフィア', population: '700万人', language: 'ブルガリア語' },
    coordinates: { lat: 42.6977, lng: 23.3219 },
    description: 'バルカン半島の東欧の国で、キリル文字発祥の地。バラの谷で有名で、ヨーグルト文化発祥の地としても知られています。',
    highlights: [
      { title: 'バラの谷の芳香', description: '世界最高品質のダマスクローズの産地。5-6月には一面のバラ畑が広がり、バラ摘み祭りで伝統文化を体験できます。' },
      { title: 'リラ修道院の美しさ', description: '10世紀創建のブルガリア正教の聖地。美しいフレスコ画と山岳地帯の景観が調和した世界遺産です。' },
      { title: 'キリル文字の発祥地', description: 'スラヴ系民族の文字を作った聖キリルと聖メトディオスの故郷。現在もロシア語など多くの言語で使われています。' },
      { title: 'ヨーグルト発祥の地', description: 'ブルガリア乳酸菌ラクトバチルス・ブルガリクスの発見地。長寿の秘訣とされる伝統的な発酵食品文化があります。' }
    ],
    whyVisit: 'バラの香りとキリル文字の歴史、正教会文化の深さを体験できる国'
  },

  'burkina faso': {
    name: 'ブルキナファソ',
    nameEn: 'Burkina Faso',
    flag: '🇧🇫',
    code: 'bf',
    basic: { capital: 'ワガドゥグー', population: '2,200万人', language: 'フランス語' },
    coordinates: { lat: 12.3714, lng: -1.5197 },
    description: '西アフリカの内陸国で「清廉潔白な人々の国」という意味の名前を持つ。豊かな伝統文化と手工芸で知られています。',
    highlights: [
      { title: 'FESPACO映画祭', description: 'アフリカ最大の映画祭。2年に一度開催される映画とテレビの祭典で、アフリカ映画文化の中心地です。' },
      { title: 'ボボ・ディウラッソの建築', description: 'スーダン・サヘル建築の傑作。泥レンガで作られた大モスクは西アフリカ建築の代表例です。' },
      { title: 'アルティザナル工芸品', description: 'ブロンズ像、木彫り、織物など質の高い手工芸品。特にブロンズ鋳造技術は1000年以上の歴史があります。' },
      { title: 'W国立公園の野生動物', description: 'ニジェール川流域の生物多様性保護区。ゾウ、ライオン、チーターなど西アフリカの野生動物を観察できます。' }
    ],
    whyVisit: '西アフリカの伝統文化と現代アート、野生動物を体験できる国'
  },

  'burundi': {
    name: 'ブルンジ',
    nameEn: 'Burundi',
    flag: '🇧🇮',
    code: 'bi',
    basic: { capital: 'ギテガ', population: '1,200万人', language: 'キルンジ語・フランス語' },
    coordinates: { lat: -3.3731, lng: 29.9189 },
    description: '東アフリカの小さな内陸国で「アフリカの心臓部」と呼ばれる。美しい丘陵地帯と湖で知られ、伝統的なドラム文化が有名です。',
    highlights: [
      { title: 'タンガニーカ湖の美しさ', description: '世界第2位の深さを誇る古代湖。透明度が高く、固有種の熱帯魚が多数生息する淡水の海です。' },
      { title: '千の丘の国の景観', description: 'なだらかな丘陵が続く美しい田園風景。緑豊かなテラス農業と伝統的な農村生活を見ることができます。' },
      { title: 'ブルンジ太鼓の伝統', description: 'ユネスコ無形文化遺産に登録された神聖な太鼓演奏。王室儀礼で演奏される力強いリズムは魂を揺さぶります。' },
      { title: 'キボ国立公園の自然', description: '山地森林とサバンナが共存する国立公園。希少なシロサイやチンパンジーなどの野生動物保護区です。' }
    ],
    whyVisit: 'アフリカの伝統文化と美しい自然景観を静かに楽しめる隠れた宝石'
  },

  'cambodia': {
    name: 'カンボジア',
    nameEn: 'Cambodia',
    flag: '🇰🇭',
    code: 'kh',
    basic: { capital: 'プノンペン', population: '1,700万人', language: 'クメール語' },
    coordinates: { lat: 11.5449, lng: 104.8922 },
    description: '東南アジアの古代クメール帝国の中心地。アンコール遺跡群で有名で、1000年の歴史を持つ壮大な石造寺院群が魅力です。',
    highlights: [
      { title: 'アンコール・ワットの壮大さ', description: '12世紀建造の世界最大の宗教建築。朝日に照らされる5つの尖塔は、地上に現れた天界の象徴です。' },
      { title: 'バイヨン寺院の観世音菩薩', description: '216の微笑む巨大な顔で覆われた神秘的な寺院。「クメールの微笑み」と呼ばれる穏やかな表情に心が癒されます。' },
      { title: 'トンレサップ湖の水上生活', description: '東南アジア最大の淡水湖。雨季には面積が4倍になり、水上村落で伝統的な漁業文化を見学できます。' },
      { title: 'クメール古典舞踊', description: 'アプサラダンスなど宮廷で育まれた優雅な舞踊。手の動きで物語を表現する繊細で美しい伝統芸能です。' }
    ],
    whyVisit: '世界最大の宗教建築群と古代クメール文明の神秘を体験できる国'
  },

  'cameroon': {
    name: 'カメルーン',
    nameEn: 'Cameroon',
    flag: '🇨🇲',
    code: 'cm',
    basic: { capital: 'ヤウンデ', population: '2,700万人', language: 'フランス語・英語' },
    coordinates: { lat: 3.8480, lng: 11.5021 },
    description: '中央アフリカの「アフリカの縮図」と呼ばれる多様性の国。熱帯雨林から砂漠まで、アフリカ大陸の全ての気候と文化が凝縮されています。',
    highlights: [
      { title: 'カメルーン山の活火山', description: '西アフリカ最高峰4,095mの活火山。山頂からギニア湾を見下ろす絶景と、希少な高山植物を観察できます。' },
      { title: 'ジャー動物保護区', description: 'ユネスコ世界遺産の熱帯雨林。ローランドゴリラやフォレストエレファントなど希少動物の宝庫です。' },
      { title: '280以上の民族の多様性', description: 'アフリカで最も多様な民族構成。バントゥー、スーダン、ハミテックなど異なる文化が共存しています。' },
      { title: 'ワザ国立公園のサバンナ', description: '北部サヘル地帯の乾燥サバンナ。ライオン、チーター、ゾウなど大型野生動物のサファリを楽しめます。' }
    ],
    whyVisit: 'アフリカ大陸の全ての自然と文化を一度に体験できる多様性の国'
  },

  'canada': {
    name: 'カナダ',
    nameEn: 'Canada',
    flag: '🇨🇦',
    code: 'ca',
    basic: { capital: 'オタワ', population: '3,900万人', language: '英語・フランス語' },
    coordinates: { lat: 45.4215, lng: -75.6972 },
    description: '世界第2位の国土を持つ北米の大国。ナイアガラの滝、ロッキー山脈、オーロラなど壮大な自然と多文化社会で知られています。',
    highlights: [
      { title: 'ナイアガラの滝の大迫力', description: '世界三大瀑布の一つ。毎分600万リットルの水が落下する壮大なスペクタクルを間近で体感できます。' },
      { title: 'カナディアンロッキーの絶景', description: 'バンフとジャスパー国立公園の山岳風景。エメラルドグリーンの湖と雪化粧した山々の調和が美しすぎます。' },
      { title: 'イエローナイフのオーロラ', description: '世界屈指のオーロラ観測地。年間240日以上オーロラが出現し、緑のカーテンが夜空を舞い踊ります。' },
      { title: 'ケベック・シティの古き良きヨーロッパ', description: '北米唯一の城郭都市。フランス植民地時代の石造建築が残る、まるでヨーロッパのような街並みです。' }
    ],
    whyVisit: '雄大な自然と多文化の調和が生み出す、地球上で最も美しい国の一つ'
  },

  'chile': {
    name: 'チリ',
    nameEn: 'Chile',
    flag: '🇨🇱',
    code: 'cl',
    basic: { capital: 'サンティアゴ', population: '1,900万人', language: 'スペイン語' },
    coordinates: { lat: -33.4489, lng: -70.6693 },
    description: '南北4,300kmに細長く伸びる国。アタカマ砂漠からパタゴニアまで、地球上で最も多様な自然環境を持つ国です。',
    highlights: [
      { title: 'アタカマ砂漠の異世界風景', description: '世界で最も乾燥した砂漠。火星のような風景と世界最高の星空観測環境で、宇宙の神秘を感じられます。' },
      { title: 'パタゴニアの氷河と山岳', description: 'トーレス・デル・パイネの奇峰群。氷河、湖、草原が織りなす地球最後の秘境の絶景です。' },
      { title: 'イースター島のモアイ像', description: '太平洋の孤島に立つ1,000体の巨石像。ポリネシア文明の謎に満ちた遺跡は世界最大のミステリーです。' },
      { title: '世界最高品質のワイン', description: 'アンデス山脈の麓で育つプレミアムワイン。チリワインは世界中で愛される上質な味わいです。' }
    ],
    whyVisit: '地球上の全ての自然環境と古代文明の謎を体験できる細長い奇跡の国'
  },

  'china': {
    name: '中国',
    nameEn: 'China',
    flag: '🇨🇳',
    code: 'cn',
    basic: { capital: '北京', population: '14億人', language: '中国語（標準中国語）' },
    coordinates: { lat: 39.9042, lng: 116.4074 },
    description: '5000年の歴史を持つ古代文明発祥の地。万里の長城、紫禁城、兵馬俑など世界的な文化遺産と急速な近代化が共存する大国です。',
    highlights: [
      { title: '万里の長城の壮大さ', description: '総延長2万キロの人類史上最大の建造物。秦の始皇帝時代から続く悠久の歴史を歩いて体感できます。' },
      { title: '兵馬俑の8000体の軍団', description: '始皇帝陵の地下に眠る等身大の兵士俑。2000年前の超絶技巧と古代中国の国力を物語る世界的発見です。' },
      { title: '桂林の水墨画風景', description: '漓江に浮かぶカルスト地形の山々。中国古典絵画そのままの幻想的な風景は「山水甲天下」と称賛されます。' },
      { title: 'チベット・ポタラ宮殿の神聖さ', description: '標高3,700mの聖地ラサに聳える宮殿。ダライ・ラマの居住地として1,000年の歴史を持つチベット仏教の聖地です。' }
    ],
    whyVisit: '人類最古の文明と現代の奇跡が同時に体験できる壮大なスケールの国'
  },

  'colombia': {
    name: 'コロンビア',
    nameEn: 'Colombia',
    flag: '🇨🇴',
    code: 'co',
    basic: { capital: 'ボゴタ', population: '5,100万人', language: 'スペイン語' },
    coordinates: { lat: 4.7110, lng: -74.0721 },
    description: '南米北西部の美しい国。太平洋とカリブ海に面し、コーヒー、エメラルド、音楽の国として世界に知られています。',
    highlights: [
      { title: '世界最高品質のコーヒー', description: 'アンデス山脈の高地で育つ極上のアラビカ種。コーヒー文化景観は世界遺産に登録された文化的価値の高い産業です。' },
      { title: 'カルタヘナの城塞都市', description: 'カリブ海に面したスペイン植民地時代の要塞都市。カラフルなコロニアル建築と城壁に囲まれた美しい旧市街です。' },
      { title: '世界最大のエメラルド産地', description: '世界のエメラルドの約70%を産出。ムソやチボールの鉱山で採れる深緑色の宝石は世界最高品質です。' },
      { title: 'クンビアとサルサの本場', description: 'ラテン音楽発祥の地の一つ。情熱的なリズムとダンス文化で、一年中どこかで音楽祭が開催されています。' }
    ],
    whyVisit: 'コーヒーの香りと音楽のリズムに包まれた、南米で最も陽気な国'
  },

  'congo': {
    name: 'コンゴ共和国',
    nameEn: 'Congo',
    flag: '🇨🇬',
    code: 'cg',
    basic: { capital: 'ブラザビル', population: '550万人', language: 'フランス語' },
    coordinates: { lat: -4.2634, lng: 15.2429 },
    description: '中央アフリカのコンゴ川流域に位置する森林国。手つかずの熱帯雨林とコンゴ川の雄大な流れで知られています。',
    highlights: [
      { title: 'オザラ国立公園のゴリラ', description: '西ローランドゴリラの重要な生息地。密林の奥でゴリラの家族と出会う貴重なゴリラトレッキング体験ができます。' },
      { title: 'コンゴ川の雄大な流れ', description: 'アフリカ第2の長さを誇る大河。ブラザビルとキンシャサを隔てる激流の迫力と、川沿いの豊かな生態系を楽しめます。' },
      { title: 'サンガ川流域の原生林', description: 'ユネスコ世界遺産の手つかずの熱帯雨林。フォレストエレファントやチンパンジーなど希少動物の最後の楽園です。' },
      { title: 'ピグミー族の森林文化', description: '数万年前から森と共に生きる先住民の知恵。持続可能な森林利用と独特の音楽文化を学ぶことができます。' }
    ],
    whyVisit: 'アフリカ中央部の原始の森で野生動物と先住民文化に出会える国'
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