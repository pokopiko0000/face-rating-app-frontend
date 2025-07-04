// 各国の厳選された美しい画像URL
export const countryImages: Record<string, string> = {
  // メジャー国
  'korea': 'https://images.unsplash.com/photo-1517154421773-0529f29ea451?w=1200&h=800&fit=crop', // ソウル夜景
  'japan': 'https://images.unsplash.com/photo-1490650034439-fd184c3c86a5?w=1200&h=800&fit=crop', // 富士山と桜
  'usa': 'https://images.unsplash.com/photo-1485738422979-f5c462d49f74?w=1200&h=800&fit=crop', // ニューヨーク
  'france': 'https://images.unsplash.com/photo-1502602898536-47ad22581b52?w=1200&h=800&fit=crop', // エッフェル塔
  'italy': 'https://images.unsplash.com/photo-1515542622106-78bda8ba0e5b?w=1200&h=800&fit=crop', // ベニス
  'china': 'https://images.unsplash.com/photo-1508804185872-d7badad00f7d?w=1200&h=800&fit=crop', // 万里の長城
  'germany': 'https://images.unsplash.com/photo-1467269204594-9661b134dd2b?w=1200&h=800&fit=crop', // ノイシュヴァンシュタイン城
  'uk': 'https://images.unsplash.com/photo-1513635269975-59663e0ac1ad?w=1200&h=800&fit=crop', // ロンドン
  
  // 中堅国
  'finland': 'https://images.unsplash.com/photo-1578662996442-48f60103fc96?w=1200&h=800&fit=crop', // オーロラ
  'thailand': 'https://images.unsplash.com/photo-1520637836862-4d197d17c13a?w=1200&h=800&fit=crop', // タイの寺院
  'vietnam': 'https://images.unsplash.com/photo-1557750255-c76072a7aad1?w=1200&h=800&fit=crop', // ハロン湾
  'mexico': 'https://images.unsplash.com/photo-1518105779142-d975f22f1b0a?w=1200&h=800&fit=crop', // チチェン・イッツァ
  'brazil': 'https://images.unsplash.com/photo-1483729558449-99ef09a8c325?w=1200&h=800&fit=crop', // リオ キリスト像
  'india': 'https://images.unsplash.com/photo-1564507592333-c60657eea523?w=1200&h=800&fit=crop', // タージマハル
  'australia': 'https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=1200&h=800&fit=crop', // シドニー オペラハウス
  'canada': 'https://images.unsplash.com/photo-1503614472-8c93d56e92ce?w=1200&h=800&fit=crop', // バンフ国立公園
  'spain': 'https://images.unsplash.com/photo-1539037116277-4db20889f2d4?w=1200&h=800&fit=crop', // サグラダファミリア
  'netherlands': 'https://images.unsplash.com/photo-1534351590666-13e3e96b5017?w=1200&h=800&fit=crop', // アムステルダム運河
  
  // マイナー国
  'bhutan': 'https://images.unsplash.com/photo-1609156239782-0b9c3b35ecf0?w=1200&h=800&fit=crop', // タイガーズネスト僧院
  'maldives': 'https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=1200&h=800&fit=crop', // モルディブビーチ
  'iceland': 'https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=1200&h=800&fit=crop', // アイスランド氷河
  'malta': 'https://images.unsplash.com/photo-1582719201952-2c3b9c88b3f1?w=1200&h=800&fit=crop', // マルタ ヴァレッタ
  'luxembourg': 'https://images.unsplash.com/photo-1571115764595-644a1f56a55c?w=1200&h=800&fit=crop', // ルクセンブルク城
  'brunei': 'https://images.unsplash.com/photo-1561114120-9b0c60e4f82a?w=1200&h=800&fit=crop', // ブルネイ モスク
  'montenegro': 'https://images.unsplash.com/photo-1545558014-8692077e9b5c?w=1200&h=800&fit=crop', // コトル湾
  'estonia': 'https://images.unsplash.com/photo-1539650116574-75c0c6d75d24?w=1200&h=800&fit=crop', // タリン旧市街
  'latvia': 'https://images.unsplash.com/photo-1529963183134-61a90db47eaf?w=1200&h=800&fit=crop', // リガ旧市街
  'lithuania': 'https://images.unsplash.com/photo-1469474968028-56623f02e42e?w=1200&h=800&fit=crop', // ヴィリニュス旧市街
  'slovenia': 'https://images.unsplash.com/photo-1520637836862-4d197d17c13a?w=1200&h=800&fit=crop', // ブレッド湖
  'cyprus': 'https://images.unsplash.com/photo-1605540436563-5bca919ae766?w=1200&h=800&fit=crop', // キプロス海岸
  'andorra': 'https://images.unsplash.com/photo-1469474968028-56623f02e42e?w=1200&h=800&fit=crop', // アンドラ山景
  'san-marino': 'https://images.unsplash.com/photo-1578662996442-48f60103fc96?w=1200&h=800&fit=crop', // サンマリノ塔
  'monaco': 'https://images.unsplash.com/photo-1582719201952-2c3b9c88b3f1?w=1200&h=800&fit=crop', // モナコ港
  'vatican': 'https://images.unsplash.com/photo-1448932223592-d1fc686e76ea?w=1200&h=800&fit=crop', // バチカン
  'seychelles': 'https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=1200&h=800&fit=crop', // セーシェル ビーチ
  'palau': 'https://images.unsplash.com/photo-1559827260-dc66d52bef19?w=1200&h=800&fit=crop', // パラオ ジェリーフィッシュレイク
  'tuvalu': 'https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=1200&h=800&fit=crop', // ツバル環礁
  'nauru': 'https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=1200&h=800&fit=crop', // ナウル島
  'kiribati': 'https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=1200&h=800&fit=crop', // キリバス環礁
  'marshall-islands': 'https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=1200&h=800&fit=crop', // マーシャル諸島
  'micronesia': 'https://images.unsplash.com/photo-1559827260-dc66d52bef19?w=1200&h=800&fit=crop', // ミクロネシア
  'tonga': 'https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=1200&h=800&fit=crop', // トンガ
  'samoa': 'https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=1200&h=800&fit=crop', // サモア
  'vanuatu': 'https://images.unsplash.com/photo-1559827260-dc66d52bef19?w=1200&h=800&fit=crop', // バヌアツ火山
  'fiji': 'https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=1200&h=800&fit=crop', // フィジー
};

// フォールバック画像（国の画像が見つからない場合）
export const getFallbackImage = (countryName: string): string => {
  return `https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=1200&h=800&fit=crop&q=80`;
};

// 画像取得関数
export const getCountryImage = (countryCode: string, countryName: string): string => {
  return countryImages[countryCode.toLowerCase()] || getFallbackImage(countryName);
};