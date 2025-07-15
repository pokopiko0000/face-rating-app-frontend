// Unsplash API integration for country images
const UNSPLASH_ACCESS_KEY = 'YOUR_UNSPLASH_ACCESS_KEY'; // 開発時は環境変数に設定

// Type definitions for Unsplash API
interface UnsplashPhoto {
  id: string;
  urls: {
    regular: string;
    small: string;
  };
  alt_description?: string;
  user: {
    name: string;
    links: {
      html: string;
    };
  };
}

// 国別の画像検索キーワード
export const countryImageKeywords: Record<string, string[]> = {
  // メジャー国
  'korea': ['seoul skyline night', 'korean traditional palace', 'jeju island landscape'],
  'japan': ['tokyo skyline', 'mount fuji', 'kyoto temple'],
  'usa': ['new york manhattan', 'grand canyon', 'golden gate bridge'],
  'france': ['eiffel tower paris', 'provence lavender', 'versailles palace'],
  'italy': ['colosseum rome', 'venice canal', 'tuscany countryside'],
  'china': ['great wall china', 'forbidden city beijing', 'shanghai skyline'],
  'germany': ['neuschwanstein castle', 'berlin brandenburg gate', 'bavarian alps'],
  'uk': ['big ben london', 'scottish highlands', 'stonehenge'],
  
  // 中堅国
  'finland': ['aurora borealis finland', 'helsinki cathedral', 'finnish forest lake'],
  'thailand': ['bangkok temple', 'phi phi island', 'floating market'],
  'vietnam': ['halong bay', 'hoi an ancient town', 'sapa terraces'],
  'mexico': ['chichen itza', 'cancun beach', 'mexico city zocalo'],
  'brazil': ['christ redeemer rio', 'iguazu falls', 'copacabana beach'],
  'india': ['taj mahal', 'kerala backwaters', 'rajasthan palace'],
  'australia': ['sydney opera house', 'uluru ayers rock', 'great barrier reef'],
  'canada': ['banff national park', 'niagara falls', 'toronto skyline'],
  'spain': ['sagrada familia barcelona', 'alhambra granada', 'ibiza beach'],
  'netherlands': ['amsterdam canal', 'keukenhof tulips', 'kinderdijk windmills'],
  
  // マイナー国
  'bhutan': ['tigers nest monastery bhutan', 'himalaya mountain bhutan', 'thimphu dzong'],
  'maldives': ['maldives resort', 'maldives crystal water', 'maldives sunset'],
  'iceland': ['northern lights iceland', 'blue lagoon iceland', 'gullfoss waterfall'],
  'malta': ['valletta malta', 'blue grotto malta', 'gozo azure window'],
  'luxembourg': ['luxembourg city', 'vianden castle', 'ardennes forest'],
  'brunei': ['sultan omar mosque brunei', 'bandar seri begawan', 'ulu temburong'],
  'montenegro': ['kotor bay', 'durmitor national park', 'sveti stefan'],
  'estonia': ['tallinn old town', 'lahemaa national park', 'saaremaa island'],
  'latvia': ['riga old town', 'gauja national park', 'jurmala beach'],
  'lithuania': ['vilnius old town', 'trakai castle', 'curonian spit'],
  'slovenia': ['lake bled', 'ljubljana castle', 'postojna cave'],
  'cyprus': ['paphos archaeology', 'troodos mountains', 'ayia napa beach'],
  'andorra': ['andorra la vella', 'pyrenees mountains', 'grandvalira ski'],
  'liechtenstein': ['vaduz castle', 'rhine valley', 'malbun alpine'],
  'san-marino': ['san marino towers', 'monte titano', 'palazzo pubblico'],
  'monaco': ['monte carlo casino', 'monaco harbor', 'prince palace'],
  'vatican': ['sistine chapel', 'st peters basilica', 'vatican gardens'],
  'seychelles': ['anse source argent', 'praslin island', 'la digue beach'],
  'palau': ['jellyfish lake palau', 'blue corner diving', 'rock islands'],
  'tuvalu': ['funafuti atoll', 'tuvalu lagoon', 'pacific island'],
  'nauru': ['nauru island aerial', 'pacific coral island', 'command ridge'],
  'kiribati': ['christmas island kiribati', 'gilbert islands', 'tarawa atoll'],
  'marshall-islands': ['majuro atoll', 'bikini atoll', 'marshall islands beach'],
  'micronesia': ['chuuk lagoon', 'pohnpei waterfall', 'yap island'],
  'tonga': ['tongatapu island', 'vavau islands', 'tonga beach'],
  'samoa': ['samoa beach', 'to sua trench', 'savaii island'],
  'vanuatu': ['vanuatu volcano', 'espiritu santo', 'port vila harbor'],
  'fiji': ['bora bora fiji', 'yasawa islands', 'coral coast'],
};

// デフォルトの検索キーワード（国固有のキーワードがない場合）
const getDefaultKeywords = (countryName: string): string[] => [
  `${countryName} landscape`,
  `${countryName} tourist attraction`,
  `${countryName} nature`
];

// Unsplash APIから画像を取得
export const fetchCountryImages = async (
  countryCode: string, 
  countryName: string,
  count: number = 3
): Promise<UnsplashImage[]> => {
  try {
    // 開発時はダミー画像を返す（APIキーが設定されていない場合）
    if (!UNSPLASH_ACCESS_KEY || UNSPLASH_ACCESS_KEY === 'YOUR_UNSPLASH_ACCESS_KEY') {
      return getDummyImages(countryName, count);
    }

    const keywords = countryImageKeywords[countryCode] || getDefaultKeywords(countryName);
    const randomKeyword = keywords[Math.floor(Math.random() * keywords.length)];
    
    const response = await fetch(
      `https://api.unsplash.com/search/photos?query=${encodeURIComponent(randomKeyword)}&per_page=${count}&orientation=landscape`,
      {
        headers: {
          'Authorization': `Client-ID ${UNSPLASH_ACCESS_KEY}`
        }
      }
    );

    if (!response.ok) {
      throw new Error('Failed to fetch images');
    }

    const data = await response.json();
    
    return data.results.map((photo: UnsplashPhoto) => ({
      id: photo.id,
      url: photo.urls.regular,
      thumb: photo.urls.small,
      alt: photo.alt_description || `${countryName} image`,
      photographer: photo.user.name,
      photographerUrl: photo.user.links.html
    }));
  } catch (error) {
    console.error('Error fetching images:', error);
    return getDummyImages(countryName, count);
  }
};

// 特定のキーワードで画像を取得（観光地別など）
export const fetchImagesByKeyword = async (
  keyword: string,
  count: number = 1
): Promise<UnsplashImage[]> => {
  try {
    if (!UNSPLASH_ACCESS_KEY || UNSPLASH_ACCESS_KEY === 'YOUR_UNSPLASH_ACCESS_KEY') {
      return [{
        id: 'dummy',
        url: `https://source.unsplash.com/800x600/?${encodeURIComponent(keyword)}`,
        thumb: `https://source.unsplash.com/400x300/?${encodeURIComponent(keyword)}`,
        alt: keyword,
        photographer: 'Unsplash',
        photographerUrl: 'https://unsplash.com'
      }];
    }

    const response = await fetch(
      `https://api.unsplash.com/search/photos?query=${encodeURIComponent(keyword)}&per_page=${count}&orientation=landscape`,
      {
        headers: {
          'Authorization': `Client-ID ${UNSPLASH_ACCESS_KEY}`
        }
      }
    );

    if (!response.ok) {
      throw new Error('Failed to fetch images');
    }

    const data = await response.json();
    
    return data.results.map((photo: UnsplashPhoto) => ({
      id: photo.id,
      url: photo.urls.regular,
      thumb: photo.urls.small,
      alt: photo.alt_description || keyword,
      photographer: photo.user.name,
      photographerUrl: photo.user.links.html
    }));
  } catch (error) {
    console.error('Error fetching images:', error);
    return getDummyImages(keyword, count);
  }
};

// ダミー画像（開発・フォールバック用）
const getDummyImages = (subject: string, count: number): UnsplashImage[] => {
  return Array.from({ length: count }, (_, index) => ({
    id: `dummy-${index}`,
    url: `https://source.unsplash.com/800x600/?${encodeURIComponent(subject)}&sig=${index}`,
    thumb: `https://source.unsplash.com/400x300/?${encodeURIComponent(subject)}&sig=${index}`,
    alt: `${subject} image`,
    photographer: 'Unsplash',
    photographerUrl: 'https://unsplash.com'
  }));
};

// 画像の型定義
export interface UnsplashImage {
  id: string;
  url: string;
  thumb: string;
  alt: string;
  photographer: string;
  photographerUrl: string;
}