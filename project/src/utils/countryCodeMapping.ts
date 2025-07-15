/**
 * 国名から国コードへのマッピング機能
 * 診断結果の国名を適切な国コードに変換して、/country/:countryCodeへのナビゲーションを可能にする
 */

// 国名から国コードへのマッピング
export const countryNameToCode: Record<string, string> = {
  // 主要国
  'Japan': 'jp',
  'United States': 'us',
  'United Kingdom': 'gb',
  'Germany': 'de',
  'France': 'fr',
  'Italy': 'it',
  'Spain': 'es',
  'China': 'cn',
  'South Korea': 'kr',
  'North Korea': 'kp',
  'Russia': 'ru',
  'India': 'in',
  'Australia': 'au',
  'Canada': 'ca',
  'Brazil': 'br',
  'Mexico': 'mx',
  'Argentina': 'ar',
  'Egypt': 'eg',
  'South Africa': 'za',
  'Nigeria': 'ng',
  'Kenya': 'ke',
  'Morocco': 'ma',
  'Algeria': 'dz',
  'Thailand': 'th',
  'Vietnam': 'vn',
  'Indonesia': 'id',
  'Malaysia': 'my',
  'Singapore': 'sg',
  'Philippines': 'ph',
  'Turkey': 'tr',
  'Greece': 'gr',
  'Netherlands': 'nl',
  'Belgium': 'be',
  'Switzerland': 'ch',
  'Austria': 'at',
  'Sweden': 'se',
  'Norway': 'no',
  'Denmark': 'dk',
  'Finland': 'fi',
  'Iceland': 'is',
  'Ireland': 'ie',
  'Portugal': 'pt',
  'Poland': 'pl',
  'Czech Republic': 'cz',
  'Hungary': 'hu',
  'Romania': 'ro',
  'Bulgaria': 'bg',
  'Croatia': 'hr',
  'Serbia': 'rs',
  'Ukraine': 'ua',
  'Belarus': 'by',
  'Lithuania': 'lt',
  'Latvia': 'lv',
  'Estonia': 'ee',
  'Israel': 'il',
  'Iran': 'ir',
  'Iraq': 'iq',
  'Saudi Arabia': 'sa',
  'United Arab Emirates': 'ae',
  'Kuwait': 'kw',
  'Qatar': 'qa',
  'Bahrain': 'bh',
  'Oman': 'om',
  'Jordan': 'jo',
  'Lebanon': 'lb',
  'Syria': 'sy',
  'Afghanistan': 'af',
  'Pakistan': 'pk',
  'Bangladesh': 'bd',
  'Sri Lanka': 'lk',
  'Nepal': 'np',
  'Bhutan': 'bt',
  'Maldives': 'mv',
  'Myanmar': 'mm',
  'Cambodia': 'kh',
  'Laos': 'la',
  'Mongolia': 'mn',
  'Kazakhstan': 'kz',
  'Uzbekistan': 'uz',
  'Kyrgyzstan': 'kg',
  'Tajikistan': 'tj',
  'Turkmenistan': 'tm',
  'Georgia': 'ge',
  'Armenia': 'am',
  'Azerbaijan': 'az',
  'Chile': 'cl',
  'Peru': 'pe',
  'Colombia': 'co',
  'Venezuela': 've',
  'Ecuador': 'ec',
  'Bolivia': 'bo',
  'Paraguay': 'py',
  'Uruguay': 'uy',
  'New Zealand': 'nz',
  'Fiji': 'fj',
  'Papua New Guinea': 'pg',
  'Solomon Islands': 'sb',
  'Vanuatu': 'vu',
  'Samoa': 'ws',
  'Tonga': 'to',
  'Palau': 'pw',
  'Micronesia': 'fm',
  'Marshall Islands': 'mh',
  'Kiribati': 'ki',
  'Nauru': 'nr',
  'Tuvalu': 'tv',
  
  // アフリカ諸国
  'Ghana': 'gh',
  'Senegal': 'sn',
  'Mali': 'ml',
  'Burkina Faso': 'bf',
  'Niger': 'ne',
  'Chad': 'td',
  'Cameroon': 'cm',
  'Central African Republic': 'cf',
  'Congo': 'cd',
  'Gabon': 'ga',
  'Equatorial Guinea': 'gq',
  'Sao Tome and Principe': 'st',
  'Namibia': 'na',
  'Botswana': 'bw',
  'Zimbabwe': 'zw',
  'Zambia': 'zm',
  'Malawi': 'mw',
  'Mozambique': 'mz',
  'Madagascar': 'mg',
  'Mauritius': 'mu',
  'Seychelles': 'sc',
  'Comoros': 'km',
  'Cape Verde': 'cv',
  'Guinea': 'gn',
  'Guinea-Bissau': 'gw',
  'Sierra Leone': 'sl',
  'Liberia': 'lr',
  'Ivory Coast': 'ci',
  'Togo': 'tg',
  'Benin': 'bj',
  'Djibouti': 'dj',
  'Eritrea': 'er',
  'Ethiopia': 'et',
  'Somalia': 'so',
  'South Sudan': 'ss',
  'Sudan': 'sd',
  'Angola': 'ao',
  'Lesotho': 'ls',
  'Eswatini': 'sz',
  'Rwanda': 'rw',
  'Burundi': 'bi',
  'Uganda': 'ug',
  'Tanzania': 'tz',
  'Libya': 'ly',
  'Tunisia': 'tn',
  
  // その他の国・地域
  'Taiwan': 'tw',
  'Hong Kong': 'hk',
  'Macao': 'mo',
  'Monaco': 'mc',
  'Vatican City': 'va',
  'San Marino': 'sm',
  'Liechtenstein': 'li',
  'Andorra': 'ad',
  'Malta': 'mt',
  'Cyprus': 'cy',
  'Luxembourg': 'lu',
  'Slovenia': 'si',
  'Slovakia': 'sk',
  'Bosnia and Herzegovina': 'ba',
  'Montenegro': 'me',
  'Albania': 'al',
  'North Macedonia': 'mk',
  'Moldova': 'md',
  'Cuba': 'cu',
  'Jamaica': 'jm',
  'Haiti': 'ht',
  'Dominican Republic': 'do',
  'Trinidad and Tobago': 'tt',
  'Barbados': 'bb',
  'Bahamas': 'bs',
  'Belize': 'bz',
  'Guatemala': 'gt',
  'Honduras': 'hn',
  'El Salvador': 'sv',
  'Nicaragua': 'ni',
  'Costa Rica': 'cr',
  'Panama': 'pa',
  'Guyana': 'gy',
  'Suriname': 'sr',
  'Brunei': 'bn',
  'East Timor': 'tl'
};

/**
 * 国名から国コードを取得する関数
 * @param countryName 国名（英語）
 * @returns 国コード（小文字）または null
 */
export function getCountryCodeFromName(countryName: string): string | null {
  // 完全一致を試行
  const exactMatch = countryNameToCode[countryName];
  if (exactMatch) {
    return exactMatch;
  }
  
  // 大文字小文字を無視して検索
  const lowerCountryName = countryName.toLowerCase();
  for (const [name, code] of Object.entries(countryNameToCode)) {
    if (name.toLowerCase() === lowerCountryName) {
      return code;
    }
  }
  
  // 部分一致を試行（"Republic of"などの接頭辞・接尾辞を無視）
  const cleanCountryName = countryName
    .replace(/^(Republic of|Kingdom of|State of|Principality of|Duchy of)\s+/i, '')
    .replace(/\s+(Republic|Kingdom|State|Principality|Duchy)$/i, '')
    .trim();
  
  for (const [name, code] of Object.entries(countryNameToCode)) {
    const cleanName = name
      .replace(/^(Republic of|Kingdom of|State of|Principality of|Duchy of)\s+/i, '')
      .replace(/\s+(Republic|Kingdom|State|Principality|Duchy)$/i, '')
      .trim();
    
    if (cleanName.toLowerCase() === cleanCountryName.toLowerCase()) {
      return code;
    }
  }
  
  return null;
}

/**
 * 診断結果から国コードを取得する関数
 * バックエンドから国コードが提供されている場合はそれを使用し、
 * ない場合は国名からマッピングを試行する
 * @param countryName 国名
 * @param countryCode バックエンドから提供された国コード（nullable）
 * @returns 国コード（小文字）または null
 */
export function getCountryCodeFromDiagnosis(
  countryName: string,
  countryCode: string | null
): string | null {
  // バックエンドから国コードが提供されている場合はそれを使用
  if (countryCode) {
    return countryCode.toLowerCase();
  }
  
  // 国名からマッピングを試行
  return getCountryCodeFromName(countryName);
}

/**
 * 国コードが有効かどうかを確認する関数
 * countries.tsに定義されている国コードと照合
 * @param countryCode 国コード
 * @returns 有効な国コードかどうか
 */
export function isValidCountryCode(countryCode: string): boolean {
  // この関数は実際のcountries.tsの内容と照合する必要がある
  // 現在は基本的な検証のみ
  return countryCode.length === 2 && /^[a-z]{2}$/.test(countryCode);
}