#!/usr/bin/env node

/**
 * 実行可能な自動化システム
 * 使用方法: node run_automation.js
 * または: node run_automation.js --dry-run (テスト実行)
 */

const fs = require('fs');
const path = require('path');

// 設定
const PROJECT_DIR = path.join(__dirname, 'project', 'src', 'data');
const COUNTRIES_FILE = path.join(PROJECT_DIR, 'countries.ts');
const BACKUP_FILE = path.join(PROJECT_DIR, 'countries.backup.ts');

// コマンドライン引数をチェック
const isDryRun = process.argv.includes('--dry-run');

// 各国の特色データベース（主要国の特化情報）
const SPECIALIZED_CONTENT = {
  'jp': {
    description: '古来より続く伝統と最先端技術が美しく融合した島国、日本。四季折々の自然美、世界遺産に登録された和食文化、1000年以上の歴史を誇る寺社仏閣など、他では体験できない独特の魅力に満ちています。「おもてなし」の心で迎えられる温かい人々との出会いも、きっと心に残る思い出となるでしょう。',
    highlights: [
      { title: '富士山と桜の絶景', description: '世界遺産に登録された富士山は日本の象徴的な美しさを誇ります。春には桜が咲き誇り、まさに日本らしい風景が楽しめます。河口湖や忍野八海からの絶景は訪れる人々を魅了し続けています。' },
      { title: '世界最高峰の和食文化', description: 'ユネスコ無形文化遺産に登録された和食は、四季の移ろいと自然への敬意が込められています。寿司、天ぷら、懐石料理など、繊細で美しい料理は五感すべてで楽しむ芸術作品です。' },
      { title: '最先端技術と伝統工芸の融合', description: 'ロボット技術やAIなど最先端技術と、1000年以上続く伝統工芸が共存する稀有な国。京都の伝統建築と東京のネオンサインが織りなすコントラストは、他では体験できない魅力です。' },
      { title: '温泉文化と癒しの時間', description: '全国3000カ所以上の温泉地で、日本人が古来より愛してきた湯治文化を体験できます。露天風呂から眺める四季の風景は、日常の疲れを洗い流してくれる究極の癒し体験です。' }
    ],
    whyVisit: '伝統と革新が織りなす美しい調和の中で、心も技術も洗練される体験を'
  },
  'kr': {
    description: 'K-POPと韓流ドラマで世界を魅了し続ける韓国は、最先端の美容・テクノロジー文化と5000年の深い歴史が共存する動的な国です。24時間眠らないソウルの活気、済州島の美しい自然、そして世界最高レベルのスキンケア文化など、現代的な魅力と伝統的な価値が絶妙に調和しています。',
    highlights: [
      { title: '世界最先端の美容・スキンケア文化', description: '韓国は世界の美容トレンドをリードする国として知られ、革新的なスキンケア製品と技術で世界中の人々を魅了しています。明洞や江南でのショッピングは、最新の美容トレンドを体験できる絶好の機会です。' },
      { title: 'K-POP・K-ドラマの聖地', description: 'BTS、BLACKPINK、『愛の不時着』『イカゲーム』など、世界中で愛される韓流コンテンツの発祥地。江南スタイルで有名な江南区や、ドラマのロケ地巡りは韓流ファンにとって夢のような体験です。' },
      { title: '24時間眠らない活気あふれるソウル', description: 'ソウルは真夜中でも活気に満ちた都市。深夜まで営業するカフェ、24時間営業のサウナ「チムジルバン」、夜通し楽しめるカラオケなど、韓国の独特な夜文化を存分に楽しめます。' },
      { title: '済州島の美しい自然とリゾート', description: 'ユネスコ世界自然遺産に登録された済州島は、韓国人にとっての憧れのリゾート地。美しい海岸線、ハルラ山の雄大な景色、そして独特の文化を持つ海女（ヘニョ）文化が体験できます。' }
    ],
    whyVisit: 'K-文化の最前線と美しい自然が織りなす、現代アジアの魅力を体験'
  }
};

// AI風コンテンツ生成関数
function generateCountryContent(countryData) {
  const { name, nameEn, basic, code } = countryData;
  
  // 特化情報があればそれを使用
  if (SPECIALIZED_CONTENT[code]) {
    return SPECIALIZED_CONTENT[code];
  }
  
  // 一般的なパターンベース生成
  const regionPatterns = {
    // ヨーロッパ
    europe: {
      description: `${name}は豊かな歴史と文化遺産で知られるヨーロッパの国です。${basic.capital}を中心とした美しい建築と、${basic.language}で語り継がれてきた伝統が、現代でも人々の心を魅了しています。ヨーロッパならではの洗練された文化と、地域独特の魅力的な風習が調和した特別な体験が待っています。`,
      highlights: [
        { title: '歴史ある美しい街並み', description: `${basic.capital}をはじめとする各都市には、長い歴史を物語る美しい建築物が立ち並びます。石畳の道を歩きながら、中世から現代まで息づく歴史の重みを感じることができる貴重な体験です。` },
        { title: '伝統的な料理と地域の味', description: `${name}の料理は地域の気候と文化が生み出した独特の味わいを持ちます。新鮮な地元食材を使った伝統レシピは、この国の人々の暮らしと文化を理解する素晴らしい入口となります。` },
        { title: '芸術と文化の宝庫', description: `${name}は長い間、芸術家や思想家を輩出してきた文化的な国です。美術館や劇場、音楽ホールでは、この国が世界に与えた文化的影響を間近で感じることができます。` },
        { title: '温かな現地の人々との交流', description: `${name}の人々は訪問者を心から歓迎する文化を持っています。カフェでの何気ない会話や地域のイベントへの参加を通じて、この国の本当の魅力を発見することができるでしょう。` }
      ]
    },
    // アジア
    asia: {
      description: `${name}は古代からの伝統と現代の発展が見事に調和したアジアの国です。首都${basic.capital}を中心に、${basic.population}の人々が${basic.language}を話し、独特の文化的アイデンティティを築いています。スピリチュアルな寺院から現代的な都市景観まで、多様な魅力が織りなす特別な体験を提供しています。`,
      highlights: [
        { title: '神秘的な寺院と精神文化', description: `${name}には数多くの美しい寺院や聖地があり、古代から受け継がれてきた精神的な文化を体験できます。静寂な祈りの場で、心の平安と精神的な充実感を得ることができる貴重な体験です。` },
        { title: '香り豊かなスパイス料理', description: `${name}の料理は豊富なスパイスと新鮮な食材が織りなす、複雑で深い味わいが特徴です。地域ごとに異なる調理法と味付けは、この国の豊かな文化的多様性を物語っています。` },
        { title: '活気あふれる市場と街並み', description: `${basic.capital}の市場や旧市街では、地元の人々の生活に触れることができます。色とりどりの商品、活気ある売り手の声、そして独特の香りが混ざり合う、エネルギッシュな体験が待っています。` },
        { title: '伝統工芸と現代アートの融合', description: `${name}では古代から伝わる手工芸技術と現代的なアート表現が美しく融合しています。職人の技と現代作家の創造性が生み出す作品は、この国の文化的進化を表現しています。` }
      ]
    },
    // デフォルト
    default: {
      description: `${name}は${basic.capital}を首都とする、独特の魅力を持つ国です。${basic.population}の人々が${basic.language}を話し、豊かな自然環境と文化的伝統を築いてきました。この国ならではの風景と人々の暮らしは、訪れる人々に新鮮な驚きと深い感動を与えてくれます。`,
      highlights: [
        { title: '豊かな自然環境', description: `${name}の自然は四季を通じて異なる美しさを見せてくれます。山々、川、森林など、多様な地形が織りなす風景は、この国特有の自然の魅力を表現しており、心の癒しをもたらしてくれます。` },
        { title: '独特の文化と伝統', description: `${name}の文化的遺産は長い歴史の中で育まれ、現在も人々の生活に深く根ざしています。伝統的な祭りや工芸品、音楽や舞踊など、この国ならではの文化的魅力を体験することができます。` },
        { title: '地域特有の美味しい料理', description: `${name}の料理は地域の気候や文化、歴史が生み出した独特の味わいを持ちます。新鮮な地元の食材を使った伝統料理は、この国の人々の生活や文化を深く理解する素晴らしい窓口となります。` },
        { title: '心温まる人々との出会い', description: `${name}の人々は訪問者を温かく迎え入れる文化を持っています。現地の人々との交流を通じて、この国の本当の魅力や日常生活の様子を知ることができる貴重な体験が待っています。` }
      ]
    }
  };

  // 地域を推定（簡易版）
  let pattern = regionPatterns.default;
  if (['at', 'be', 'bg', 'hr', 'cy', 'cz', 'dk', 'ee', 'fi', 'fr', 'de', 'gr', 'hu', 'ie', 'it', 'lv', 'lt', 'lu', 'mt', 'nl', 'pl', 'pt', 'ro', 'sk', 'si', 'es', 'se', 'gb', 'is', 'no', 'ch', 'al', 'ad', 'by', 'ba', 'mk', 'md', 'mc', 'me', 'rs', 'ua', 'va'].includes(code)) {
    pattern = regionPatterns.europe;
  } else if (['cn', 'jp', 'kr', 'kp', 'mn', 'tw', 'hk', 'mo', 'th', 'vn', 'la', 'kh', 'mm', 'my', 'sg', 'id', 'ph', 'bn', 'tl', 'in', 'pk', 'bd', 'lk', 'np', 'bt', 'mv', 'af', 'ir', 'iq', 'sy', 'jo', 'lb', 'tr', 'am', 'az', 'ge', 'kz', 'kg', 'tj', 'tm', 'uz'].includes(code)) {
    pattern = regionPatterns.asia;
  }

  const whyVisitOptions = [
    `${name}ならではの特別な文化体験と美しい自然があなたを待っています`,
    `豊かな伝統と現代の魅力が調和した${name}で、忘れられない思い出を`,
    `${name}の独特な魅力と温かい人々との出会いが心を豊かにしてくれます`
  ];

  return {
    description: pattern.description,
    highlights: pattern.highlights,
    whyVisit: whyVisitOptions[Math.floor(Math.random() * whyVisitOptions.length)]
  };
}

// 国データ抽出関数
function extractCountryData(countryMatch) {
  const nameMatch = countryMatch.match(/name:\s*'([^']*)'/) || [];
  const nameEnMatch = countryMatch.match(/nameEn:\s*'([^']*)'/) || [];
  const codeMatch = countryMatch.match(/'([a-z]+)':\s*\{/) || [];
  const capitalMatch = countryMatch.match(/capital:\s*'([^']*)'/) || [];
  const populationMatch = countryMatch.match(/population:\s*'([^']*)'/) || [];
  const languageMatch = countryMatch.match(/language:\s*'([^']*)'/) || [];

  return {
    name: nameMatch[1] || '',
    nameEn: nameEnMatch[1] || '',
    code: codeMatch[1] || '',
    basic: {
      capital: capitalMatch[1] || '',
      population: populationMatch[1] || '',
      language: languageMatch[1] || ''
    }
  };
}

// メイン処理
async function processCountries() {
  console.log('🌍 各国ページ自動生成システム開始...');
  console.log(`🔧 モード: ${isDryRun ? 'ドライラン（実際の変更は行いません）' : '実際の更新を実行'}`);
  
  try {
    // ファイル読み込み
    const content = fs.readFileSync(COUNTRIES_FILE, 'utf8');
    console.log('📖 countries.tsファイルを読み込みました');

    // バックアップ作成
    if (!isDryRun) {
      fs.copyFileSync(COUNTRIES_FILE, BACKUP_FILE);
      console.log('💾 バックアップファイルを作成しました');
    }

    let updatedContent = content;
    let updatedCount = 0;
    let processedCountries = [];

    // 各国のブロックを処理
    const countryRegex = /('[\w]+'):\s*\{[\s\S]*?whyVisit:\s*'[^']*'/g;
    
    updatedContent = updatedContent.replace(countryRegex, (match) => {
      // テンプレート的な内容があるかチェック
      if (match.includes('この国独特の文化を体験できます。')) {
        const countryData = extractCountryData(match);
        
        if (countryData.name && countryData.basic.capital) {
          // 新しいコンテンツを生成
          const newContent = generateCountryContent(countryData);
          
          // エスケープ処理
          const escapeQuotes = (str) => str.replace(/'/g, "\\'");
          
          // 元のマッチ内容を新しい内容に置換
          let updatedMatch = match
            .replace(/description:\s*'[^']*'/, `description: '${escapeQuotes(newContent.description)}'`)
            .replace(/whyVisit:\s*'[^']*'/, `whyVisit: '${escapeQuotes(newContent.whyVisit)}'`);

          // ハイライトを更新
          const highlightSection = `highlights: [
      {
        title: '${escapeQuotes(newContent.highlights[0].title)}',
        description: '${escapeQuotes(newContent.highlights[0].description)}'
      },
      {
        title: '${escapeQuotes(newContent.highlights[1].title)}',
        description: '${escapeQuotes(newContent.highlights[1].description)}'
      },
      {
        title: '${escapeQuotes(newContent.highlights[2].title)}',
        description: '${escapeQuotes(newContent.highlights[2].description)}'
      },
      {
        title: '${escapeQuotes(newContent.highlights[3].title)}',
        description: '${escapeQuotes(newContent.highlights[3].description)}'
      },
    ]`;

          updatedMatch = updatedMatch.replace(/highlights:\s*\[[\s\S]*?\]/m, highlightSection);

          updatedCount++;
          processedCountries.push(`${countryData.name} (${countryData.code})`);
          
          if (updatedCount <= 5 || updatedCount % 10 === 0) {
            console.log(`✨ ${countryData.name} (${countryData.code}) を更新しました`);
          }

          return updatedMatch;
        }
      }
      return match;
    });

    // 結果を保存またはプレビュー
    if (!isDryRun) {
      fs.writeFileSync(COUNTRIES_FILE, updatedContent, 'utf8');
      console.log(`🎉 完了! ${updatedCount}カ国のページを更新しました`);
      console.log(`💾 バックアップ: ${BACKUP_FILE}`);
    } else {
      console.log(`🎯 ドライラン完了: ${updatedCount}カ国が更新対象です`);
      console.log('\n📋 更新対象国リスト（最初の10カ国）:');
      processedCountries.slice(0, 10).forEach(country => console.log(`  • ${country}`));
      if (processedCountries.length > 10) {
        console.log(`  ... 他 ${processedCountries.length - 10}カ国`);
      }
    }

    console.log('\n📊 処理統計:');
    console.log(`  • 総処理国数: ${updatedCount}カ国`);
    console.log(`  • フィンランドスタイルの特化コンテンツ: ${Object.keys(SPECIALIZED_CONTENT).length}カ国`);
    console.log(`  • パターンベース生成: ${updatedCount - Object.keys(SPECIALIZED_CONTENT).length}カ国`);

  } catch (error) {
    console.error('❌ エラーが発生しました:', error.message);
    process.exit(1);
  }
}

// 実行
if (isDryRun) {
  console.log('🚀 ドライラン実行中...\n');
  processCountries();
} else {
  console.log('⚠️  この操作は実際にファイルを変更します。');
  console.log('💡 先にドライランで確認することをお勧めします: node run_automation.js --dry-run');
  console.log('🔄 5秒後に実行開始... (Ctrl+Cで中止)');
  
  setTimeout(() => {
    processCountries();
  }, 5000);
}