#!/usr/bin/env node

/**
 * 自動化スクリプト: 各国ページのコンテンツを生成
 * フィンランドで確立したノウハウを他の229カ国に適用
 */

const fs = require('fs');
const path = require('path');

// 設定
const PROJECT_DIR = path.join(__dirname, 'project', 'src', 'data');
const COUNTRIES_FILE = path.join(PROJECT_DIR, 'countries.ts');
const OUTPUT_FILE = path.join(PROJECT_DIR, 'countries_updated.ts');

// 各国の特色ある文化・観光要素のデータベース（一部例）
const COUNTRY_HIGHLIGHTS = {
  'jp': {
    highlights: [
      { title: '富士山と桜の絶景', description: '世界遺産に登録された富士山は日本の象徴的な美しさを誇ります。春には桜が咲き誇り、まさに日本らしい風景が楽しめます。河口湖や忍野八海からの絶景は訪れる人々を魅了し続けています。' },
      { title: '世界最高峰の和食文化', description: 'ユネスコ無形文化遺産に登録された和食は、四季の移ろいと自然への敬意が込められています。寿司、天ぷら、懐石料理など、繊細で美しい料理は五感すべてで楽しむ芸術作品です。' },
      { title: '最先端技術と伝統工芸の融合', description: 'ロボット技術やAIなど最先端技術と、1000年以上続く伝統工芸が共存する稀有な国。京都の伝統建築と東京のネオンサインが織りなすコントラストは、他では体験できない魅力です。' },
      { title: '温泉文化と癒しの時間', description: '全国3000カ所以上の温泉地で、日本人が古来より愛してきた湯治文化を体験できます。露天風呂から眺める四季の風景は、日常の疲れを洗い流してくれる究極の癒し体験です。' }
    ],
    description: '古来より続く伝統と最先端技術が美しく融合した島国、日本。四季折々の自然美、世界遺産に登録された和食文化、1000年以上の歴史を誇る寺社仏閣など、他では体験できない独特の魅力に満ちています。「おもてなし」の心で迎えられる温かい人々との出会いも、きっと心に残る思い出となるでしょう。',
    whyVisit: '伝統と革新が織りなす美しい調和の中で、心も技術も洗練される体験を'
  },
  // 他の国の例も追加可能
};

// AI風のコンテンツ生成関数（実際のAI APIの代わりに、パターンベースで生成）
function generateCountryContent(countryData) {
  const { name, nameEn, basic } = countryData;
  
  // 基本的な説明文の生成パターン
  const descriptionTemplates = [
    `${name}は${basic.capital}を首都とする魅力的な国です。${basic.population}の人々が${basic.language}を話し、豊かな文化と歴史を築いています。独特の自然環境と伝統文化が織りなす美しい風景は、訪れる人々に深い感動を与えます。`,
    `${basic.capital}を中心に発展した${name}は、${basic.population}の多様な人々が暮らす国です。${basic.language}の響きと共に育まれた文化は、長い歴史の中で独自の発展を遂げ、現代でも多くの人々を魅了し続けています。`,
    `${name}（${nameEn}）は${basic.population}の人口を持つ、文化的に豊かな国です。首都${basic.capital}をはじめ、各地域が持つ独特の魅力は、${basic.language}という言語と共に受け継がれてきた歴史の深さを物語っています。`
  ];

  // ランダムに説明文を選択
  const description = descriptionTemplates[Math.floor(Math.random() * descriptionTemplates.length)];

  // ハイライトの基本パターン
  const highlightTemplates = [
    {
      title: '豊かな文化遺産',
      description: `${name}の文化的遺産は長い歴史の中で育まれ、現在も人々の生活に深く根ざしています。伝統的な建築様式や工芸品、音楽や舞踊など、この国ならではの文化的魅力を体験することができます。`
    },
    {
      title: '美しい自然環境',
      description: `${name}の自然は四季を通じて異なる表情を見せ、訪れる人々を魅了します。山々、川、森林、海岸線など、多様な地形が織りなす風景は、この国特有の自然の美しさを表現しています。`
    },
    {
      title: '特色ある料理文化',
      description: `${name}の料理は地域の気候や文化、歴史が生み出した独特の味わいを持ちます。新鮮な地元の食材を使った伝統料理は、この国の人々の生活や文化を深く理解する素晴らしい窓口となります。`
    },
    {
      title: '温かな人々との出会い',
      description: `${name}の人々は訪問者を温かく迎え入れる文化を持っています。現地の人々との交流を通じて、この国の本当の魅力や日常生活の様子を知ることができる貴重な体験が待っています。`
    }
  ];

  const whyVisitTemplates = [
    `${name}ならではの特別な体験と出会いが待っています`,
    `豊かな文化と自然に包まれた${name}で、心に残る旅の思い出を`,
    `${name}の魅力的な文化と美しい風景があなたを待っています`
  ];

  return {
    description,
    highlights: highlightTemplates,
    whyVisit: whyVisitTemplates[Math.floor(Math.random() * whyVisitTemplates.length)]
  };
}

// メイン処理
function processCountriesFile() {
  console.log('🚀 国別ページ自動生成システム開始...');
  
  try {
    // ファイル読み込み
    const content = fs.readFileSync(COUNTRIES_FILE, 'utf8');
    console.log('📖 countries.tsファイルを読み込みました');

    // テンプレート的な内容を持つ国を検索・置換
    let updatedContent = content;
    let updatedCount = 0;

    // 正規表現でテンプレート的な内容を検索
    const countryRegex = /('[\w]+'):\s*\{[\s\S]*?whyVisit:\s*'[^']*'/g;
    
    updatedContent = updatedContent.replace(countryRegex, (match) => {
      // テンプレート的な内容があるかチェック
      if (match.includes('この国独特の文化を体験できます。')) {
        // 国データを抽出
        const nameMatch = match.match(/name:\s*'([^']*)'/) || [];
        const nameEnMatch = match.match(/nameEn:\s*'([^']*)'/) || [];
        const capitalMatch = match.match(/capital:\s*'([^']*)'/) || [];
        const populationMatch = match.match(/population:\s*'([^']*)'/) || [];
        const languageMatch = match.match(/language:\s*'([^']*)'/) || [];

        if (nameMatch[1] && capitalMatch[1]) {
          const countryData = {
            name: nameMatch[1],
            nameEn: nameEnMatch[1] || '',
            basic: {
              capital: capitalMatch[1],
              population: populationMatch[1] || '',
              language: languageMatch[1] || ''
            }
          };

          // 新しいコンテンツを生成
          const newContent = generateCountryContent(countryData);
          
          // 元のマッチ内容を新しい内容に置換
          const updatedMatch = match
            .replace(/description:\s*'[^']*'/, `description: '${newContent.description}'`)
            .replace(
              /highlights:\s*\[[\s\S]*?\]/,
              `highlights: [
      {
        title: '${newContent.highlights[0].title}',
        description: '${newContent.highlights[0].description}'
      },
      {
        title: '${newContent.highlights[1].title}',
        description: '${newContent.highlights[1].description}'
      },
      {
        title: '${newContent.highlights[2].title}',
        description: '${newContent.highlights[2].description}'
      },
      {
        title: '${newContent.highlights[3].title}',
        description: '${newContent.highlights[3].description}'
      },
    ]`
            )
            .replace(/whyVisit:\s*'[^']*'/, `whyVisit: '${newContent.whyVisit}'`);

          updatedCount++;
          console.log(`✨ ${countryData.name} (${countryData.nameEn}) のコンテンツを更新しました`);
          return updatedMatch;
        }
      }
      return match;
    });

    // 更新されたファイルを保存
    fs.writeFileSync(OUTPUT_FILE, updatedContent, 'utf8');
    
    console.log(`🎉 完了! ${updatedCount}カ国のページを更新しました`);
    console.log(`📁 更新されたファイル: ${OUTPUT_FILE}`);
    console.log(`💡 元のファイルをバックアップとして残し、新しいファイルを作成しました`);
    console.log(`🔄 新しいファイルを確認後、元のファイルと置き換えてください`);

  } catch (error) {
    console.error('❌ エラーが発生しました:', error.message);
    process.exit(1);
  }
}

// 実行確認
console.log('🌍 各国ページ自動生成システム');
console.log('このスクリプトは229カ国のテンプレート内容を充実した内容に更新します。');
console.log('続行しますか? (Ctrl+C で中止)');

// 5秒後に実行開始
setTimeout(() => {
  processCountriesFile();
}, 5000);