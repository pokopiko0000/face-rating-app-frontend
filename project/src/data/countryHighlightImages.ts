type HighlightImageMap = Record<string, Record<string, string>>;

const highlightImages: HighlightImageMap = {
  'finland': {
    '神秘的なオーロラ観測体験': 'https://images.unsplash.com/photo-1531366936337-7c912a4589a7?w=400&h=300&fit=crop&q=80',
    'サウナ文化の本場': 'https://images.unsplash.com/photo-1571902943202-507ec2618e8f?w=400&h=300&fit=crop&q=80',
    'ムーミンの故郷': 'https://images.unsplash.com/photo-1578925518681-39e3ded07b3a?w=400&h=300&fit=crop&q=80',
    '18万8千の美しい湖': 'https://images.unsplash.com/photo-1527004013197-933c4bb611b3?w=400&h=300&fit=crop&q=80'
  },
  'bhutan': {
    '世界で唯一の炭素マイナス国': 'https://images.unsplash.com/photo-1441974231531-c6227db76b6e?w=400&h=300&fit=crop&q=80',
    'タイガーズネスト僧院の絶景': 'https://images.unsplash.com/photo-1605649487212-47bdab064df0?w=400&h=300&fit=crop&q=80',
    '国民総幸福量（GNH）の哲学': 'https://images.unsplash.com/photo-1529156069898-49953e39b3ac?w=400&h=300&fit=crop&q=80',
    '伝統的な民族衣装文化': 'https://images.unsplash.com/photo-1544112190-6d6c1e9efed8?w=400&h=300&fit=crop&q=80'
  },
  'korea': {
    '世界最先端の美容・スキンケア文化': 'https://images.unsplash.com/photo-1596462502278-27bfdc403348?w=400&h=300&fit=crop&q=80',
    'K-POP・K-ドラマの聖地': 'https://images.unsplash.com/photo-1571167936266-2e4c4e7a7b6a?w=400&h=300&fit=crop&q=80',
    '24時間眠らない活気あふれるソウル': 'https://images.unsplash.com/photo-1566140967404-b8b3c83bca3b?w=400&h=300&fit=crop&q=80',
    '済州島の美しい自然とリゾート': 'https://images.unsplash.com/photo-1533577116850-9cc66cad8a9b?w=400&h=300&fit=crop&q=80'
  },
  'japan': {
    '富士山と桜の絶景': 'https://images.unsplash.com/photo-1490806843957-31f4c9a91c65?w=400&h=300&fit=crop&q=80',
    '世界最高峰の和食文化': 'https://images.unsplash.com/photo-1580822184713-fc5400e7fe10?w=400&h=300&fit=crop&q=80',
    '最先端技術と伝統工芸の融合': 'https://images.unsplash.com/photo-1526481280693-3bfa7568e0f3?w=400&h=300&fit=crop&q=80',
    '温泉文化と癒しの時間': 'https://images.unsplash.com/photo-1559181567-c3190ca9959b?w=400&h=300&fit=crop&q=80'
  },
  'maldives': {
    '世界最高レベルの透明度の海': 'https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=400&h=300&fit=crop&q=80',
    '一島一リゾートの究極プライベート': 'https://images.unsplash.com/photo-1499756630622-6a7fd76720ab?w=400&h=300&fit=crop&q=80',
    '色とりどりのサンゴ礁とダイビング': 'https://images.unsplash.com/photo-1581996645667-0d2ca0ff4d73?w=400&h=300&fit=crop&q=80',
    '水上ヴィラでの非日常体験': 'https://images.unsplash.com/photo-1566073771259-6a8506099945?w=400&h=300&fit=crop&q=80'
  },
  'iceland': {
    '世界最大の露天温泉ブルーラグーン': 'https://images.unsplash.com/photo-1539650116574-75c0c6d5d9b7?w=400&h=300&fit=crop&q=80',
    '迫力満点のガイザー間欠泉': 'https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=400&h=300&fit=crop&q=80',
    '幻想的なオーロラ観測': 'https://images.unsplash.com/photo-1531366936337-7c912a4589a7?w=400&h=300&fit=crop&q=80',
    '氷河と火山の雄大な絶景': 'https://images.unsplash.com/photo-1519904981063-b0cf448d479e?w=400&h=300&fit=crop&q=80'
  }
};

const fallbackImage = 'https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=400&h=300&fit=crop&q=80';

export function getHighlightImage(title: string, countryCode: string): string {
  return highlightImages[countryCode]?.[title] || fallbackImage;
}