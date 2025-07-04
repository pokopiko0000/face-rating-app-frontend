# AI顔診断 - 世界のどこでモテる？

AI技術を使って、あなたの顔がどの国で最も魅力的に見えるかを診断するWebアプリケーションです。

## 🚀 機能

- **AI顔分析**: InsightFaceを使った高精度な顔認識
- **250カ国対応**: 世界中の国々との顔の相性を診断
- **性別別診断**: 男女それぞれに最適化された分析
- **リアルタイム処理**: 画像をアップロードして即座に結果表示
- **プライバシー重視**: 画像は保存せず、処理後即座に削除

## 🛠 技術スタック

### フロントエンド
- **React 18** + **TypeScript**
- **Vite** (ビルドツール)
- **Tailwind CSS** (スタイリング)
- **React Router** (ルーティング)

### バックエンド
- **FastAPI** (Python)
- **InsightFace** (顔認識AI)
- **OpenCV** (画像処理)
- **NumPy** (数値計算)

### インフラ
- **Cloudflare R2** (画像配信)
- **Google AdSense** (広告)
- **Google Analytics** (分析)

## 📁 プロジェクト構造

```
/
├── README.md              # プロジェクト概要（このファイル）
├── CLAUDE.md              # AI開発指示書
├── REQUIREMENTS.md        # 要件定義書
├── plan.md               # 収益化計画
├── metadata.csv          # 国別顔データメタ情報
├── .gitignore            # Git除外設定
├── backend/              # バックエンドAPI
│   ├── main.py          # FastAPIメインサーバー
│   ├── crop_face.py     # 顔検出・切り抜き
│   ├── rank_face.py     # 顔類似度ランキング
│   ├── requirements.txt # Python依存関係
│   ├── yolov8n.pt      # YOLOモデル
│   ├── data/           # 事前処理済みデータ
│   │   ├── country_metadata.json
│   │   ├── country_prototypes_gender.npz
│   │   └── country_representatives_gender.json
│   └── scripts/        # データ生成スクリプト
└── project/            # フロントエンドアプリ
    ├── src/
    │   ├── components/ # Reactコンポーネント
    │   ├── services/   # API通信
    │   ├── types/      # TypeScript型定義
    │   └── utils/      # ユーティリティ
    ├── public/         # 静的ファイル
    └── package.json    # Node.js依存関係
```

## 🚀 開発環境セットアップ

### 前提条件
- Python 3.8+
- Node.js 18+
- npm または yarn

### バックエンドセットアップ
```bash
cd backend
pip install -r requirements.txt
python main.py
```

### フロントエンドセットアップ
```bash
cd project
npm install
npm run dev
```

## 🌐 デプロイ

### フロントエンド
- **Vercel** または **Netlify** 推奨

### バックエンド  
- **Render** または **Railway** 推奨

## 📊 アルゴリズム

1. **顔検出**: InsightFaceで顔の位置とランドマークを検出
2. **特徴抽出**: 512次元の顔埋め込みベクトルを生成
3. **類似度計算**: コサイン類似度で各国代表顔と比較
4. **ボーナス計算**: 地理的距離と希少性を考慮
5. **ランキング**: 最終スコアでトップ5を決定

## 🔒 プライバシー

- アップロードされた画像は処理後即座に削除
- 顔データや個人情報は一切保存しません
- Cookieは広告表示とアクセス解析のみに使用

## 📄 ライセンス

このプロジェクトは商用利用を目的としており、適切なライセンス下で運用されています。

## 🤝 貢献

バグ報告や機能要望は、Issueまたはお問い合わせページからご連絡ください。

---

**AI顔診断** - あなたの魅力が世界のどこで輝くかを発見しよう！