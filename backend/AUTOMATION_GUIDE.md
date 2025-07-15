# 国データ自動生成システム 使用ガイド

このガイドでは、243カ国すべてのデータを自動生成するシステムの使用方法を説明します。

## 📋 概要

このシステムは以下のコンポーネントで構成されています：

1. **統一データスキーマ** - バックエンドとフロントエンドで共通のデータ構造
2. **基本情報自動取得** - REST Countries APIから国の基本情報を取得
3. **画像URL自動生成** - Unsplash APIから各国の代表的な画像を取得
4. **コンテンツ自動生成** - テンプレートベースで魅力的な説明文を生成
5. **品質管理システム** - 生成されたデータの品質チェックと修正
6. **統合管理** - 既存のフロントエンドシステムとの統合

## 🚀 クイックスタート

### 1. 依存関係のインストール

```bash
cd backend
pip install -r requirements.txt
```

### 2. 全自動実行（推奨）

```bash
cd backend/scripts
python master_data_generator.py
```

### 3. 既存システムとの統合

```bash
python integration_manager.py
```

## 🔧 詳細な実行手順

### Phase 1: 基本データ生成

#### 1.1 基本情報の取得

```bash
cd backend/scripts
python enhanced_country_data_generator.py
```

出力ファイル: `backend/data/country_raw_data.json`

#### 1.2 画像URLの生成

```bash
python image_url_generator.py
```

出力ファイル: `backend/data/country_images.json`

#### 1.3 コンテンツの生成

```bash
python content_generator.py
```

出力ファイル: `backend/data/country_contents.json`

### Phase 2: 統合と品質管理

#### 2.1 データ統合

```bash
python master_data_generator.py
```

出力ファイル: 
- `backend/data/generated/complete_country_data_YYYYMMDD_HHMMSS.json`
- `backend/data/generated/frontend_country_data_YYYYMMDD_HHMMSS.json`

#### 2.2 品質チェック

```bash
python data_quality_manager.py
```

出力ファイル: 
- `backend/data/quality_reports/quality_report_YYYYMMDD_HHMMSS.md`
- `backend/data/quality_reports/quality_report_YYYYMMDD_HHMMSS.json`

### Phase 3: システム統合

#### 3.1 既存システムとの統合

```bash
python integration_manager.py
```

更新されるファイル:
- `project/src/data/countries.ts`
- `project/src/data/countryImages.ts`
- `project/src/types/country.ts`

#### 3.2 統合結果の確認

```bash
cd ../../project
npm run typecheck
npm run lint
```

## 🔑 オプション設定

### Unsplash API キー（推奨）

より良い画像品質のために、Unsplash APIキーを設定：

```bash
python master_data_generator.py --unsplash-key YOUR_UNSPLASH_ACCESS_KEY
```

### カスタム設定

```bash
python master_data_generator.py \
  --unsplash-key YOUR_KEY \
  --batch-size 5 \
  --retry-count 3
```

## 📊 出力ファイル構造

### 完全データセット (`complete_country_data_*.json`)

```json
{
  "jp": {
    "name": "日本",
    "nameEn": "Japan",
    "flag": "🇯🇵",
    "code": "jp",
    "basic": {
      "capital": "東京",
      "population": "1億2,500万人",
      "language": "日本語"
    },
    "coordinates": {
      "lat": 35.6762,
      "lng": 139.6503
    },
    "metadata": {
      "continent": "AS",
      "rarity": 2,
      "populationNumber": 125000000
    },
    "content": {
      "description": "...",
      "highlights": [...],
      "whyVisit": "..."
    },
    "images": {
      "primary": "https://...",
      "highlights": ["https://...", ...],
      "fallback": "https://..."
    },
    "lastUpdated": "2024-01-01T00:00:00.000Z"
  }
}
```

### フロントエンド用データセット (`frontend_country_data_*.json`)

```json
{
  "jp": {
    "name": "日本",
    "nameEn": "Japan",
    "flag": "🇯🇵",
    "code": "jp",
    "basic": {...},
    "coordinates": {...},
    "description": "...",
    "highlights": [...],
    "whyVisit": "..."
  }
}
```

## 🔍 品質管理

### 品質基準

- **Description**: 50-250文字
- **Highlights**: 正確に4つ、各30-150文字
- **WhyVisit**: 15-60文字
- **座標**: 有効な緯度・経度範囲
- **画像**: 有効なHTTPS URL

### 品質レポート例

```
# データ品質レポート
生成日時: 2024-01-01T00:00:00.000Z

## 概要
- 総国数: 243
- 有効な国: 240 (98.8%)
- 無効な国: 3 (1.2%)
- 警告のある国: 15 (6.2%)

## エラー統計
- Description too short: 2件
- Missing coordinates: 1件
```

## 🛠️ トラブルシューティング

### 一般的な問題

#### 1. API Rate Limit エラー

```bash
# 解決方法: バッチサイズを小さくする
python master_data_generator.py --batch-size 3
```

#### 2. 画像取得エラー

```bash
# 解決方法: Unsplash APIキーを設定
python master_data_generator.py --unsplash-key YOUR_KEY
```

#### 3. 統合エラー

```bash
# 解決方法: バックアップから復元
python integration_manager.py --restore backend/data/backups/backup_YYYYMMDD_HHMMSS
```

### ログファイル

エラーの詳細は以下のファイルで確認できます：

- `backend/data/quality_reports/` - 品質レポート
- `backend/data/generated/generation_stats_*.json` - 生成統計
- `backend/data/integration_report_*.json` - 統合レポート

## 📈 パフォーマンス

### 実行時間目安

- **基本情報取得**: 約5分（243カ国）
- **画像URL生成**: 約10分（Unsplash API使用時）
- **コンテンツ生成**: 約2分
- **品質チェック**: 約1分
- **統合処理**: 約30秒

**合計**: 約18分（APIキー使用時）

### 最適化のヒント

1. **並列処理**: `--batch-size`を調整
2. **APIキー**: Unsplash API キーを設定
3. **キャッシュ**: 生成済みデータの再利用

## 🔒 セキュリティ

### APIキー管理

```bash
# 環境変数で管理（推奨）
export UNSPLASH_ACCESS_KEY="your_key_here"
python master_data_generator.py --unsplash-key $UNSPLASH_ACCESS_KEY
```

### データ保護

- すべてのデータはローカルに保存
- 自動バックアップ機能
- 品質チェック済みデータのみ統合

## 🎯 カスタマイズ

### コンテンツテンプレート

`backend/scripts/content_generator.py`の以下のメソッドを編集：

- `_define_description_templates()`
- `_define_highlight_templates()`
- `_define_why_visit_templates()`

### 画像検索キーワード

`backend/scripts/image_url_generator.py`の以下の辞書を編集：

- `country_search_keywords`

### 品質基準

`backend/scripts/data_quality_manager.py`の以下のメソッドを編集：

- `_define_quality_standards()`

## 🤝 サポート

問題が発生した場合：

1. **品質レポート**を確認
2. **バックアップ**から復元
3. **ログファイル**を確認
4. **段階的実行**でエラー箇所を特定

---

このシステムにより、手動で200時間かかる作業を約20分で完了できます。🎉