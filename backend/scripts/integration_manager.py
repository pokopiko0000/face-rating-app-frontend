"""
既存システム統合管理スクリプト
生成されたデータを既存のフロントエンドシステムに統合
"""

import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import sys

# 親ディレクトリをパスに追加
sys.path.append(str(Path(__file__).parent.parent))


class IntegrationManager:
    """既存システム統合管理クラス"""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent.parent
        self.data_dir = self.base_dir / "data"
        self.frontend_dir = self.base_dir.parent / "project"
        self.frontend_data_dir = self.frontend_dir / "src" / "data"
        self.frontend_types_dir = self.frontend_dir / "src" / "types"
        
        # バックアップディレクトリ
        self.backup_dir = self.data_dir / "backups"
        self.backup_dir.mkdir(exist_ok=True)
        
        # 統合対象ファイル
        self.target_files = {
            "countries.ts": self.frontend_data_dir / "countries.ts",
            "countryImages.ts": self.frontend_data_dir / "countryImages.ts",
            "country.ts": self.frontend_types_dir / "country.ts"
        }
    
    def create_backup(self):
        """既存ファイルのバックアップを作成"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_subdir = self.backup_dir / f"backup_{timestamp}"
        backup_subdir.mkdir(exist_ok=True)
        
        print(f"📁 バックアップを作成中: {backup_subdir}")
        
        for file_name, file_path in self.target_files.items():
            if file_path.exists():
                backup_file = backup_subdir / file_name
                shutil.copy2(file_path, backup_file)
                print(f"  ✅ {file_name} -> {backup_file}")
        
        # バックアップ情報ファイル
        backup_info = {
            "timestamp": timestamp,
            "backup_path": str(backup_subdir),
            "files_backed_up": [f for f, p in self.target_files.items() if p.exists()]
        }
        
        with open(backup_subdir / "backup_info.json", 'w', encoding='utf-8') as f:
            json.dump(backup_info, f, ensure_ascii=False, indent=2)
        
        print(f"✅ バックアップ完了: {len(backup_info['files_backed_up'])} files")
        return backup_subdir
    
    def update_country_types(self):
        """country.ts型定義を更新"""
        print("🔄 TypeScript型定義を更新中...")
        
        # 共有型定義をコピー
        shared_types_file = self.base_dir.parent / "shared" / "types" / "country.ts"
        target_types_file = self.target_files["country.ts"]
        
        if shared_types_file.exists():
            # フロントエンド専用の型定義に調整
            with open(shared_types_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # バックエンド専用の型を削除/調整
            frontend_content = content.replace(
                "// 統一国データスキーマ - バックエンドとフロントエンドで共通使用",
                "// 国データ型定義 - フロントエンド用"
            )
            
            # 不要な型を削除
            lines_to_remove = [
                "export interface CountryRawData",
                "export interface GenerationConfig",
                "export interface GenerationResult",
                "export interface ValidationResult"
            ]
            
            for line_start in lines_to_remove:
                # 複数行のinterface定義を削除
                lines = frontend_content.split('\n')
                filtered_lines = []
                skip_block = False
                brace_count = 0
                
                for line in lines:
                    if any(line.strip().startswith(remove_line) for remove_line in lines_to_remove):
                        skip_block = True
                        brace_count = 0
                    
                    if skip_block:
                        brace_count += line.count('{') - line.count('}')
                        if brace_count <= 0 and '}' in line:
                            skip_block = False
                        continue
                    
                    filtered_lines.append(line)
                
                frontend_content = '\n'.join(filtered_lines)
            
            with open(target_types_file, 'w', encoding='utf-8') as f:
                f.write(frontend_content)
            
            print("✅ TypeScript型定義更新完了")
        else:
            print("⚠️  共有型定義ファイルが見つかりません")
    
    def update_countries_data(self, data_file: Optional[str] = None):
        """countries.tsデータを更新"""
        print("🔄 国データファイルを更新中...")
        
        # 最新の生成データを取得
        if not data_file:
            # 最新の生成データファイルを自動検出
            generated_dir = self.data_dir / "generated"
            if generated_dir.exists():
                frontend_files = list(generated_dir.glob("frontend_country_data_*.json"))
                if frontend_files:
                    data_file = max(frontend_files, key=lambda x: x.stat().st_mtime)
                else:
                    print("❌ 生成されたデータファイルが見つかりません")
                    return
        
        if not Path(data_file).exists():
            print(f"❌ データファイルが見つかりません: {data_file}")
            return
        
        # データを読み込み
        with open(data_file, 'r', encoding='utf-8') as f:
            countries_data = json.load(f)
        
        # TypeScriptファイルを生成
        target_file = self.target_files["countries.ts"]
        
        content = """import { CountryDataMap } from '../types/country';

export const countryData: CountryDataMap = {
"""
        
        # 各国データを追加
        for country_code, data in countries_data.items():
            content += f"  '{country_code}': {{\n"
            content += f"    name: '{self._escape_string(data['name'])}',\n"
            content += f"    nameEn: '{self._escape_string(data['nameEn'])}',\n"
            content += f"    flag: '{data['flag']}',\n"
            content += f"    code: '{data['code']}',\n"
            content += f"    basic: {{\n"
            content += f"      capital: '{self._escape_string(data['basic']['capital'])}',\n"
            content += f"      population: '{self._escape_string(data['basic']['population'])}',\n"
            content += f"      language: '{self._escape_string(data['basic']['language'])}'\n"
            content += f"    }},\n"
            content += f"    coordinates: {{\n"
            content += f"      lat: {data['coordinates']['lat']},\n"
            content += f"      lng: {data['coordinates']['lng']}\n"
            content += f"    }},\n"
            content += f"    description: '{self._escape_string(data['description'])}',\n"
            content += f"    highlights: [\n"
            
            for highlight in data['highlights']:
                content += f"      {{\n"
                content += f"        title: '{self._escape_string(highlight['title'])}',\n"
                content += f"        description: '{self._escape_string(highlight['description'])}'\n"
                content += f"      }},\n"
            
            content += f"    ],\n"
            content += f"    whyVisit: '{self._escape_string(data['whyVisit'])}'\n"
            content += f"  }},\n\n"
        
        content += "};\n"
        
        # ファイルを書き込み
        with open(target_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"✅ 国データファイル更新完了: {len(countries_data)} countries")
    
    def update_country_images(self):
        """countryImages.tsを更新"""
        print("🔄 画像データファイルを更新中...")
        
        # 最新の画像データを取得
        images_data_file = self.data_dir / "country_images.json"
        
        if not images_data_file.exists():
            print("⚠️  画像データファイルが見つかりません")
            return
        
        with open(images_data_file, 'r', encoding='utf-8') as f:
            images_data = json.load(f)
        
        # TypeScriptファイルを生成
        target_file = self.target_files["countryImages.ts"]
        
        content = """// 各国の厳選された美しい画像URL（自動生成）
export const countryImages: Record<string, string> = {
"""
        
        for country_code, image_data in images_data.items():
            primary_url = image_data.get('primary', '')
            content += f'  "{country_code}": "{primary_url}",\n'
        
        content += """};\n\n// フォールバック画像
export const getFallbackImage = (countryName: string): string => {
  return "https://images.unsplash.com/photo-1469474968028-56623f02e42e?w=1200&h=800&fit=crop&q=80";
};

// 画像取得関数
export const getCountryImage = (countryCode: string, countryName: string): string => {
  return countryImages[countryCode.toLowerCase()] || getFallbackImage(countryName);
};
"""
        
        with open(target_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"✅ 画像データファイル更新完了: {len(images_data)} countries")
    
    def _escape_string(self, text: str) -> str:
        """文字列をTypeScript用にエスケープ"""
        if not text:
            return ""
        return text.replace("'", "\\'").replace("\\", "\\\\")
    
    def verify_integration(self):
        """統合後の検証"""
        print("🔍 統合結果を検証中...")
        
        issues = []
        
        # ファイル存在チェック
        for file_name, file_path in self.target_files.items():
            if not file_path.exists():
                issues.append(f"Missing file: {file_name}")
            else:
                # ファイルサイズチェック
                size = file_path.stat().st_size
                if size == 0:
                    issues.append(f"Empty file: {file_name}")
                elif size < 1000:  # 1KB未満は疑わしい
                    issues.append(f"Suspiciously small file: {file_name} ({size} bytes)")
        
        # TypeScript構文チェック（簡易版）
        countries_file = self.target_files["countries.ts"]
        if countries_file.exists():
            with open(countries_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if "export const countryData" not in content:
                issues.append("countries.ts: Missing export statement")
            
            if content.count('{') != content.count('}'):
                issues.append("countries.ts: Mismatched braces")
        
        # 結果レポート
        if issues:
            print("❌ 統合検証で問題が発見されました:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print("✅ 統合検証完了: 問題なし")
        
        return len(issues) == 0
    
    def create_integration_report(self, backup_dir: Path):
        """統合レポートを作成"""
        print("📊 統合レポートを作成中...")
        
        # 国数カウント
        countries_file = self.target_files["countries.ts"]
        country_count = 0
        
        if countries_file.exists():
            with open(countries_file, 'r', encoding='utf-8') as f:
                content = f.read()
            country_count = content.count("':")  # 各国エントリをカウント
        
        # 画像数カウント
        images_file = self.target_files["countryImages.ts"]
        image_count = 0
        
        if images_file.exists():
            with open(images_file, 'r', encoding='utf-8') as f:
                content = f.read()
            image_count = content.count('": "')  # 各画像エントリをカウント
        
        # レポート内容
        report = {
            "integration_timestamp": datetime.now().isoformat(),
            "backup_location": str(backup_dir),
            "integration_summary": {
                "total_countries": country_count,
                "total_images": image_count,
                "updated_files": list(self.target_files.keys())
            },
            "file_status": {
                file_name: {
                    "exists": file_path.exists(),
                    "size": file_path.stat().st_size if file_path.exists() else 0,
                    "last_modified": file_path.stat().st_mtime if file_path.exists() else 0
                }
                for file_name, file_path in self.target_files.items()
            }
        }
        
        # レポートファイル保存
        report_file = self.data_dir / f"integration_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 統合レポート作成完了: {report_file}")
        return report
    
    def run_integration(self, data_file: Optional[str] = None):
        """統合プロセスを実行"""
        print("🔄 既存システム統合を開始します")
        print("=" * 50)
        
        try:
            # Step 1: バックアップ作成
            backup_dir = self.create_backup()
            
            # Step 2: TypeScript型定義更新
            self.update_country_types()
            
            # Step 3: 国データ更新
            self.update_countries_data(data_file)
            
            # Step 4: 画像データ更新
            self.update_country_images()
            
            # Step 5: 統合検証
            verification_passed = self.verify_integration()
            
            # Step 6: 統合レポート作成
            report = self.create_integration_report(backup_dir)
            
            # 完了メッセージ
            if verification_passed:
                print("\n🎉 システム統合が正常に完了しました！")
                print(f"   統合国数: {report['integration_summary']['total_countries']}")
                print(f"   画像数: {report['integration_summary']['total_images']}")
                print(f"   バックアップ: {backup_dir}")
            else:
                print("\n⚠️  統合は完了しましたが、問題が検出されました。")
                print("   詳細は上記の検証結果を確認してください。")
            
        except Exception as e:
            print(f"\n❌ 統合プロセスでエラーが発生しました: {str(e)}")
            print("   バックアップから復元してください。")
            raise
    
    def restore_from_backup(self, backup_dir: str):
        """バックアップから復元"""
        print(f"🔄 バックアップから復元中: {backup_dir}")
        
        backup_path = Path(backup_dir)
        if not backup_path.exists():
            print(f"❌ バックアップディレクトリが見つかりません: {backup_dir}")
            return
        
        backup_info_file = backup_path / "backup_info.json"
        if backup_info_file.exists():
            with open(backup_info_file, 'r', encoding='utf-8') as f:
                backup_info = json.load(f)
            
            for file_name in backup_info['files_backed_up']:
                backup_file = backup_path / file_name
                target_file = self.target_files[file_name]
                
                if backup_file.exists():
                    shutil.copy2(backup_file, target_file)
                    print(f"✅ 復元完了: {file_name}")
        
        print("🎉 バックアップからの復元が完了しました")


def main():
    """メイン関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Integration Manager")
    parser.add_argument("--data-file", help="Custom data file path")
    parser.add_argument("--restore", help="Restore from backup directory")
    
    args = parser.parse_args()
    
    manager = IntegrationManager()
    
    if args.restore:
        manager.restore_from_backup(args.restore)
    else:
        manager.run_integration(args.data_file)


if __name__ == "__main__":
    main()