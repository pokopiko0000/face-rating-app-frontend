"""
ファイル操作共通ユーティリティ
"""

from pathlib import Path
import json
from typing import Dict, Any

def get_base_directories():
    """プロジェクト基本ディレクトリを取得"""
    base_dir = Path(__file__).parent.parent.parent
    return {
        'base': base_dir,
        'data': base_dir / 'data',
        'frontend': base_dir.parent / 'project'
    }

def ensure_directory(path: Path):
    """ディレクトリの存在確認・作成"""
    path.mkdir(parents=True, exist_ok=True)

def load_json_file(file_path: Path) -> Dict:
    """JSONファイルの安全な読み込み"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def save_json_file(data: Dict, file_path: Path):
    """JSONファイルの保存"""
    ensure_directory(file_path.parent)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)