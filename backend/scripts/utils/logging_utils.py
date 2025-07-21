"""
ログ・進捗表示共通ユーティリティ
"""

from datetime import datetime
from typing import Optional

class ProgressLogger:
    """統一された進捗表示クラス"""
    
    def __init__(self, total_items: int = 0):
        self.total_items = total_items
        self.completed_items = 0
        self.failed_items = 0
    
    def log(self, message: str, level: str = "INFO"):
        """統一フォーマットでログ出力"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        symbols = {
            "INFO": "ℹ️",
            "SUCCESS": "✅", 
            "WARNING": "⚠️",
            "ERROR": "❌",
            "PROGRESS": "🔄"
        }
        symbol = symbols.get(level, "📝")
        print(f"[{timestamp}] {symbol} {message}")
    
    def log_progress(self, message: str, completed: Optional[int] = None):
        """進捗付きログ出力"""
        if completed is not None:
            self.completed_items = completed
        
        if self.total_items > 0:
            percentage = (self.completed_items / self.total_items) * 100
            progress_msg = f"{message} ({self.completed_items}/{self.total_items} - {percentage:.1f}%)"
        else:
            progress_msg = message
        
        self.log(progress_msg, "PROGRESS")