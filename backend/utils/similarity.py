"""
類似度計算ユーティリティ
"""

import numpy as np
from typing import Union

def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> Union[float, np.ndarray]:
    """
    2つのベクトル間のコサイン類似度を計算する
    
    Args:
        v1: 第1のベクトル
        v2: 第2のベクトル
        
    Returns:
        コサイン類似度（-1から1の範囲）
    """
    # ベクトルの正規化
    v1_normalized = v1 / np.linalg.norm(v1)
    v2_normalized = v2 / np.linalg.norm(v2)
    
    # コサイン類似度の計算
    similarity = np.dot(v1_normalized, v2_normalized)
    
    return similarity