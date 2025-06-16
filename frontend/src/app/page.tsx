'use client';

import { useState, useRef } from 'react';

interface RankingResult {
  country: string;
  similarity: number;
  country_code: string;
}

interface ComparisonData {
  user_image: string;
  representative_image: string;
  country: string;
  similarity: number;
}

export default function Home() {
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string>('');
  const [selectedGender, setSelectedGender] = useState<'male' | 'female'>('female');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [results, setResults] = useState<RankingResult[]>([]);
  const [comparisonData, setComparisonData] = useState<ComparisonData | null>(null);
  const [error, setError] = useState<string>('');
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleImageSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setSelectedImage(file);
      const url = URL.createObjectURL(file);
      setPreviewUrl(url);
      setError('');
      setResults([]);
      setComparisonData(null);
    }
  };

  const handleDragOver = (event: React.DragEvent) => {
    event.preventDefault();
  };

  const handleDrop = (event: React.DragEvent) => {
    event.preventDefault();
    const file = event.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
      setSelectedImage(file);
      const url = URL.createObjectURL(file);
      setPreviewUrl(url);
      setError('');
      setResults([]);
      setComparisonData(null);
    }
  };

  const analyzeImage = async () => {
    if (!selectedImage) {
      setError('画像を選択してください');
      return;
    }

    setIsAnalyzing(true);
    setError('');

    try {
      const formData = new FormData();
      formData.append('file', selectedImage);
      formData.append('gender', selectedGender);

      const response = await fetch('http://localhost:8003/analyze', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('分析に失敗しました');
      }

      const data = await response.json();
      setResults(data.ranking);

      // 比較画像を取得
      if (data.ranking && data.ranking.length > 0) {
        const topCountry = data.ranking[0].country;
        const comparisonResponse = await fetch(
          `http://localhost:8003/comparison?country=${encodeURIComponent(
            topCountry
          )}&gender=${selectedGender}`
        );

        if (comparisonResponse.ok) {
          const comparisonBlob = await comparisonResponse.blob();
          const comparisonUrl = URL.createObjectURL(comparisonBlob);
          setComparisonData({
            user_image: previewUrl,
            representative_image: comparisonUrl,
            country: topCountry,
            similarity: data.ranking[0].similarity
          });
        }
      }
    } catch (err) {
      setError('分析中にエラーが発生しました: ' + (err as Error).message);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const shareToTwitter = () => {
    if (results.length > 0) {
      const topCountry = results[0].country;
      const similarity = (results[0].similarity * 100).toFixed(1);
      const text = `私の顔は${topCountry}の理想顔に${similarity}%似ているそうです！あなたも試してみませんか？`;
      const url = window.location.href;
      window.open(`https://twitter.com/intent/tweet?text=${encodeURIComponent(text)}&url=${encodeURIComponent(url)}`, '_blank');
    }
  };

  return (
    <div style={{
      minHeight: '100vh',
      background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
      padding: '20px'
    }}>
      <div style={{
        maxWidth: '800px',
        margin: '0 auto',
        backgroundColor: 'white',
        borderRadius: '20px',
        padding: '40px',
        boxShadow: '0 20px 40px rgba(0,0,0,0.1)'
      }}>
        {/* ヘッダー */}
        <div style={{ textAlign: 'center', marginBottom: '40px' }}>
          <h1 style={{
            fontSize: '36px',
            fontWeight: 'bold',
            color: '#2d3748',
            marginBottom: '10px'
          }}>
            🌍 どの国の理想顔？
          </h1>
          <p style={{
            fontSize: '18px',
            color: '#718096',
            lineHeight: '1.6'
          }}>
            AIがあなたの顔を分析し、どの国の理想顔に最も近いかを診断します
          </p>
        </div>

        {/* 性別選択 */}
        <div style={{ marginBottom: '30px', textAlign: 'center' }}>
          <h3 style={{ fontSize: '18px', marginBottom: '15px', color: '#2d3748' }}>性別を選択してください</h3>
          <div style={{ display: 'flex', justifyContent: 'center', gap: '20px' }}>
            <button
              onClick={() => setSelectedGender('female')}
              style={{
                padding: '12px 24px',
                borderRadius: '25px',
                border: '2px solid',
                borderColor: selectedGender === 'female' ? '#667eea' : '#e2e8f0',
                backgroundColor: selectedGender === 'female' ? '#667eea' : 'white',
                color: selectedGender === 'female' ? 'white' : '#4a5568',
                fontSize: '16px',
                cursor: 'pointer',
                transition: 'all 0.3s ease'
              }}
            >
              👩 女性
            </button>
            <button
              onClick={() => setSelectedGender('male')}
              style={{
                padding: '12px 24px',
                borderRadius: '25px',
                border: '2px solid',
                borderColor: selectedGender === 'male' ? '#667eea' : '#e2e8f0',
                backgroundColor: selectedGender === 'male' ? '#667eea' : 'white',
                color: selectedGender === 'male' ? 'white' : '#4a5568',
                fontSize: '16px',
                cursor: 'pointer',
                transition: 'all 0.3s ease'
              }}
            >
              👨 男性
            </button>
          </div>
        </div>

        {/* 画像アップロード */}
        <div style={{ marginBottom: '30px' }}>
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            onChange={handleImageSelect}
            style={{ display: 'none' }}
          />
          
          <div
            onClick={() => fileInputRef.current?.click()}
            onDragOver={handleDragOver}
            onDrop={handleDrop}
            style={{
              border: '3px dashed #cbd5e0',
              borderRadius: '15px',
              padding: '40px',
              textAlign: 'center',
              cursor: 'pointer',
              backgroundColor: '#f7fafc',
              transition: 'all 0.3s ease',
              minHeight: '200px',
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center'
            }}
          >
            {previewUrl ? (
              <div>
                <img
                  src={previewUrl}
                  alt="プレビュー"
                  style={{
                    maxWidth: '200px',
                    maxHeight: '200px',
                    borderRadius: '10px',
                    marginBottom: '15px'
                  }}
                />
                <p style={{ color: '#4a5568', fontSize: '16px' }}>
                  クリックして画像を変更
                </p>
              </div>
            ) : (
              <div>
                <div style={{ fontSize: '48px', marginBottom: '15px' }}>📸</div>
                <p style={{ fontSize: '18px', color: '#4a5568', marginBottom: '10px' }}>
                  画像をクリックまたはドラッグ&ドロップ
                </p>
                <p style={{ fontSize: '14px', color: '#a0aec0' }}>
                  JPG, PNG, GIF対応
                </p>
              </div>
            )}
          </div>
        </div>

        {/* 分析ボタン */}
        <div style={{ textAlign: 'center', marginBottom: '30px' }}>
          <button
            onClick={analyzeImage}
            disabled={!selectedImage || isAnalyzing}
            style={{
              padding: '15px 40px',
              fontSize: '18px',
              fontWeight: 'bold',
              color: 'white',
              backgroundColor: (!selectedImage || isAnalyzing) ? '#a0aec0' : '#667eea',
              border: 'none',
              borderRadius: '25px',
              cursor: (!selectedImage || isAnalyzing) ? 'not-allowed' : 'pointer',
              transition: 'all 0.3s ease',
              boxShadow: '0 4px 15px rgba(102, 126, 234, 0.4)'
            }}
          >
            {isAnalyzing ? '🔍 分析中...' : '✨ 分析開始'}
          </button>
        </div>

        {/* エラー表示 */}
        {error && (
          <div style={{
            backgroundColor: '#fed7d7',
            color: '#c53030',
            padding: '15px',
            borderRadius: '10px',
            marginBottom: '20px',
            textAlign: 'center'
          }}>
            {error}
          </div>
        )}

        {/* 比較画像表示 */}
        {comparisonData && (
          <div style={{
            backgroundColor: '#f0fff4',
            padding: '30px',
            borderRadius: '15px',
            marginBottom: '30px',
            border: '2px solid #68d391'
          }}>
            <h3 style={{
              fontSize: '24px',
              fontWeight: 'bold',
              color: '#2d3748',
              textAlign: 'center',
              marginBottom: '20px'
            }}>
              🏆 あなたは{comparisonData.country}の理想顔に最も似ています！
            </h3>
            <div style={{
              display: 'flex',
              justifyContent: 'center',
              alignItems: 'center',
              gap: '30px',
              flexWrap: 'wrap'
            }}>
              <div style={{ textAlign: 'center' }}>
                <p style={{ fontSize: '16px', marginBottom: '10px', color: '#4a5568' }}>あなたの顔</p>
                <img
                  src={comparisonData.user_image}
                  alt="あなたの顔"
                  style={{
                    width: '150px',
                    height: '150px',
                    objectFit: 'cover',
                    borderRadius: '10px',
                    border: '3px solid #667eea'
                  }}
                />
              </div>
              <div style={{
                fontSize: '24px',
                color: '#667eea',
                fontWeight: 'bold'
              }}>
                VS
              </div>
              <div style={{ textAlign: 'center' }}>
                <p style={{ fontSize: '16px', marginBottom: '10px', color: '#4a5568' }}>
                  {comparisonData.country}の理想顔
                </p>
                <img
                  src={comparisonData.representative_image}
                  alt={`${comparisonData.country}の理想顔`}
                  style={{
                    width: '150px',
                    height: '150px',
                    objectFit: 'cover',
                    borderRadius: '10px',
                    border: '3px solid #48bb78'
                  }}
                />
              </div>
            </div>
            <p style={{
              textAlign: 'center',
              fontSize: '18px',
              color: '#2d3748',
              marginTop: '20px',
              fontWeight: 'bold'
            }}>
              類似度: {(comparisonData.similarity * 100).toFixed(1)}%
            </p>
          </div>
        )}

        {/* ランキング結果 */}
        {results.length > 0 && (
          <div style={{ marginBottom: '30px' }}>
            <h3 style={{
              fontSize: '24px',
              fontWeight: 'bold',
              color: '#2d3748',
              textAlign: 'center',
              marginBottom: '25px'
            }}>
              📊 類似度ランキング TOP 10
            </h3>
            <div style={{ display: 'grid', gap: '15px' }}>
              {results.slice(0, 10).map((result, index) => (
                <div
                  key={result.country}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    padding: '15px',
                    backgroundColor: index === 0 ? '#fff5f5' : '#f7fafc',
                    borderRadius: '10px',
                    border: index === 0 ? '2px solid #f56565' : '1px solid #e2e8f0',
                    boxShadow: index === 0 ? '0 4px 15px rgba(245, 101, 101, 0.2)' : 'none'
                  }}
                >
                  <div style={{
                    fontSize: '24px',
                    fontWeight: 'bold',
                    color: index === 0 ? '#f56565' : '#4a5568',
                    minWidth: '40px',
                    textAlign: 'center'
                  }}>
                    {index + 1}
                  </div>
                  <div style={{
                    fontSize: '24px',
                    margin: '0 15px',
                    width: '32px',
                    height: '24px',
                  }}>
                    {result.country_code ? (
                      <img 
                        src={`https://flagcdn.com/${result.country_code.toLowerCase()}.svg`}
                        alt={`${result.country}の国旗`}
                        style={{ width: '100%', height: '100%', objectFit: 'cover', borderRadius: '4px' }}
                      />
                    ) : '🏳️'}
                  </div>
                  <div style={{ flex: 1 }}>
                    <div style={{
                      fontSize: '18px',
                      fontWeight: 'bold',
                      color: '#2d3748'
                    }}>
                      {result.country}
                    </div>
                  </div>
                  <div style={{
                    fontSize: '16px',
                    fontWeight: 'bold',
                    color: index === 0 ? '#f56565' : '#4a5568'
                  }}>
                    {(result.similarity * 100).toFixed(1)}%
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* SNS共有ボタン */}
        {results.length > 0 && (
          <div style={{ textAlign: 'center' }}>
            <button
              onClick={shareToTwitter}
              style={{
                padding: '12px 30px',
                fontSize: '16px',
                fontWeight: 'bold',
                color: 'white',
                backgroundColor: '#1da1f2',
                border: 'none',
                borderRadius: '25px',
                cursor: 'pointer',
                transition: 'all 0.3s ease',
                boxShadow: '0 4px 15px rgba(29, 161, 242, 0.4)'
              }}
            >
              🐦 Twitterで共有
            </button>
          </div>
        )}
      </div>
    </div>
  );
}