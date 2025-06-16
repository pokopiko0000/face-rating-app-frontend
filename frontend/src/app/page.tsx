'use client';

import { useState } from 'react';
import GenderSelector from '../components/GenderSelector';
import ImageUploader from '../components/ImageUploader';
import ResultsDisplay from '../components/ResultsDisplay';

type Result = {
  representative_name: string;
  country: string;
  similarity: number;
  image_path: string;
};

export default function Home() {
  const [gender, setGender] = useState<string>('male');
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [result, setResult] = useState<Result | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const handleGenderChange = (selectedGender: string) => {
    setGender(selectedGender);
  };

  const handleImageChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setImageFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setResult(null);
      setError(null);
    }
  };

  const handleSubmit = async () => {
    if (!imageFile) {
      setError('画像を選択してください。');
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    const formData = new FormData();
    formData.append('file', imageFile);
    formData.append('gender', gender);

    try {
      const response = await fetch('http://localhost:8003/compare_face', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('サーバーでエラーが発生しました。');
      }

      const data = await response.json();
      
      const imageUrl = `http://localhost:8003/${data.image_path}`;
      setResult({ ...data, image_path: imageUrl });

    } catch (err) {
      setError(err instanceof Error ? err.message : '不明なエラーが発生しました。');
    } finally {
      setLoading(false);
    }
  };
  
  const shareOnTwitter = () => {
    if (!result) return;
    const score = (result.similarity * 100).toFixed(1);
    const text = `診断結果、私は${result.country}の${result.representative_name}に${score}%似ているそうです！ #似ている有名人診断`;
    const url = "https://example.com"; // TODO: あとでアプリのURLに変更する
    const twitterUrl = `https://twitter.com/intent/tweet?text=${encodeURIComponent(text)}&url=${encodeURIComponent(url)}`;
    window.open(twitterUrl, '_blank');
  };

  return (
    <div className="container mx-auto p-4 text-center">
      <h1 className="text-3xl font-bold my-8">似ている有名人診断</h1>

      <GenderSelector gender={gender} onGenderChange={handleGenderChange} />
      
      <ImageUploader previewUrl={previewUrl} onImageChange={handleImageChange} />

      <button
        onClick={handleSubmit}
        disabled={loading || !imageFile}
        className="bg-green-500 text-white px-6 py-2 rounded disabled:bg-gray-400"
      >
        {loading ? '診断中...' : '診断する'}
      </button>

      {error && <p className="text-red-500 mt-4">{error}</p>}

      <ResultsDisplay result={result} onShare={shareOnTwitter} />
    </div>
  );
}
