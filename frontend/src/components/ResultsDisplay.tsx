type Result = {
  representative_name: string;
  country: string;
  similarity: number;
  image_path: string;
};

type Props = {
  result: Result | null;
  onShare: () => void;
};

export default function ResultsDisplay({ result, onShare }: Props) {
  if (!result) return null;

  return (
    <div className="mt-8 border p-4 rounded">
      <h2 className="text-2xl font-bold mb-4">診断結果</h2>
      <img src={result.image_path} alt={result.representative_name} className="mx-auto h-64 mb-4" />
      <p className="text-xl">あなたは <span className="font-bold">{result.country}</span> の <span className="font-bold">{result.representative_name}</span> に似ています！</p>
      <p className="text-lg">類似度: <span className="font-bold">{(result.similarity * 100).toFixed(1)}%</span></p>
      <button onClick={onShare} className="mt-4 bg-black text-white px-4 py-2 rounded">
        Xでシェアする
      </button>
    </div>
  );
} 