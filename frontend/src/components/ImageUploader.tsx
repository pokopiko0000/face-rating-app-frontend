type Props = {
  previewUrl: string | null;
  onImageChange: (e: React.ChangeEvent<HTMLInputElement>) => void;
};

export default function ImageUploader({ previewUrl, onImageChange }: Props) {
  return (
    <div className="mb-4">
      <input type="file" accept="image/*" onChange={onImageChange} className="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-violet-50 file:text-violet-700 hover:file:bg-violet-100" />
      {previewUrl && <img src={previewUrl} alt="Preview" className="mt-4 mx-auto h-64" />}
    </div>
  );
} 