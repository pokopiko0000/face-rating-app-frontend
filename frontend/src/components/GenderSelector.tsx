type Props = {
  gender: string;
  onGenderChange: (gender: string) => void;
};

export default function GenderSelector({ gender, onGenderChange }: Props) {
  return (
    <div className="mb-4">
      <button
        className={`px-4 py-2 mr-2 rounded ${gender === 'male' ? 'bg-blue-500 text-white' : 'bg-gray-200'}`}
        onClick={() => onGenderChange('male')}
      >
        男性
      </button>
      <button
        className={`px-4 py-2 rounded ${gender === 'female' ? 'bg-pink-500 text-white' : 'bg-gray-200'}`}
        onClick={() => onGenderChange('female')}
      >
        女性
      </button>
    </div>
  );
} 