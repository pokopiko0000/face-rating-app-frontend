import React from 'react';
import { LucideIcon } from 'lucide-react';

interface InfoCardProps {
  icon: LucideIcon;
  title: string;
  value: string;
  iconColor?: string;
}

export default function InfoCard({ icon: Icon, title, value, iconColor = 'text-purple-300' }: InfoCardProps) {
  return (
    <div className="flex items-center gap-3 bg-white/10 rounded-lg p-4">
      <Icon className={`w-5 h-5 ${iconColor} flex-shrink-0`} />
      <div>
        <div className="text-sm text-white/70">{title}</div>
        <div className="font-semibold text-white text-lg">{value}</div>
      </div>
    </div>
  );
}