'use client';

import { Button } from '@/components/ui/button';
import { useState } from 'react';

interface ChoiceButtonsProps {
  options: string[];
  onSelect: (value: string) => void;
}

export function ChoiceButtons({ options, onSelect }: ChoiceButtonsProps) {
  const [selectedValue, setSelectedValue] = useState<string | null>(null);

  const handleSelect = (option: string) => {
    setSelectedValue(option);
    onSelect(option);
  };

  return (
    <div className="flex flex-wrap justify-center gap-3 p-4">
      {options.map((option) => (
        <Button
          key={option}
          variant={selectedValue === option ? 'default' : 'outline'}
          onClick={() => handleSelect(option)}
          disabled={!!selectedValue && selectedValue !== option}
          className="transition-all duration-300"
        >
          {option}
        </Button>
      ))}
    </div>
  );
}
