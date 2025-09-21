'use client';

import { useState } from 'react';
import { Slider } from '@/components/ui/slider';
import { Button } from '@/components/ui/button';

interface PainScaleProps {
  onSelect: (value: number) => void;
}

export function PainScale({ onSelect }: PainScaleProps) {
  const [value, setValue] = useState(5);
  const [submitted, setSubmitted] = useState(false);

  const handleSubmit = () => {
    setSubmitted(true);
    onSelect(value);
  };

  const getPainColor = (val: number) => {
    const hue = 120 - val * 12; // 120 (green) to 0 (red)
    return `hsl(${hue}, 80%, 50%)`;
  };

  return (
    <div className="w-full max-w-sm mx-auto p-4 space-y-6">
      <div className="relative">
        <Slider
          value={[value]}
          min={1}
          max={10}
          step={1}
          onValueChange={(values) => setValue(values[0])}
          disabled={submitted}
        />
        <style jsx global>{`
          .relative .bg-primary {
            background: linear-gradient(to right, hsl(120, 70%, 70%), hsl(174, 63%, 47%));
          }
        `}</style>
      </div>
      <div className="flex items-center justify-center gap-4">
        <div className="text-4xl font-bold w-12 text-center" style={{ color: getPainColor(value) }}>
          {value}
        </div>
        <Button onClick={handleSubmit} disabled={submitted}>
          Confirm
        </Button>
      </div>
    </div>
  );
}
