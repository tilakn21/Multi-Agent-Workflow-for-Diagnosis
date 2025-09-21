'use client';

import { useEffect, useState } from 'react';

export function SoundWave() {
  const [barHeights, setBarHeights] = useState<number[]>([]);

  useEffect(() => {
    const initialHeights = Array.from({ length: 60 }, () => Math.random() * 0.8 + 0.2);
    setBarHeights(initialHeights);

    const interval = setInterval(() => {
      setBarHeights(heights =>
        heights.map(() => Math.random() * 0.8 + 0.2)
      );
    }, 200);

    return () => clearInterval(interval);
  }, []);

  return (
    <div className="absolute inset-0 flex items-center justify-center gap-1 opacity-10 overflow-hidden">
      {barHeights.map((height, i) => (
        <div
          key={i}
          className="w-2 rounded-full bg-primary transition-all duration-200 ease-in-out"
          style={{
            height: `${height * 100}%`,
            animation: `waveform 1.5s ease-in-out infinite alternate`,
            animationDelay: `${i * 0.02}s`,
          }}
        />
      ))}
    </div>
  );
}
