'use client';

import { Mic } from 'lucide-react';
import { SoundWave } from '@/components/sound-wave';

interface RecordingOverlayProps {
  isRecording: boolean;
  stopRecording: () => void;
}

export function RecordingOverlay({ isRecording, stopRecording }: RecordingOverlayProps) {
  if (!isRecording) {
    return null;
  }

  return (
    <div
      className="fixed inset-0 z-50 flex cursor-pointer flex-col items-center justify-center bg-background/90 backdrop-blur-sm animate-fade-in bg-blurry-gradient"
      aria-modal="true"
      role="dialog"
      onClick={stopRecording}
    >
      <div className="pointer-events-none absolute left-1/2 top-1/2 h-full w-full -translate-x-1/2 -translate-y-1/2 overflow-hidden">
        <SoundWave />
      </div>
      
      <div className="pointer-events-none relative flex h-64 w-64 items-center justify-center">
        {[...Array(3)].map((_, i) => (
          <div
            key={i}
            className="absolute h-full w-full rounded-full border border-primary/30"
            style={{
              animation: `ripple 2.5s cubic-bezier(0.4, 0, 0.2, 1) infinite`,
              animationDelay: `${i * 0.5}s`,
            }}
          />
        ))}
        <Mic className="h-16 w-16 text-primary" />
      </div>

      <div className="pointer-events-none z-10 mt-8 flex flex-col items-center text-center text-foreground">
        <p className="text-2xl font-semibold">Listening...</p>
        <p className="mt-2 text-muted-foreground">
          Click anywhere to stop recording.
        </p>
      </div>
    </div>
  );
}
