'use client';

import { Loader } from '@/components/ui/loader';

interface TranscriptionOverlayProps {
  isTranscribing: boolean;
}

export function TranscriptionOverlay({ isTranscribing }: TranscriptionOverlayProps) {
  if (!isTranscribing) {
    return null;
  }

  return (
    <div
      className="fixed inset-0 z-50 flex flex-col items-center justify-center bg-background/90 backdrop-blur-sm animate-fade-in"
      aria-modal="true"
      role="dialog"
    >
      <Loader />
      <p className="mt-4 text-lg font-semibold text-foreground">
        Transcribing audio...
      </p>
    </div>
  );
}
