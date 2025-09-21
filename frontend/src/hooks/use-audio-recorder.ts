'use client';

import { useState, useRef, useEffect, useCallback } from 'react';

export type UseAudioRecorder = {
  isRecording: boolean;
  startRecording: () => void;
  stopRecording: () => void;
  audioDataUri: string | null;
  clearAudioData: () => void;
};

export function useAudioRecorder(): UseAudioRecorder {
  const [isRecording, setIsRecording] = useState(false);
  const [audioDataUri, setAudioDataUri] = useState<string | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const streamRef = useRef<MediaStream | null>(null);

  const stopStream = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
  }, []);

  const startRecording = async () => {
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
      try {
        stopStream(); // Stop any existing stream
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        streamRef.current = stream;

        mediaRecorderRef.current = new MediaRecorder(stream, { mimeType: 'audio/webm' });
        audioChunksRef.current = [];
        setAudioDataUri(null); // Clear previous recording

        mediaRecorderRef.current.ondataavailable = (event) => {
          if (event.data.size > 0) {
            audioChunksRef.current.push(event.data);
          }
        };

        mediaRecorderRef.current.onstop = () => {
          const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
          const reader = new FileReader();
          reader.readAsDataURL(audioBlob);
          reader.onloadend = () => {
            setAudioDataUri(reader.result as string);
          };
          
          stopStream();
          setIsRecording(false);
        };

        mediaRecorderRef.current.start();
        setIsRecording(true);
      } catch (error) {
        console.error('Error accessing microphone:', error);
        alert('Could not access the microphone. Please check your browser permissions.');
        stopStream();
        setIsRecording(false);
      }
    } else {
        alert('Audio recording is not supported by your browser.');
    }
  };

  const stopRecording = () => {
      if (mediaRecorderRef.current && isRecording) {
        mediaRecorderRef.current.stop();
        // The onstop event handler will set isRecording to false and stop the stream
      } else {
        stopStream();
        setIsRecording(false);
      }
  };

  const clearAudioData = useCallback(() => {
    setAudioDataUri(null);
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopStream();
    };
  }, [stopStream]);

  return {
    isRecording,
    startRecording,
    stopRecording,
    audioDataUri,
    clearAudioData,
  };
}
