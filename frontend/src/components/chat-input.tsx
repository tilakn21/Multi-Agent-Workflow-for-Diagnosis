"use client";

import { useState, useEffect, useRef } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Mic, Send, Paperclip, Image as ImageIcon } from "lucide-react";
import { cn } from "@/lib/utils";
import type { UseAudioRecorder } from "@/hooks/use-audio-recorder";

interface ChatInputProps {
  onSubmit: (value: string, audioDataUri?: string) => void;
  onFileSubmit: (file: File) => void;
  onImageSubmit?: (file: File) => void;
  isLoading: boolean;
  audioRecorder: UseAudioRecorder;
  handleDownloadSubmit: () => void;
}

export function ChatInput({
  onSubmit,
  onFileSubmit,
  onImageSubmit,
  isLoading,
  audioRecorder,
  handleDownloadSubmit,
}: ChatInputProps) {
  const [inputValue, setInputValue] = useState("");
  const { startRecording, stopRecording, isRecording } = audioRecorder;
  const fileInputRef = useRef<HTMLInputElement>(null);
  const imageInputRef = useRef<HTMLInputElement>(null);

  const handleTextSubmit = () => {
    if (inputValue.trim() && !isLoading) {
      onSubmit(inputValue);
      setInputValue("");
    }
  };

  const toggleRecording = () => {
    if (isRecording) {
      stopRecording();
    } else {
      setInputValue(""); // Clear input when starting recording
      startRecording();
    }
  };

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      onFileSubmit(file);
    }
  };

  const handleImageChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file && onImageSubmit) {
      onImageSubmit(file);
    }
  };

  return (
    <div className="flex w-full items-center gap-3 p-4">
      <div className="relative flex-1">
        <Input
          placeholder={
            isRecording
              ? "Listening..."
              : "Enter patient symptoms and history..."
          }
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          disabled={isLoading || isRecording}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              handleTextSubmit();
            }
          }}
          className="w-full rounded-full bg-secondary py-6 pl-5 pr-14 text-base"
        />
        {!isRecording && inputValue && (
          <Button
            type="button"
            size="icon"
            variant="ghost"
            className="absolute right-2 top-1/2 h-10 w-10 -translate-y-1/2 rounded-full"
            onClick={handleTextSubmit}
            disabled={isLoading}
          >
            <Send />
          </Button>
        )}
      </div>
      {/* Audio file upload */}
      <input
        type="file"
        ref={fileInputRef}
        onChange={handleFileChange}
        className="hidden"
        accept="audio/mp3,audio/wav,audio/webm,audio/ogg,audio/flac,audio/m4a"
        disabled={isLoading}
      />
      {/* Image file upload */}
      <input
        type="file"
        ref={imageInputRef}
        onChange={handleImageChange}
        className="hidden"
        accept="image/png,image/jpeg,image/jpg,image/webp"
        disabled={isLoading}
      />
      <Button
        type="button"
        size="icon"
        variant="outline"
        onClick={() => fileInputRef.current?.click()}
        className="h-14 w-14 flex-shrink-0 rounded-full"
        aria-label="Upload audio file"
        disabled={isLoading || isRecording}
      >
        <Paperclip className="h-6 w-6" />
      </Button>
      <Button
        type="button"
        size="icon"
        variant="outline"
        onClick={() => imageInputRef.current?.click()}
        className="h-14 w-14 flex-shrink-0 rounded-full"
        aria-label="Upload image file"
        disabled={isLoading}
      >
        <ImageIcon className="h-6 w-6" />
        <span className="sr-only">Upload image</span>
      </Button>
      <Button
        type="button"
        size="icon"
        variant="default"
        onClick={toggleRecording}
        className={cn(
          "relative h-14 w-14 flex-shrink-0 rounded-full transition-all duration-300",
          isRecording ? "bg-destructive scale-110 animate-glow" : "bg-primary"
        )}
        aria-label={isRecording ? "Stop recording" : "Start recording"}
        disabled={isLoading}
      >
        <Mic className={cn("z-10 h-6 w-6")} />
      </Button>
    </div>
  );
}
