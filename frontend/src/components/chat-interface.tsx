"use client";

import { useState, useRef, useEffect } from "react";
import type { Message, Answers } from "@/lib/types";
import { questions } from "@/lib/questions";
import { useToast } from "@/hooks/use-toast";
import { useAudioRecorder } from "@/hooks/use-audio-recorder";
import { ChatMessages } from "@/components/chat-messages";
import { ChatInput } from "@/components/chat-input";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { RecordingOverlay } from "@/components/recording-overlay";
import { TranscriptionOverlay } from "@/components/transcription-overlay";
import { transcribeAudio } from "@/ai/flows/transcribe-audio";
import { useChatHistory } from "@/hooks/use-chat-history";
import { ArrowLeft } from "lucide-react";
import { useGetInitialState } from "@/logic/chat-interface-logic";
import {
  sendCaseToNgrok,
  handleAudioSubmit,
} from "@/logic/chat-interface-logic";
import { Dialog } from "@/components/ui/dialog";
import {
  ANALYZE_IMAGE_API,
  CASE_CREATION_API,
  DOWNLOAD_REPORT_API,
} from "@/helpers/consts";
import { X } from "lucide-react";

interface ChatInterfaceProps {
  conversationId: string;
  onNewChat: () => void;
  sendMessage: (conversationId: string, message: string) => Promise<void>;
}

export function ChatInterface({
  conversationId,
  onNewChat,
  sendMessage,
}: ChatInterfaceProps) {
  const { getConversation, saveConversation } = useChatHistory();

  // Use the extracted useGetInitialState function
  const getInitialState = useGetInitialState(conversationId);

  const [messages, setMessages] = useState<Message[]>(
    getInitialState().messages
  );
  const [currentQuestionIndex, setCurrentQuestionIndex] = useState(
    getInitialState().questionIndex
  );
  const [answers, setAnswers] = useState<Answers>(getInitialState().answers);
  const [isBotLoading, setIsBotLoading] = useState(false);
  const [isTranscribing, setIsTranscribing] = useState(false);
  const [isCaseLoading, setIsCaseLoading] = useState(false);
  const [imageDownloadResponse, setImageDownloadResponse] = useState(null);

  const { toast } = useToast();
  const bottomRef = useRef<HTMLDivElement>(null);
  const audioRecorder = useAudioRecorder();

  // Effect for saving conversation to history
  useEffect(() => {
    // Only save if there's more than the initial bot message
    if (messages.length > 1) {
      saveConversation(conversationId, messages, answers);
    }
  }, [messages, answers, conversationId, saveConversation]);

  const addMessage = (message: Omit<Message, "id">) => {
    setMessages((prev) => [
      ...prev,
      {
        id: Date.now().toString() + Math.random(),
        ...message,
        text: message.text || "", // Ensure text is always defined
        content: message.content || null, // Add support for content
      },
    ]);
  };

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Helper function to send case to ngrok backend with simplified headers
  async function sendCaseToNgrok(symptomDescription: string) {
    const url = CASE_CREATION_API;
    const payload = { symptomDescription };
    const headers = {
      "Content-Type": "application/json",
      "ngrok-skip-browser-warning": "true",
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Allow-Methods": "POST, OPTIONS",
      "Access-Control-Allow-Headers":
        "Content-Type, ngrok-skip-browser-warning",
    };

    try {
      const response = await fetch(url, {
        method: "POST",
        headers,
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Failed to send case to backend: ${errorText}`);
      }

      return await response.json();
    } catch (error) {
      console.error("Error in sendCaseToNgrok:", error);
      throw error;
    }
  }

  // Add image upload handler and API call for diagnosis
  async function sendImageForDiagnosis(base64Image: string) {
    const url = ANALYZE_IMAGE_API;
    const formData = new FormData();
    formData.append("image", base64Image);
    try {
      const response = await fetch(url, {
        method: "POST",
        body: formData,
        headers: {
          "ngrok-skip-browser-warning": "true",
          "Access-Control-Allow-Origin": "*",
          "Access-Control-Allow-Methods": "POST, OPTIONS",
          "Access-Control-Allow-Headers": "ngrok-skip-browser-warning",
        },
      });
      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Failed to send image to backend: ${errorText}`);
      }
      return await response.json();
    } catch (error) {
      console.error("Error in sendImageForDiagnosis:", error);
      throw error;
    }
  }

  const handleAudioSubmitWithLoader = async (
    audioDataUri: string,
    fileName: string | undefined
  ) => {
    addMessage({
      sender: "ui",
      content: fileName
        ? `Uploaded: ${fileName}`
        : `<audio controls src='${audioDataUri}' class='w-full'></audio>`,
    });
    setIsCaseLoading(true);
    try {
      // Transcribe audio
      const match = audioDataUri.match(/^data:(.*);base64,(.*)$/);
      if (!match) throw new Error("Invalid audio data URI format.");
      const [_, mimeType, audioData] = match;
      const { transcription } = await transcribeAudio({
        audioData,
        mimeType,
        language: "en",
      });
      addMessage({ sender: "ui", content: "" });
      if (transcription && transcription.trim()) {
        addMessage({ sender: "user", text: transcription });
        // Call backend API with transcribed text
        const backendResponse = await sendCaseToNgrok(transcription);
        setDownloadResponse(backendResponse);
        // Map the response to a detailed UI message
        const detailedMessage = (
          <div className="space-y-4">
            <p className="font-bold">Patient Problem:</p>
            <p>{backendResponse.final_diagnosis.patient_problem}</p>
            <p className="font-bold">Possible Diagnoses:</p>
            <ul className="list-disc list-inside">
              {backendResponse.final_diagnosis.diagnosis.map(
                (diag: string, index: number) => (
                  <li key={index}>{diag}</li>
                )
              )}
            </ul>
            <p className="font-bold">Simplified Diagnoses:</p>
            <ul className="list-disc list-inside">
              {backendResponse.final_diagnosis.diagnosis_simplified.map(
                (diag: string, index: number) => (
                  <li key={index}>{diag}</li>
                )
              )}
            </ul>
            <p className="font-bold">Metadata:</p>
            <ul className="list-disc list-inside">
              {Object.entries(backendResponse.final_diagnosis.metadata).map(
                ([key, value], index) => (
                  <li key={index}>
                    <strong>{key}:</strong>{" "}
                    {Array.isArray(value) ? value.join(", ") : String(value)}
                  </li>
                )
              )}
            </ul>
            <p className="font-bold">Treatment Plan:</p>
            <ul className="list-disc list-inside">
              <li>
                <strong>Medications:</strong>{" "}
                {backendResponse.final_diagnosis.treatment_plan.medications.join(
                  ", "
                )}
              </li>
              <li>
                <strong>Lifestyle Modifications:</strong>{" "}
                {backendResponse.final_diagnosis.treatment_plan.lifestyle_modifications.join(
                  ", "
                )}
              </li>
            </ul>
            <Button
              onClick={() => handleDownload(backendResponse)}
              variant="secondary"
            >
              Generate Report Link
            </Button>
          </div>
        );
        addMessage({ sender: "bot", content: detailedMessage });
      } else {
        addMessage({
          sender: "bot",
          text: "Transcription was empty. Please try again.",
        });
      }
    } catch (error) {
      console.error(error);
      toast({
        title: "Error",
        description: `Could not process audio or case. ${
          error instanceof Error ? error.message : "Unknown error"
        }`,
        variant: "destructive",
      });
    } finally {
      setIsCaseLoading(false);
    }
  };

  const handleFileUpload = (file: File) => {
    const reader = new FileReader();
    reader.readAsDataURL(file);
    reader.onloadend = () =>
      handleAudioSubmitWithLoader(reader.result as string, file.name);
    reader.onerror = (err) => {
      console.error("Error reading file:", err);
      toast({
        title: "Error",
        description: "Could not read the uploaded file.",
        variant: "destructive",
      });
    };
  };

  useEffect(() => {
    if (audioRecorder.audioDataUri && !audioRecorder.isRecording) {
      handleAudioSubmitWithLoader(audioRecorder.audioDataUri, undefined);
      audioRecorder.clearAudioData();
    }
  }, [audioRecorder.audioDataUri, audioRecorder.isRecording, audioRecorder]);

  const advanceQuestion = (newAnswers?: Answers) => {
    const nextIndex = currentQuestionIndex + 1;
    if (nextIndex < questions.length) {
      setCurrentQuestionIndex(nextIndex);
      const nextQuestion = questions[nextIndex];
      const message: Omit<Message, "id"> = { sender: "bot" };

      // Remove summary card logic
      message.text = nextQuestion.text;
      setTimeout(() => addMessage(message), 500);
    }
  };

  // Add a loader component to indicate loading state
  const Loader = () => (
    <div className="flex justify-center items-center h-full">
      <div className="loader" />
    </div>
  );

  // Loader overlay for full screen with animated status texts
  const LoaderOverlay = () => {
    const loadingSteps = [
      "Extracting patient history",
      "Searching in medical literature...",
      "Scraping articles from PubMed...",
      "Making embeddings...",
      "Searching FAISS...",
      "Searching relationships from graphRAG...",
      "Guardlines and verification...",
      "Retrieving response",
    ];
    const [step, setStep] = useState(0);

    useEffect(() => {
      if (step < loadingSteps.length - 1) {
        const timer = setTimeout(() => setStep(step + 1), 1500);
        return () => clearTimeout(timer);
      }
    }, [step]);

    return (
      <div className="fixed inset-0 z-50 flex flex-col items-center justify-center bg-black bg-opacity-90">
        <div className="loader mb-6" />
        <div className="text-white text-xl font-semibold animate-pulse">
          {loadingSteps[step]}
        </div>
      </div>
    );
  };

  // Update the handleSubmitInitial function to ensure the loader is displayed properly
  const handleSubmitInitial = async (symptomDescription: string) => {
    addMessage({ sender: "user", text: symptomDescription });
    setIsBotLoading(true);

    setMessages((prev) => {
      if (prev.some((m) => m.isLoading)) return prev;
      return [...prev, { sender: "bot", isLoading: true, id: "loading" }];
    });

    try {
      const backendResponse = await sendCaseToNgrok(symptomDescription);

      // Remove loader before showing the response
      setMessages((prev) => prev.filter((m) => !m.isLoading));

      // Map the response to a detailed UI message
      const detailedMessage = (
        <div className="space-y-4">
          <p className="font-bold">Patient Problem:</p>
          <p>{backendResponse.final_diagnosis.patient_problem}</p>
          <p className="font-bold">Possible Diagnoses:</p>
          <ul className="list-disc list-inside">
            {backendResponse.final_diagnosis.diagnosis.map(
              (diag: string, index: number) => (
                <li key={index}>{diag}</li>
              )
            )}
          </ul>
          <p className="font-bold">Simplified Diagnoses:</p>
          <ul className="list-disc list-inside">
            {backendResponse.final_diagnosis.diagnosis_simplified.map(
              (diag: string, index: number) => (
                <li key={index}>{diag}</li>
              )
            )}
          </ul>
          <p className="font-bold">Metadata:</p>
          <ul className="list-disc list-inside">
            {Object.entries(backendResponse.final_diagnosis.metadata).map(
              ([key, value], index) => (
                <li key={index}>
                  <strong>{key}:</strong>{" "}
                  {Array.isArray(value) ? value.join(", ") : String(value)}
                </li>
              )
            )}
          </ul>
          <p className="font-bold">Treatment Plan:</p>
          <ul className="list-disc list-inside">
            <li>
              <strong>Medications:</strong>{" "}
              {backendResponse.final_diagnosis.treatment_plan.medications.join(
                ", "
              )}
            </li>
            <li>
              <strong>Lifestyle Modifications:</strong>{" "}
              {backendResponse.final_diagnosis.treatment_plan.lifestyle_modifications.join(
                ", "
              )}
            </li>
          </ul>
          <div className="mt-4 flex justify-end"></div>
        </div>
      );

      addMessage({ sender: "bot", content: detailedMessage });
    } catch (error) {
      console.error(error);
      toast({
        title: "Error",
        description: "Could not process case. Please try again.",
        variant: "destructive",
      });
      setMessages((prev) => prev.filter((m) => !m.isLoading));
    } finally {
      setIsBotLoading(false);
    }
  };

  // Declare handleImageUpload before usage
  const handleImageUpload = (file: File) => {
    const reader = new FileReader();
    reader.readAsDataURL(file);
    reader.onloadend = async () => {
      const base64 = (reader.result as string).split(",")[1];
      addMessage({ sender: "ui", content: `Uploaded image: ${file.name}` });
      setIsCaseLoading(true);
      try {
        const backendResponse = await sendImageForDiagnosis(base64);

        // Map the response to a detailed UI message
        const detailedMessage = (
          <div className="space-y-4">
            <p className="font-bold">Image Diagnosis Result:</p>
            <p>{backendResponse.analysis}</p>
            {backendResponse.details && (
              <div>
                <p className="font-bold">Details:</p>
                <pre className="bg-gray-100 p-2 rounded text-sm whitespace-pre-wrap">
                  {JSON.stringify(backendResponse.details, null, 2)}
                </pre>
              </div>
            )}
            <div className="mt-4 flex justify-end"></div>
          </div>
        );
        addMessage({ sender: "bot", content: detailedMessage });
      } catch (error) {
        toast({
          title: "Error",
          description: `Could not process image. ${
            error instanceof Error ? error.message : "Unknown error"
          }`,
          variant: "destructive",
        });
      } finally {
        setIsCaseLoading(false);
      }
    };
    reader.onerror = (err) => {
      toast({
        title: "Error",
        description: "Could not read the uploaded image.",
        variant: "destructive",
      });
    };
  };

  const currentQuestion = questions[currentQuestionIndex];

  // Add popup state and handler
  const [showDownloadDialog, setShowDownloadDialog] = useState(false);
  const [medicinesInput, setMedicinesInput] = useState("");
  const [downloadLoading, setDownloadLoading] = useState(false);
  const [downloadError, setDownloadError] = useState("");
  const [downloadSuccess, setDownloadSuccess] = useState(false);
  const [downloadResponse, setDownloadResponse] = useState<any>(null);
  let lastBackendResponse: any = null;

  const handleDownload = (backendResponse: any) => {
    lastBackendResponse = backendResponse;
    setShowDownloadDialog(true);
  };

  const handleDownloadSubmit = async () => {
    setDownloadError("");

    try {
      const url = DOWNLOAD_REPORT_API;
      const payload = downloadResponse
        ? downloadResponse
        : imageDownloadResponse;
      const headers = {
        "Content-Type": "application/json",
        "ngrok-skip-browser-warning": "true",
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "POST, OPTIONS",
        "Access-Control-Allow-Headers":
          "Content-Type, ngrok-skip-browser-warning",
      };
      const response = await fetch(url, {
        method: "POST",
        headers,
        body: JSON.stringify(payload),
      });
      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Failed to download report: ${errorText}`);
      }
      const result = await response.json();
      setDownloadResponse(result);
      setDownloadSuccess(true);
    } catch (error) {
      setDownloadError(
        error instanceof Error ? error.message : "Unknown error"
      );
    } finally {
      setDownloadLoading(false);
    }
  };

  // Helper to add Download Report button to any backend response message
  function getDetailedMessageWithDownload(backendResponse: any) {
    return (
      <div className="space-y-4">
        <p className="font-bold">Patient Problem:</p>
        <p>
          {backendResponse.final_diagnosis?.patient_problem ||
            backendResponse.result}
        </p>
        {backendResponse.final_diagnosis?.diagnosis && (
          <>
            <p className="font-bold">Possible Diagnoses:</p>
            <ul className="list-disc list-inside">
              {backendResponse.final_diagnosis.diagnosis.map(
                (diag: string, index: number) => (
                  <li key={index}>{diag}</li>
                )
              )}
            </ul>
          </>
        )}
        {backendResponse.final_diagnosis?.diagnosis_simplified && (
          <>
            <p className="font-bold">Simplified Diagnoses:</p>
            <ul className="list-disc list-inside">
              {backendResponse.final_diagnosis.diagnosis_simplified.map(
                (diag: string, index: number) => (
                  <li key={index}>{diag}</li>
                )
              )}
            </ul>
          </>
        )}
        {backendResponse.final_diagnosis?.metadata && (
          <>
            <p className="font-bold">Metadata:</p>
            <ul className="list-disc list-inside">
              {Object.entries(backendResponse.final_diagnosis.metadata).map(
                ([key, value], index) => (
                  <li key={index}>
                    <strong>{key}:</strong>{" "}
                    {Array.isArray(value) ? value.join(", ") : String(value)}
                  </li>
                )
              )}
            </ul>
          </>
        )}
        {backendResponse.final_diagnosis?.treatment_plan && (
          <>
            <p className="font-bold">Treatment Plan:</p>
            <ul className="list-disc list-inside">
              <li>
                <strong>Medications:</strong>{" "}
                {backendResponse.final_diagnosis.treatment_plan.medications.join(
                  ", "
                )}
              </li>
              <li>
                <strong>Lifestyle Modifications:</strong>{" "}
                {backendResponse.final_diagnosis.treatment_plan.lifestyle_modifications.join(
                  ", "
                )}
              </li>
            </ul>
          </>
        )}
        {backendResponse.details && (
          <div>
            <p className="font-bold">Details:</p>
            <pre className="bg-gray-100 p-2 rounded text-sm whitespace-pre-wrap">
              {JSON.stringify(backendResponse.details, null, 2)}
            </pre>
          </div>
        )}
        <div className="mt-4 flex justify-end">
          <Button
            onClick={() => handleDownload(backendResponse)}
            variant="secondary"
          >
            Generate Report
          </Button>
        </div>
      </div>
    );
  }

  // Add helper to download image from URL
  function downloadImageFromUrl(url: string, filename: string = "report.png") {
    const link = document.createElement("a");
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  }

  return (
    <div className="flex flex-col flex-1 h-full min-h-0 bg-background">
      <RecordingOverlay
        isRecording={audioRecorder.isRecording}
        stopRecording={audioRecorder.stopRecording}
      />
      <TranscriptionOverlay isTranscribing={isTranscribing} />
      <header className="flex items-center gap-4 border-b bg-secondary/30 p-4">
        <Button
          variant="ghost"
          size="icon"
          onClick={onNewChat}
          className="h-9 w-9"
        >
          <ArrowLeft className="h-5 w-5" />
        </Button>
        <h2 className="text-lg font-semibold">Diagnostic Assistant</h2>
      </header>
      <ChatMessages messages={messages} />
      <div ref={bottomRef} />

      <div className="p-4 shrink-0 bg-background/0">
        {!isBotLoading && !isCaseLoading && currentQuestion && (
          <div className="animate-fade-in">
            {currentQuestion.type === "initial" && (
              <ChatInput
                onSubmit={handleSubmitInitial}
                onFileSubmit={handleFileUpload}
                onImageSubmit={handleImageUpload}
                isLoading={isBotLoading || isCaseLoading}
                audioRecorder={audioRecorder}
                handleDownloadSubmit={handleDownloadSubmit}
              />
            )}
            {currentQuestion.type === "final" && (
              <div className="text-center">
                <Button onClick={onNewChat}>Start New Case</Button>
              </div>
            )}
          </div>
        )}
      </div>
      {(isBotLoading || isCaseLoading) && <LoaderOverlay />}
      {showDownloadDialog && (
        <Dialog open={showDownloadDialog} onOpenChange={setShowDownloadDialog}>
          <div className="p-6 relative">
            <button
              className="absolute right-4 top-4 text-gray-500 hover:text-gray-700"
              onClick={() => setShowDownloadDialog(false)}
              aria-label="Close"
            >
              <X className="h-5 w-5" />
            </button>

            <Button
              onClick={handleDownloadSubmit}
              disabled={downloadLoading}
              style={{ alignItems: "center" }}
            >
              {downloadLoading ? "Downloading..." : "Generate Report"}
            </Button>
            {downloadError && (
              <p className="text-red-500 mt-2">{downloadError}</p>
            )}
            {downloadSuccess && downloadResponse && (
              <div className="mt-4">
                <p className="text-green-600">Report generated successfully!</p>
                {/* You can add logic to show/download the file here */}
                {downloadResponse.image_url && (
                  <div className="mt-4 flex flex-col items-center">
                    <p className="text-green-600 mb-2">
                      Report downloaded successfully!
                    </p>
                    <Button
                      onClick={() =>
                        downloadImageFromUrl(downloadResponse.image_url)
                      }
                      variant="outline"
                    >
                      Download Prescription Image
                    </Button>
                  </div>
                )}
              </div>
            )}
          </div>
        </Dialog>
      )}
    </div>
  );
}
