import { useCallback } from "react";
import { questions } from "@/lib/questions";
import type { Message } from "@/lib/types";
import { useChatHistory } from "@/hooks/use-chat-history";
import { transcribeAudio } from "@/ai/flows/transcribe-audio";
import { toast } from "@/hooks/use-toast";
import React from "react";

/**
 * Initializes the state for the chat interface based on the conversation ID.
 * @param conversationId - The ID of the conversation.
 * @returns The initial state including messages, question index, and answers.
 */
export const useGetInitialState = (conversationId: string) => {
  const { getConversation } = useChatHistory();

  return useCallback(() => {
    if (!conversationId.startsWith("new_")) {
      const existingConversation = getConversation(conversationId);
      if (existingConversation && existingConversation.messages.length > 0) {
        const lastBotMessage = existingConversation.messages
          .slice()
          .reverse()
          .find((m) => m.sender === "bot");

        // Find the index of the last question asked
        const lastQuestion = questions.find(
          (q) => q.text === lastBotMessage?.text
        );
        const questionIndex = lastQuestion
          ? questions.indexOf(lastQuestion)
          : 0;

        return {
          messages: existingConversation.messages,
          questionIndex: questionIndex,
          answers: existingConversation.answers || {},
        };
      }
    }
    // Default state for a new chat
    return {
      messages: [
        {
          id: "initial-message",
          sender: "bot",
          text: questions[0].text,
        } as Message,
      ],
      questionIndex: 0,
      answers: {},
    };
  }, [conversationId, getConversation]);
};

/**
 * Sends a symptom description to the ngrok backend.
 * @param symptomDescription - The description of the symptoms.
 * @returns The backend response.
 */
export async function sendCaseToNgrok(symptomDescription: string) {
  const url = "https://5796209a4fff.ngrok-free.app/compose_case";
  const payload = { symptomDescription };
  const headers = {
    "Content-Type": "application/json",
    "ngrok-skip-browser-warning": "true",
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type, ngrok-skip-browser-warning",
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

/**
 * Handles audio submission by transcribing the audio and sending the message.
 * @param audioDataUri - The audio data URI.
 * @param fileName - The name of the uploaded file (optional).
 * @param addMessage - Function to add a message to the chat.
 * @param sendMessage - Function to send a message.
 * @param conversationId - The ID of the conversation.
 */
export const handleAudioSubmit = async (
  audioDataUri: string,
  fileName: string | undefined,
  addMessage: (message: Omit<Message, "id">) => void,
  sendMessage: (conversationId: string, message: string) => Promise<void>,
  conversationId: string
) => {
  addMessage({
    sender: "ui",
    content: fileName
      ? `Uploaded: ${fileName}`
      : `<audio controls src='${audioDataUri}' class='w-full'></audio>`,
  });

  try {
    const match = audioDataUri.match(/^data:(.*);base64,(.*)$/);
    if (!match) throw new Error("Invalid audio data URI format.");

    const [_, mimeType, audioData] = match;
    const { transcription } = await transcribeAudio({
      audioData,
      mimeType,
      language: "en",
    });

    // Remove the UI placeholder for the audio element/file name
    addMessage({ sender: "ui", content: "" });

    if (transcription && transcription.trim()) {
      // Send the transcribed text using sendMessage
      await sendMessage(conversationId, transcription);
    } else {
      addMessage({
        sender: "bot",
        text: "Transcription was empty. Please try again.",
      });
    }
  } catch (error) {
    console.error("Error in handleAudioSubmit:", error);
    toast({
      title: "Error",
      description: `Could not transcribe audio. ${
        error instanceof Error ? error.message : "Unknown error"
      }`,
      variant: "destructive",
    });
    // Clean up UI placeholders on error
    addMessage({ sender: "ui", content: "" });
  }
};
