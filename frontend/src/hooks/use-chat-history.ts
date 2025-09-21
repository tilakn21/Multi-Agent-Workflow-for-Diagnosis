"use client";

import { useState, useEffect, useCallback } from "react";
import type { Conversation, Message, Answers } from "@/lib/types";

const CHAT_HISTORY_KEY = "medimind-chat-history";

export function useChatHistory() {
  const [conversations, setConversations] = useState<Conversation[]>([]);

  useEffect(() => {
    try {
      const storedHistory = localStorage.getItem(CHAT_HISTORY_KEY);
      if (storedHistory) {
        const parsedHistory: Conversation[] = JSON.parse(storedHistory);
        // Sort by last modified date, newest first
        parsedHistory.sort((a, b) => b.lastModified - a.lastModified);
        setConversations(parsedHistory);
      }
    } catch (error) {
      console.error("Failed to load chat history from localStorage", error);
    }
  }, []);

  const saveToLocalStorage = (history: Conversation[]) => {
    try {
      localStorage.setItem(CHAT_HISTORY_KEY, JSON.stringify(history));
    } catch (error) {
      console.error("Failed to save chat history to localStorage", error);
    }
  };

  const getConversation = useCallback(
    (id: string): Conversation | undefined => {
      return conversations.find((c) => c.id === id);
    },
    [conversations]
  );

  const saveConversation = useCallback(
    (id: string, messages: Message[], answers: Answers) => {
      setConversations((prev) => {
        const existingConversationIndex = prev.findIndex((c) => c.id === id);
        let newConversations: Conversation[];

        if (existingConversationIndex > -1) {
          // Update existing conversation
          const updatedConversation = {
            ...prev[existingConversationIndex],
            messages,
            answers,
            lastModified: Date.now(),
          };
          newConversations = [...prev];
          newConversations[existingConversationIndex] = updatedConversation;
        } else {
          // Create new conversation
          const userMessage = messages.find(
            (m) =>
              m.sender === "user" &&
              typeof m.text === "string" &&
              m.text.trim() !== ""
          );
          const title = userMessage
            ? (userMessage.text as string).substring(0, 40) + "..."
            : "New Case";

          const newConversation: Conversation = {
            id,
            title,
            messages,
            answers,
            createdAt: Date.now(),
            lastModified: Date.now(),
          };
          newConversations = [newConversation, ...prev];
        }

        // Sort by last modified date, newest first
        newConversations.sort((a, b) => b.lastModified - a.lastModified);
        saveToLocalStorage(newConversations);
        return newConversations;
      });
    },
    []
  );

  const deleteConversation = useCallback((id: string) => {
    setConversations((prev) => {
      const newConversations = prev.filter((c) => c.id !== id);
      saveToLocalStorage(newConversations);
      return newConversations;
    });
  }, []);

  const addMessage = useCallback((id: string, message: Message) => {
    setConversations((prev) => {
      const conversationIndex = prev.findIndex((c) => c.id === id);
      if (conversationIndex === -1) return prev;

      const updatedConversation = {
        ...prev[conversationIndex],
        messages: [...prev[conversationIndex].messages, message],
        lastModified: Date.now(),
      };

      const newConversations = [...prev];
      newConversations[conversationIndex] = updatedConversation;
      saveToLocalStorage(newConversations);
      return newConversations;
    });
  }, []);

  const messages = useCallback(
    (id: string): Message[] => {
      const conversation = conversations.find((c) => c.id === id);
      return conversation ? conversation.messages : [];
    },
    [conversations]
  );

  return {
    conversations,
    getConversation,
    saveConversation,
    deleteConversation,
    addMessage,
    messages,
  };
}
