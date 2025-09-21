"use client";

import { useState } from "react";
import { Logo } from "@/components/logo";
import { Button } from "@/components/ui/button";
import { MessageCircle, PlusCircle, Trash2 } from "lucide-react";
import { ChatInterface } from "@/components/chat-interface";
import { useChatHistory } from "@/hooks/use-chat-history";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from "@/components/ui/alert-dialog";
import { useToast } from "@/hooks/use-toast";

export default function ChatPage() {
  const { conversations, deleteConversation, addMessage, messages } =
    useChatHistory();
  const [selectedConversationId, setSelectedConversationId] = useState<
    string | null
  >(null);
  const { toast } = useToast();

  // Send message to ngrok API and update chat
  const sendMessage = async (conversationId: string, message: string) => {
    // Add a loading message
    const loadingId = Date.now().toString() + "-loading";
    addMessage(conversationId, {
      id: loadingId,
      sender: "bot",
      text: "Loading...",
    });
    try {
      const payload = { symptomDescription: message };
      const response = await fetch(
        "https://5796209a4fff.ngrok-free.app/compose_case",
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        }
      );
      if (!response.ok) throw new Error("Failed to fetch API response");
      const data = await response.json();
      addMessage(conversationId, {
        id: Date.now().toString(),
        sender: "bot",
        text: data.response || "No response received.",
      });
    } catch (error) {
      toast({
        title: "Error",
        description: "Failed to send message. Please try again.",
        variant: "destructive",
      });
      addMessage(conversationId, {
        id: Date.now().toString(),
        sender: "bot",
        text: "Error: Could not get response.",
      });
    }
  };

  const handleStartNewChat = () => {
    // Use a new timestamp for the key to ensure a fresh component instance
    setSelectedConversationId(`new_${Date.now()}`);
  };

  const handleDelete = (id: string) => {
    deleteConversation(id);
    if (selectedConversationId === id) {
      setSelectedConversationId(null);
    }
  };

  if (selectedConversationId) {
    return (
      <main className="h-screen w-full flex bg-background">
        <ChatInterface
          key={selectedConversationId}
          conversationId={selectedConversationId}
          onNewChat={() => setSelectedConversationId(null)}
          sendMessage={sendMessage}
        />
      </main>
    );
  }

  return (
    <main className="flex h-screen w-full flex-col items-center justify-center bg-background bg-blurry-gradient p-4 md:p-8">
      <div className="flex flex-col items-center text-center">
        <Logo className="mb-6 h-28 w-28" />
        <h1 className="text-4xl font-bold tracking-tighter text-foreground sm:text-5xl">
          Welcome, Doctor
        </h1>
        <p className="mt-4 max-w-md text-muted-foreground">
          Start a new case or review a previous one.
        </p>
      </div>

      <div className="mt-12 w-full max-w-2xl flex-1">
        <Card className="h-full flex flex-col bg-card/50 backdrop-blur-sm">
          <CardHeader className="flex flex-row items-center justify-between">
            <CardTitle>Recent Cases</CardTitle>
            <Button variant="outline" size="sm" onClick={handleStartNewChat}>
              <PlusCircle className="mr-2 h-4 w-4" />
              New Case
            </Button>
          </CardHeader>
          <CardContent className="flex-1 p-0">
            <ScrollArea className="h-full">
              {conversations.length === 0 ? (
                <div className="flex h-full items-center justify-center p-6 text-muted-foreground">
                  <p>No recent cases found.</p>
                </div>
              ) : (
                <div className="space-y-2 p-4 pt-0">
                  {conversations.map((convo) => (
                    <div
                      key={convo.id}
                      className="flex items-center justify-between rounded-lg border p-3 transition-colors hover:bg-accent"
                    >
                      <button
                        onClick={() => setSelectedConversationId(convo.id)}
                        className="flex-1 text-left"
                      >
                        <p className="font-semibold truncate">{convo.title}</p>
                        <p className="text-xs text-muted-foreground">
                          {new Date(convo.lastModified).toLocaleString()}
                        </p>
                      </button>
                      <AlertDialog>
                        <AlertDialogTrigger asChild>
                          <Button
                            variant="ghost"
                            size="icon"
                            className="h-8 w-8 text-muted-foreground hover:text-destructive"
                          >
                            <Trash2 className="h-4 w-4" />
                          </Button>
                        </AlertDialogTrigger>
                        <AlertDialogContent>
                          <AlertDialogHeader>
                            <AlertDialogTitle>Are you sure?</AlertDialogTitle>
                            <AlertDialogDescription>
                              This will permanently delete this chat case. This
                              action cannot be undone.
                            </AlertDialogDescription>
                          </AlertDialogHeader>
                          <AlertDialogFooter>
                            <AlertDialogCancel>Cancel</AlertDialogCancel>
                            <AlertDialogAction
                              onClick={() => handleDelete(convo.id)}
                              className="bg-destructive hover:bg-destructive/90"
                            >
                              Delete
                            </AlertDialogAction>
                          </AlertDialogFooter>
                        </AlertDialogContent>
                      </AlertDialog>
                    </div>
                  ))}
                </div>
              )}
            </ScrollArea>
          </CardContent>
        </Card>
      </div>

      <div className="mt-8 flex w-full max-w-md flex-col items-center">
        <Button
          size="lg"
          className="w-full rounded-full bg-primary/90 text-lg font-semibold shadow-lg backdrop-blur-sm transition-all hover:bg-primary/100 hover:shadow-xl"
          onClick={handleStartNewChat}
        >
          <MessageCircle className="mr-2" />
          Start a new case
        </Button>
        <p className="mt-2 text-xs text-muted-foreground">
          MediMind is a decision support tool and not a substitute for clinical
          judgment.
        </p>
      </div>
    </main>
  );
}
