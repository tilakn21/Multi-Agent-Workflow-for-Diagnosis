export interface Message {
  id: string;
  sender: 'user' | 'bot' | 'system' | 'ui';
  text?: string;
  content?: React.ReactNode;
  isLoading?: boolean;
}

export type Question = {
  id: number;
  key: 'initial' | 'summary' | 'final';
  text: string;
  type: 'initial' | 'summary' | 'final';
  options?: string[];
};

export type Answers = {
  caseDetails?: string;
  analysis?: string;
};

export type Conversation = {
  id: string;
  title: string;
  messages: Message[];
  answers: Answers;
  createdAt: number;
  lastModified: number;
};
