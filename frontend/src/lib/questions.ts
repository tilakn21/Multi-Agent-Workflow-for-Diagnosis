import type { Question } from './types';

export const questions: Question[] = [
  { id: 1, key: 'initial', text: "Welcome, Doctor. Please enter the patient's symptoms, relevant history, and any initial findings. You can type or use the microphone.", type: 'initial' },
  { id: 2, key: 'summary', text: "Thank you. Here is a summary of the case for your review:", type: 'summary' },
  { id: 3, key: 'final', text: "The analysis is complete. Remember that this information is for decision support and should not replace your professional clinical judgment. You can start a new case at any time.", type: 'final'}
];
