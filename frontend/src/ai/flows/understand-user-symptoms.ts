'use server';
/**
 * @fileOverview This file defines a Genkit flow for understanding and categorizing user symptoms from free-text input.
 *
 * The flow takes a user's symptom description as input and returns a categorized understanding of the symptoms.
 * This is then used to proceed with more specific questions in a chatbot interface.
 *
 * @fileOverview
 * - `understandUserSymptoms`: The main function to start the symptom understanding flow.
 * - `UnderstandUserSymptomsInput`: The input type for the `understandUserSymptoms` function.
 * - `UnderstandUserSymptomsOutput`: The output type for the `understandUserSymptoms` function.
 */

import {ai} from '@/ai/genkit';
import {z} from 'genkit';

const UnderstandUserSymptomsInputSchema = z.object({
  symptomDescription: z.string().describe('The user-provided description of their symptoms.'),
});
export type UnderstandUserSymptomsInput = z.infer<typeof UnderstandUserSymptomsInputSchema>;

const UnderstandUserSymptomsOutputSchema = z.object({
  understoodSymptoms: z.string().describe('A categorized and detailed understanding of the user\'s symptoms.'),
});
export type UnderstandUserSymptomsOutput = z.infer<typeof UnderstandUserSymptomsOutputSchema>;

export async function understandUserSymptoms(input: UnderstandUserSymptomsInput): Promise<UnderstandUserSymptomsOutput> {
  return understandUserSymptomsFlow(input);
}

const prompt = ai.definePrompt({
  name: 'understandUserSymptomsPrompt',
  input: {schema: UnderstandUserSymptomsInputSchema},
  output: {schema: UnderstandUserSymptomsOutputSchema},
  prompt: `You are an AI health assistant. Your task is to understand the user's symptoms from the description they provide.

  Description: {{{symptomDescription}}}

  Provide a categorized summary of the symptoms described by the user so that the chatbot can proceed with more specific questions.`,
});

const understandUserSymptomsFlow = ai.defineFlow(
  {
    name: 'understandUserSymptomsFlow',
    inputSchema: UnderstandUserSymptomsInputSchema,
    outputSchema: UnderstandUserSymptomsOutputSchema,
  },
  async input => {
    const {output} = await prompt(input);
    return output!;
  }
);
