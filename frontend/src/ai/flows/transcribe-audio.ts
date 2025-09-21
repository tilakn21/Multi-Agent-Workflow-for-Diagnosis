'use server';
/**
 * @fileOverview A Genkit flow for transcribing audio data using Groq's Whisper API.
 *
 * This flow takes base64-encoded audio data, saves it to a temporary file, streams it to Groq for transcription,
 * and returns the transcribed text. Supports English ('en'), Hindi ('hi'), and Marathi ('mr'), with optional language
 * specification for auto-detection in mixed-language audio.
 *
 * - transcribeAudio - A function to trigger the audio transcription flow.
 * - TranscribeAudioInput - The input type for the transcribeAudio function.
 * - TranscribeAudioOutput - The return type for the transcribeAudio function.
 */

import { ai } from '@/ai/genkit';
import { z } from 'genkit';
import Groq from 'groq-sdk';
import fs from 'fs';
import path from 'path';
import os from 'os';

// Define allowed languages (ISO-639-1 codes)
const allowedLanguages = ['en', 'hi', 'mr'] as const;

const TranscribeAudioInputSchema = z.object({
  audioData: z
    .string()
    .describe(
      'The base64-encoded audio data. The data URI prefix (e.g., "data:audio/webm;base64,") should be removed.'
    ),
  mimeType: z.string().describe('The MIME type of the audio data (e.g., "audio/webm").'),
  language: z.enum(allowedLanguages).optional().describe('Optional language code for transcription (en: English, hi: Hindi, mr: Marathi). If omitted, auto-detects for mixed languages.'),
});
export type TranscribeAudioInput = z.infer<typeof TranscribeAudioInputSchema>;

const TranscribeAudioOutputSchema = z.object({
  transcription: z.string().describe('The transcribed text from the audio.'),
});
export type TranscribeAudioOutput = z.infer<typeof TranscribeAudioOutputSchema>;

export async function transcribeAudio(
  input: TranscribeAudioInput
): Promise<TranscribeAudioOutput> {
  return transcribeAudioFlow(input);
}

const transcribeAudioFlow = ai.defineFlow(
  {
    name: 'transcribeAudioFlow',
    inputSchema: TranscribeAudioInputSchema,
    outputSchema: TranscribeAudioOutputSchema,
  },
  async (input) => {
    if (!process.env.GROQ_API_KEY) {
      throw new Error('GROQ_API_KEY is not set in the environment variables.');
    }

    const groq = new Groq({ apiKey: process.env.GROQ_API_KEY });

    // Determine file extension from MIME type
    const extension = input.mimeType.split('/')[1]?.split(';')[0] || 'webm';
    if (!['mp3', 'wav', 'webm', 'ogg', 'flac', 'm4a'].includes(extension)) {
      throw new Error(`Unsupported audio format: ${extension}. Supported: mp3, m4a, wav, webm, ogg, flac.`);
    }
    
    // Create a temporary file path
    const tempDir = os.tmpdir();
    const tempFilePath = path.join(tempDir, `audio-${Date.now()}.${extension}`);

    try {
      // Decode the base64 string and write to the temporary file
      const buffer = Buffer.from(input.audioData, 'base64');
      await fs.promises.writeFile(tempFilePath, buffer);

      // Create a read stream from the temporary file
      const fileStream = fs.createReadStream(tempFilePath);

      // Prepare API options, omitting language if not provided for auto-detection
      const apiOptions: any = {
        file: fileStream,
        model: 'whisper-large-v3-turbo',
      };
      if (input.language) {
        apiOptions.language = input.language;
      }

      const transcription = await groq.audio.transcriptions.create(apiOptions);

      if (!transcription.text) {
        throw new Error('Transcription failed: No text returned from API.');
      }

      return { transcription: transcription.text.trim() };
    } catch (error) {
      console.error('Transcription error:', error);
      if (error instanceof Error && (error as any).code === 'ENOENT') {
        throw new Error('File not found at the provided path.');
      }
      throw new Error('Failed to transcribe audio. Please check the input and try again.');
    } finally {
      // Clean up the temporary file
      try {
        await fs.promises.unlink(tempFilePath);
      } catch (cleanupError) {
        console.error('Failed to clean up temporary audio file:', cleanupError);
      }
    }
  }
);
