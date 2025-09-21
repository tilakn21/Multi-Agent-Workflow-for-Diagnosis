import { config } from 'dotenv';
config();

import '@/ai/flows/summarize-symptom-description.ts';
import '@/ai/flows/understand-user-symptoms.ts';
import '@/ai/flows/transcribe-audio.ts';
