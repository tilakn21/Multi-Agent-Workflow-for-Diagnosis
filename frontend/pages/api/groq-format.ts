import type { NextApiRequest, NextApiResponse } from "next";
import { Groq } from "groq-sdk";

const groq = new Groq({ apiKey: process.env.GROQ_API_KEY });

const CORS_HEADERS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "POST, OPTIONS",
  "Access-Control-Allow-Headers": "*",
};

export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse
) {
  // Handle CORS preflight
  if (req.method === "OPTIONS") {
    Object.entries(CORS_HEADERS).forEach(([key, value]) =>
      res.setHeader(key, value)
    );
    return res.status(200).end();
  }

  Object.entries(CORS_HEADERS).forEach(([key, value]) =>
    res.setHeader(key, value)
  );

  if (req.method !== "POST") {
    return res.status(405).json({ error: "Method not allowed" });
  }

  const { input } = req.body;
  if (!input) {
    return res.status(400).json({ error: "Missing input" });
  }

  try {
    const chatCompletion = await groq.chat.completions.create({
      messages: [{ role: "user", content: input }],
      model: "llama-3.3-70b-versatile",
      temperature: 1,
      max_tokens: 1024,
      top_p: 1,
      stream: false,
      stop: null,
    });
    const content = chatCompletion.choices[0].message.content;
    if (!content) {
      return res.status(500).json({ error: "Groq API returned null content" });
    }
    return res.status(200).json({ content });
  } catch (error: any) {
    return res.status(500).json({ error: error.message || "Groq API error" });
  }
}
