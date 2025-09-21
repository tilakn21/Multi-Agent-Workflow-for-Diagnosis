import { NextRequest, NextResponse } from "next/server";

const CORS_HEADERS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "POST, OPTIONS",
  "Access-Control-Allow-Headers": "Content-Type, ngrok-skip-browser-warning",
};

export async function OPTIONS() {
  return new NextResponse(null, {
    status: 200,
    headers: CORS_HEADERS,
  });
}

export async function POST(request: NextRequest) {
  // Handle CORS preflight for fetch requests
  if (request.method === "OPTIONS") {
    return new NextResponse(null, {
      status: 200,
      headers: CORS_HEADERS,
    });
  }

  const data = await request.json();
  // Your logic here
  return NextResponse.json(
    { success: true },
    {
      headers: CORS_HEADERS,
    }
  );
}
