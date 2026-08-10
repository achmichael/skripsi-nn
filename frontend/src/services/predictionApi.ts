const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "/api";
const USE_MOCK_API = false;

import type { FormData, PredictionResponse } from "../types/prediction";

function delay(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function mockPrepaid(_data: FormData): Promise<PredictionResponse> {
  await delay(1500);
  return {
    success: true,
    prediction: Math.floor(Math.random() * 25) + 5,
  };
}

async function mockPostpaid(_data: FormData): Promise<PredictionResponse> {
  await delay(1500);
  return {
    success: true,
    prediction: Math.floor(Math.random() * 400000) + 100000,
  };
}

// TODO: Replace mock functions with real API calls when backend is available.
export async function predictPrepaid(data: FormData): Promise<PredictionResponse> {
  if (USE_MOCK_API) {
    return mockPrepaid(data);
  }

  const response = await fetch(`${API_BASE_URL}/predict/prepaid`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });

  if (!response.ok) {
    throw new Error(`HTTP ${response.status}`);
  }

  return response.json() as Promise<PredictionResponse>;
}

export async function predictPostpaid(data: FormData): Promise<PredictionResponse> {
  if (USE_MOCK_API) {
    return mockPostpaid(data);
  }

  const response = await fetch(`${API_BASE_URL}/predict/postpaid`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });

  if (!response.ok) {
    throw new Error(`HTTP ${response.status}`);
  }

  return response.json() as Promise<PredictionResponse>;
}
