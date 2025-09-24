// src/api.js
import config from "./config";

const apiFetch = async (endpoint, options = {}) => {
  const url = `${config.API_BASE_URL}${endpoint}`;
  const response = await fetch(url, options);
  return response;
};

export const uploadCSV = async (file) => {
  const formData = new FormData();
  formData.append("file", file);
  return apiFetch("/api/ingest-csv", {
    method: "POST",
    body: formData
  });
};

export const getInsights = async () => apiFetch("/api/insights");
export const postNLQuery = async (query) =>
  apiFetch("/api/llm-query", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt: query })
  });
// Add more API calls as needed
