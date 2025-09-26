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
export const getTrends = async () => {
  const res = await apiFetch("/api/trends");
  return res.json();
};

export const getCasesForTrend = async (trendId) => {
  const res = await apiFetch(`/api/cases/${trendId}`);
  return res.json();
};

export const uploadCSVToDB = async (file, collection) => {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("collection", collection);
  return apiFetch("/api/ingest-csv-to-db", {
    method: "POST",
    body: formData
  });
};
// Add more API calls as needed
