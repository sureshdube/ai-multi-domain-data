

import React, { useState } from "react";
import { uploadCSV } from "./api";

function CSVIngestPage() {
  const [file, setFile] = useState(null);
  const [success, setSuccess] = useState(false);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
    setSuccess(false);
    setError("");
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError("");
    setSuccess(false);
    if (!file) {
      setError("Please select a CSV file.");
      setLoading(false);
      return;
    }
    try {
      const res = await uploadCSV(file);
      if (!res.ok) {
        const err = await res.json();
        setError(err.error || "Unknown error");
      } else {
        setSuccess(true);
      }
    } catch (err) {
      setError("Request failed: " + err.message);
    }
    setLoading(false);
  };

  return (
    <div style={{ padding: 32 }}>
      <h2>CSV Ingest Test Page</h2>
      <form onSubmit={handleSubmit} style={{ marginBottom: 16 }}>
        <label>
          Select CSV File:
          <input
            type="file"
            accept=".csv"
            onChange={handleFileChange}
            style={{ marginLeft: 8 }}
            required
          />
        </label>
        <button type="submit" style={{ marginLeft: 16 }} disabled={loading}>
          {loading ? "Uploading..." : "Upload and Ingest"}
        </button>
      </form>
      {error && <div style={{ color: "red" }}>{error}</div>}
      {success && <div style={{ color: "green" }}>CSV file processed successfully!</div>}
    </div>
  );
}

export default CSVIngestPage;
