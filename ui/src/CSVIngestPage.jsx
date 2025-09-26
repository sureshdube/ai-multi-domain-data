import React, { useState } from "react";
import { uploadCSVToDB } from "./api";

function CSVIngestPage() {
  const [file, setFile] = useState(null);
  const [collection, setCollection] = useState("");
  const [success, setSuccess] = useState(false);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    setFile(selectedFile);
    setSuccess(false);
    setError("");
    if (selectedFile && selectedFile.name) {
      // Remove .csv extension and prepopulate collection name
      const baseName = selectedFile.name.replace(/\.csv$/i, "");
      setCollection(baseName);
    }
  };

  const handleCollectionChange = (e) => {
    setCollection(e.target.value);
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
    if (!collection) {
      setError("Please enter a collection name.");
      setLoading(false);
      return;
    }
    try {
      const res = await uploadCSVToDB(file, collection);
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
      <h2>CSV Ingest to Database</h2>
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
        <label style={{ marginLeft: 16 }}>
          Collection Name:
          <input
            type="text"
            value={collection}
            onChange={handleCollectionChange}
            style={{ marginLeft: 8 }}
            required
          />
        </label>
        <button type="submit" style={{ marginLeft: 16 }} disabled={loading}>
          {loading ? "Uploading..." : "Upload and Load to DB"}
        </button>
      </form>
      {error && <div style={{ color: "red" }}>{error}</div>}
      {success && <div style={{ color: "green" }}>CSV file loaded to DB successfully!</div>}
    </div>
  );
}

export default CSVIngestPage;
