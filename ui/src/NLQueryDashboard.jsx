import React, { useState } from "react";
import { postNLQuery } from "./api";

function NLQueryDashboard() {
  const [query, setQuery] = useState("");
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError("");
    setResult(null);
    try {
      const res = await postNLQuery(query);
      if (!res.ok) {
        const err = await res.json();
        setError(err.error || "Unknown error");
      } else {
        const data = await res.json();
        setResult(data.llm_result || data.result || data);
      }
    } catch (err) {
      setError("Request failed: " + err.message);
    }
    setLoading(false);
  };

  return (
    <div style={{ padding: 32 }}>
      <h2>Natural Language Query Dashboard</h2>
      <form onSubmit={handleSubmit} style={{ marginBottom: 16 }}>
        <input
          type="text"
          value={query}
          onChange={e => setQuery(e.target.value)}
          placeholder="Ask a question about your data..."
          style={{ width: 400, marginRight: 16 }}
          required
        />
        <button type="submit" disabled={loading}>
          {loading ? "Processing..." : "Ask"}
        </button>
      </form>
      {error && <div style={{ color: "red" }}>{error}</div>}
      {result && (
        <div style={{ background: "#f4f4f4", padding: 16, borderRadius: 8 }}>
          <h4>Result:</h4>
          <pre>{typeof result === "string" ? result : JSON.stringify(result, null, 2)}</pre>
        </div>
      )}
    </div>
  );
}

export default NLQueryDashboard;
