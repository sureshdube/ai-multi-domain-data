import React, { useState } from "react";
import { postNLQuery } from "./api";

function NLQueryDashboard() {
  const [query, setQuery] = useState("");
  const [result, setResult] = useState(null);
  const [explanation, setExplanation] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError("");
    setResult(null);
    setExplanation(null);
    try {
      const res = await postNLQuery(query);
      if (!res.ok) {
        const err = await res.json();
        setError(err.error || "Unknown error");
      } else {
        const data = await res.json();
        setResult(data.llm_result || data.result || data);
        setExplanation(data.explanation || null);
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
        <textarea
          value={query}
          onChange={e => setQuery(e.target.value)}
          placeholder="Ask a question about your data..."
          style={{ width: 600, height: 80, marginRight: 16, resize: "vertical", fontSize: 16, padding: 8 }}
          required
        />
        <button type="submit" disabled={loading}>
          {loading ? "Processing..." : "Ask"}
        </button>
      </form>
      {error && <div style={{ color: "red" }}>{error}</div>}
      {/* Show formatted LLM result and explanation first */}
      {explanation && explanation.choices && explanation.choices[0] && explanation.choices[0].message && explanation.choices[0].message.content && (
        <div style={{ background: "#e8f5e9", padding: 16, borderRadius: 8, marginTop: 16, marginBottom: 24 }}>
          <h4>Explanation:</h4>
          <div style={{ whiteSpace: "pre-line", fontSize: 15 }}>{explanation.choices[0].message.content}</div>
        </div>
      )}
      {/* Show the previous Result section (table/object rendering) below the explanation for advanced users */}
      {result && (
        <div style={{ background: "#fffde7", padding: 16, borderRadius: 8, marginTop: 16 }}>
          <h4>Raw Result Data:</h4>
          {typeof result === "string" ? (
            <div>{result}</div>
          ) : Array.isArray(result) ? (
            <table border="1" cellPadding="8" style={{ borderCollapse: "collapse", width: "100%" }}>
              <thead>
                <tr>
                  {Object.keys(result[0] || {}).map((k) => (
                    <th key={k}>{k}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {result.map((row, i) => (
                  <tr key={i}>
                    {Object.values(row).map((v, j) => (
                      <td key={j}>{v}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          ) : typeof result === "object" ? (
            <table border="1" cellPadding="8" style={{ borderCollapse: "collapse", width: "100%" }}>
              <tbody>
                {Object.entries(result).map(([k, v]) => (
                  <tr key={k}>
                    <td style={{ fontWeight: "bold" }}>{k}</td>
                    <td>{typeof v === "object" ? JSON.stringify(v) : v}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : null}
        </div>
      )}
    </div>
  );
}

export default NLQueryDashboard;
