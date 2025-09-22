import React from "react";

function DummyAPITestPage() {
  const [insights, setInsights] = React.useState(null);
  const [nlResponse, setNlResponse] = React.useState("");

  const fetchInsights = async () => {
    const res = await fetch("/api/insights");
    setInsights(await res.json());
  };

  const sendNLQuery = async () => {
    const res = await fetch("/api/nlquery", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: "Why are deliveries delayed?" })
    });
    setNlResponse((await res.json()).result);
  };

  return (
    <div style={{ padding: 32 }}>
      <h2>Dummy API Test Page</h2>
      <button onClick={fetchInsights}>Fetch Dummy Insights</button>
      {insights && <pre>{JSON.stringify(insights, null, 2)}</pre>}
      <button onClick={sendNLQuery}>Send NL Query</button>
      {nlResponse && <div>NL Query Response: {nlResponse}</div>}
    </div>
  );
}

export default DummyAPITestPage;
