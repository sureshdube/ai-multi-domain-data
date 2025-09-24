import React, { useState } from "react";
import CSVIngestPage from './CSVIngestPage';
import NLQueryDashboard from './NLQueryDashboard';

function App() {
  const [page, setPage] = useState("csv");
  return (
    <div>
      <nav style={{ padding: 16, borderBottom: "1px solid #ccc", marginBottom: 24 }}>
        <button onClick={() => setPage("csv")} style={{ marginRight: 16 }}>
          CSV Ingest
        </button>
        <button onClick={() => setPage("nlq")}>Natural Language Query</button>
      </nav>
      {page === "csv" && <CSVIngestPage />}
      {page === "nlq" && <NLQueryDashboard />}
    </div>
  );
}

export default App;