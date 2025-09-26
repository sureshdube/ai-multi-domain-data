import React, { useState } from "react";
import CSVIngestPage from './CSVIngestPage';
import NLQueryDashboard from './NLQueryDashboard';
import Dashboard from './Dashboard';

function App() {
  const [page, setPage] = useState("csv");
  return (
    <div>
      <nav style={{ padding: 16, borderBottom: "1px solid #ccc", marginBottom: 24 }}>
        <button onClick={() => setPage("csv")} style={{ marginRight: 16 }}>
          CSV Ingest
        </button>
        <button onClick={() => setPage("nlq")} style={{ marginRight: 16 }}>
          Natural Language Query
        </button>
        <button onClick={() => setPage("dashboard")}>Dashboard</button>
      </nav>
      {page === "csv" && <CSVIngestPage />}
      {page === "nlq" && <NLQueryDashboard />}
      {page === "dashboard" && <Dashboard />}
    </div>
  );
}

export default App;