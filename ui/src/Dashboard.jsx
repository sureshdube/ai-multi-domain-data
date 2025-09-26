import React, { useEffect, useState } from "react";
import { getTrends, getCasesForTrend } from "./api";

export default function Dashboard() {
  const [trends, setTrends] = useState([]);
  const [selectedTrend, setSelectedTrend] = useState(null);
  const [cases, setCases] = useState([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    getTrends().then((data) => setTrends(data.trends || []));
  }, []);

  const handleTrendClick = (trend) => {
    setSelectedTrend(trend);
    setLoading(true);
    getCasesForTrend(trend.id).then((data) => {
      setCases(data.cases || []);
      setLoading(false);
    });
  };

  return (
    <div style={{ padding: 24 }}>
      <h2>Delivery Trends</h2>
      <ul style={{ listStyle: "none", padding: 0 }}>
        {trends.map((trend) => (
          <li key={trend.id} style={{ margin: "8px 0" }}>
            <button onClick={() => handleTrendClick(trend)} style={{ fontWeight: selectedTrend?.id === trend.id ? "bold" : "normal" }}>
              {trend.name} ({trend.count})
            </button>
          </li>
        ))}
      </ul>
      {selectedTrend && (
        <div style={{ marginTop: 32 }}>
          <h3>Cases for: {selectedTrend.name}</h3>
          {loading ? (
            <div>Loading cases...</div>
          ) : cases.length === 0 ? (
            <div>No cases found for this trend.</div>
          ) : (
            <table border="1" cellPadding="8" style={{ borderCollapse: "collapse", width: "100%" }}>
              <thead>
                <tr>
                  <th>Case ID</th>
                  <th>Order ID</th>
                  <th>Status</th>
                  <th>Customer</th>
                  <th>Details</th>
                </tr>
              </thead>
              <tbody>
                {cases.map((c) => (
                  <tr key={c.case_id}>
                    <td>{c.case_id}</td>
                    <td>{c.order_id}</td>
                    <td>{c.status}</td>
                    <td>{c.customer}</td>
                    <td>{c.details}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      )}
    </div>
  );
}
