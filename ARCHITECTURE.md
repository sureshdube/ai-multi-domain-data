# High-Level Architecture: Multi-Domain Delivery Insights Platform

## Overview
This architecture is designed to meet the business and technical requirements outlined in the dashboard and technical PRDs. It separates the system into a React-based UI and a Python FastAPI backend, with clear extensibility and API-driven integration.

---

## 1. UI Layer (React)
- **Purpose:** User dashboard, natural language query interface, drill-down navigation, and schedule management.
- **Key Components:**
  - DashboardPage: Main insights and trends
  - DrilldownPage: Detailed delivery case view
  - NLQueryBox: Natural language query input
  - ScheduleConfig: UI for managing aggregation schedules
  - DummyAPITestPage: For testing API connectivity
- **API Integration:** All data and actions are fetched via REST API endpoints from the backend.

---

## 2. Backend Layer (Python FastAPI)
- **Purpose:** Data ingestion, event correlation, LLM integration, analytics, and API services.
- **Key Components:**
  - /api/insights: Returns dummy insights data
  - /api/drilldown: Returns dummy delivery case details
  - /api/nlquery: Accepts NL queries, returns dummy response
  - /api/schedule: Accepts/returns dummy schedule config
- **Extensibility:** Designed to add real data processing, ML, and LLM integration in future phases.

---

## 3. Data Layer (Placeholder)
- **Purpose:** Simulate data storage and retrieval for demonstration.
- **Implementation:** In-memory or static JSON for dummy endpoints.

---

## 4. Deployment
- **UI:** React app (ui/)
- **Backend:** FastAPI app (backend/)
- **Containerization:** Both layers can be containerized for EC2 deployment.

---

## 5. Diagram

```
+-------------------+        REST API         +-------------------+
|    React UI       | <--------------------> |   FastAPI Backend |
| (Dashboard, NLQ)  |                       | (Dummy Endpoints) |
+-------------------+                       +-------------------+
```

---

## Next Steps
- Implement UI test page and dummy API endpoints only (no business logic).
- No real data processing or feature implementation at this stage.
