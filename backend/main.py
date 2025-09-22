from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/insights")
def get_insights():
    return {"insights": ["Dummy insight 1", "Dummy insight 2"]}

@app.post("/api/nlquery")
async def nl_query(request: Request):
    data = await request.json()
    return {"result": f"Dummy response to: {data.get('query', '')}"}

@app.get("/api/drilldown")
def get_drilldown():
    return {"case": "Dummy delivery case details"}

@app.get("/api/schedule")
def get_schedule():
    return {"schedule": "Dummy schedule config"}

@app.post("/api/schedule")
async def set_schedule(request: Request):
    data = await request.json()
    return {"status": "Schedule updated", "config": data}
