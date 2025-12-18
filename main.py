# main.py
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import uvicorn


from main_api import handle_user_query, end_session  # import the core functions

app = FastAPI(title="Car Recommendation Agent API")


class QueryRequest(BaseModel):
    session_id: Optional[str] = None
    user_email: str
    user_query: str


class EndSessionRequest(BaseModel):
    session_id: str
    user_email: str


@app.get("/health")
def health():
    return {"status": "ok", "service": "car_recommendation_agent"}


@app.post("/query")
def api_query(req: QueryRequest):
    if not req.user_email or not req.user_query:
        raise HTTPException(
            status_code=400, detail="user_email and user_query required"
        )
    res = handle_user_query(req.session_id, req.user_email, req.user_query)
    return res


@app.post("/end_session")
def api_end_session(req: EndSessionRequest):
    if not req.session_id or not req.user_email:
        raise HTTPException(
            status_code=400, detail="session_id and user_email required"
        )
    res = end_session(req.session_id, req.user_email)
    return res


if __name__ == "__main__":
    # for local development
    uvicorn.run(
        "main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), reload=True
    )
