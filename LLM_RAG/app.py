import os
from time import perf_counter
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, Response
from google import genai
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field

from rag_helper import RAGBase

load_dotenv()
gemini_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
rag_system = RAGBase(index=None, llm_client=gemini_client)

app = FastAPI(title="Moroccan Labor Code RAG")

# --- PROMETHEUS METRICS ---
REQUEST_COUNT = Counter(
    "rag_requests_total", "Total number of RAG questions asked"
)
NO_INFO_COUNT = Counter(
    "rag_no_info_total",
    "Questions where RAG returned Je n'ai pas l'information",
)
FEEDBACK_COUNT = Counter(
    "rag_feedback_total",
    "User feedback on RAG answers",
    ["vote"],  # "up" or "down"
)
LATENCY_HISTOGRAM = Histogram(
    "rag_stage_duration_seconds",
    "Time spent in each RAG stage",
    ["stage"],
)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)


class FeedbackRequest(BaseModel):
    vote: Literal["up", "down"]


@app.post("/ask")
def ask(request: AskRequest):
    REQUEST_COUNT.inc()
    start = perf_counter()

    articles = rag_system.search(request.question, num_results=10, k_rrf=1)
    after_search = perf_counter()
    retrieval_sec = after_search - start
    LATENCY_HISTOGRAM.labels(stage="retrieval").observe(retrieval_sec)

    prompt = rag_system.build_prompt(request.question, articles)
    answer = rag_system.llm(prompt)
    end = perf_counter()
    llm_sec = end - after_search
    LATENCY_HISTOGRAM.labels(stage="llm").observe(llm_sec)

    if "Je n'ai pas l'information" in answer:
        NO_INFO_COUNT.inc()

    return {
        "answer": answer,
        "retrieved_articles": articles,
        "retrieval_ms": round(retrieval_sec * 1000, 1),
        "total_ms": round((end - start) * 1000, 1),
    }


@app.post("/feedback")
def feedback(request: FeedbackRequest):
    FEEDBACK_COUNT.labels(vote=request.vote).inc()
    return {"status": "recorded", "vote": request.vote}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)