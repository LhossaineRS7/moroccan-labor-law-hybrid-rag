import os
from time import perf_counter

from dotenv import load_dotenv
from fastapi import FastAPI
from google import genai
from pydantic import BaseModel, Field

from rag_helper import RAGBase

load_dotenv()
gemini_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
rag_system = RAGBase(index=None, llm_client=gemini_client)

app = FastAPI(title="Moroccan Labor Code RAG")


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)


@app.post("/ask")
def ask(request: AskRequest):
    start = perf_counter()

    articles = rag_system.search(request.question, num_results=5)
    after_search = perf_counter()

    prompt = rag_system.build_prompt(request.question, articles)
    answer = rag_system.llm(prompt)
    end = perf_counter()

    return {
        "answer": answer,
        "retrieved_articles": articles,
        "retrieval_ms": round((after_search - start) * 1000, 1),
        "total_ms": round((end - start) * 1000, 1),
    }