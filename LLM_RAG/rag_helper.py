from minsearch import Index
from google.genai import types
from ingest import load_labor_code_data, build_index
from tenacity import retry, stop_after_attempt, wait_exponential
from sentence_transformers import SentenceTransformer
from elasticsearch import Elasticsearch

documents = load_labor_code_data()
index = build_index(documents)

INSTRUCTIONS = """
Vous êtes un expert en droit du travail marocain (Loi 65-99).
Votre tâche est de répondre aux questions en vous basant uniquement sur le contexte fourni.
Si la réponse ne se trouve pas dans le contexte, répondez par "Je n'ai pas l'information."
"""

USER_PROMPT_TEMPLATE = """
QUESTION: {question}

CONTEXT:
{context}
""".strip()


class RAGBase:
    def __init__(
            self,
            index,
            llm_client,
            instructions=INSTRUCTIONS,
            prompt_template=USER_PROMPT_TEMPLATE,
            model="gemini-3.1-flash-lite"
    ):
        self.index = index
        self.llm_client = llm_client
        self.instructions = instructions
        self.prompt_template = prompt_template
        self.model = model

        # NEW: Hybrid Search dependencies
        self.es_client = Elasticsearch("http://localhost:9200", request_timeout=60)
        self.index_name = "juris_maroc_vectors"
        self.embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    def search(self, query, num_results=5, k_rrf=1):
        """
        HYBRID SEARCH: BM25 (Keyword) + Vector k-NN (Meaning) : RRF Fusion
        """

        # --- A. KEYWORD SEARCH (BM25) ---
        bm25_res = self.es_client.search(
            index=self.index_name,
            query={
                "multi_match": {
                    "query": query,
                    "fields": ["text", "article_number^2"]
                }
            },
            size=20,
            _source=["article_number", "text"]
        )
        bm25_hits = bm25_res["hits"]["hits"]

        # --- B. VECTOR SEARCH (k-NN) ---
        query_vector = self.embedding_model.encode(query).tolist()
        knn_res = self.es_client.search(
            index=self.index_name,
            knn={
                "field": "text_vector",
                "query_vector": query_vector,
                "k": 20,
                "num_candidates": 100
            },
            _source=["article_number", "text"]
        )
        knn_hits = knn_res["hits"]["hits"]

        # --- C. RECIPROCAL RANK FUSION (RRF) ---
        rrf_scores = {}
        doc_store = {}

        for rank, hit in enumerate(bm25_hits):
            doc_id = hit["_id"]
            doc_store[doc_id] = hit["_source"]
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (rank + 1 + k_rrf))

        for rank, hit in enumerate(knn_hits):
            doc_id = hit["_id"]
            doc_store[doc_id] = hit["_source"]
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (rank + 1 + k_rrf))

        # Sort by RRF Fusion score → Top 5
        top_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:num_results]

        # --- D. FORMAT: Map 'text' → 'content' so build_context() works ---
        results = []
        for doc_id in top_ids:
            doc = doc_store[doc_id]
            results.append({
                "article_number": doc.get("article_number", "N/A"),
                "content": doc.get("text", ""),
                "rrf_score": round(rrf_scores[doc_id], 5)
            })

        return results

    def build_context(self, search_results):
        lines = []

        for doc in search_results:
            lines.append(f"Article {doc['article_number']}:")
            lines.append(doc['content'])
            lines.append("")

        return "\n".join(lines).strip()

    def build_prompt(self, question, results):
        context = self.build_context(results)
        prompt = self.prompt_template.format(
            question=question,
            context=context
        )
        return prompt.strip()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def llm(self, prompt):
        response = self.llm_client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=self.instructions,
                temperature=0.3
            )
        )
        return response.text

    def rag(self, query):
        search_results = self.search(query)
        prompt = self.build_prompt(query, search_results)
        answer = self.llm(prompt)
        return answer