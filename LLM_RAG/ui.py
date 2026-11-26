import requests
import streamlit as st

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="JurisMaroc IA - Code du Travail",
    page_icon="⚖️",
    layout="wide",
)

# --- SIDEBAR ---
with st.sidebar:
    st.title("⚖️ JurisMaroc IA")
    st.caption("Assistant Juridique – Loi n° 65-99 (Code du Travail Marocain)")
    st.divider()
    st.markdown("### 🏗️ Architecture")
    st.markdown(
        "- **Recherche Hybride :** Elasticsearch BM25 + k-NN (`MiniLM-L12-v2`)\n"
        "- **Fusion :** Reciprocal Rank Fusion (`k_rrf=1`, `Top-10`)\n"
        "- **LLM :** Gemini (`temperature=0.3`)\n"
        "- **Monitoring :** Prometheus & Grafana"
    )
    st.divider()
    st.markdown("### 💡 Exemples de questions")
    examples = [
        "Quelles sont les conditions de la démission et le délai de préavis ?",
        "Je travaille dans la même entreprise depuis 6 ans, ai-je droit à des jours de congé supplémentaires ?",
        "Dans quels cas un employeur peut-il licencier un salarié sans préavis ni indemnité ?",
        "Quelle est la durée légale hebdomadaire du travail ?",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True):
            st.session_state["pending_question"] = ex

# --- MAIN HEADER ---
st.header("⚖️ Assistant IA – Code du Travail Marocain")
st.write(
    "Posez vos questions sur le droit du travail marocain (Loi 65-99). "
    "Les réponses sont strictement basées sur les articles officiels."
)

if "messages" not in st.session_state:
    st.session_state.messages = []


def send_feedback(vote: str, msg_idx: int):
    try:
        requests.post(f"{API_URL}/feedback", json={"vote": vote}, timeout=5)
        st.session_state.messages[msg_idx]["feedback"] = vote
        st.toast("Merci pour votre retour ! (Enregistré dans Prometheus/Grafana)")
    except Exception as e:
        st.error(f"Erreur d'envoi du feedback : {e}")


# --- DISPLAY CHAT HISTORY ---
for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if msg["role"] == "assistant" and "meta" in msg:
            meta = msg["meta"]
            st.caption(
                f"⚡ Recherche Elasticsearch : **{meta['retrieval_ms']} ms** | "
                f"Temps total : **{meta['total_ms']} ms**"
            )

            with st.expander(
                f"📚 Voir les {len(meta['retrieved_articles'])} articles consultés (RRF)"
            ):
                for art in meta["retrieved_articles"]:
                    st.markdown(
                        f"**Article {art['article_number']}** *(Score RRF : `{art['rrf_score']}`)*"
                    )
                    st.text(art["content"])
                    st.divider()

            # Feedback buttons
            col1, col2, _ = st.columns([1, 1, 10])
            current_vote = msg.get("feedback")
            with col1:
                st.button(
                    "👍",
                    key=f"up_{idx}",
                    disabled=current_vote is not None,
                    on_click=send_feedback,
                    args=("up", idx),
                )
            with col2:
                st.button(
                    "👎",
                    key=f"down_{idx}",
                    disabled=current_vote is not None,
                    on_click=send_feedback,
                    args=("down", idx),
                )

# --- HANDLE NEW INPUT ---
user_input = st.chat_input("Posez votre question juridique ici...")
if "pending_question" in st.session_state:
    user_input = st.session_state.pop("pending_question")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Recherche des articles et génération de la réponse..."):
            try:
                res = requests.post(
                    f"{API_URL}/ask", json={"question": user_input}, timeout=60
                )
                res.raise_for_status()
                data = res.json()

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": data["answer"],
                    "meta": {
                        "retrieved_articles": data["retrieved_articles"],
                        "retrieval_ms": data["retrieval_ms"],
                        "total_ms": data["total_ms"],
                    },
                })
                st.rerun()
            except Exception as e:
                st.error(f"Erreur de connexion à l'API FastAPI ({API_URL}) : {e}")