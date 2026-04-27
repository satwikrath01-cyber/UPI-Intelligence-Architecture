"""
app.py — UPI Intelligence Architecture Streamlit UI.
Run: streamlit run app.py
"""

import tempfile
from pathlib import Path
import streamlit as st

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="UPI Intelligence Architecture",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Guard ──────────────────────────────────────────────────────────────────────
if not Path("chroma_db").exists():
    st.error("Vector store not found. Run `python ingest.py` first.")
    st.stop()

# ── Cached resources ───────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading embedding model...")
def _embeddings():
    from ingest import get_embeddings
    return get_embeddings()

@st.cache_resource(show_spinner="Connecting to vector store...")
def _vectorstore():
    from ingest import get_vectorstore
    return get_vectorstore(_embeddings())

@st.cache_resource(show_spinner="Starting UPI Intelligence Architecture...")
def _chain_and_retriever():
    from rag import build_chain
    return build_chain(_vectorstore())

embeddings  = _embeddings()
vectorstore = _vectorstore()
chain, retriever = _chain_and_retriever()

# ── Session state ──────────────────────────────────────────────────────────────
if "messages"         not in st.session_state:
    st.session_state.messages = []
if "pending"          not in st.session_state:
    st.session_state.pending = None
if "processed_uploads" not in st.session_state:
    st.session_state.processed_uploads = set()

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 💳 UPI Intelligence Architecture")
    st.caption("Powered by Groq · LLaMA 3.3 70B")
    st.divider()

    # ── Upload new circular ────────────────────────────────────────────────────
    st.markdown("### ➕ Add New Circular")
    from ingest import tesseract_available
    if not tesseract_available():
        st.warning(
            "**Tesseract not found** — scanned PDFs won't be OCR'd.\n\n"
            "Install from: https://github.com/UB-Mannheim/tesseract/wiki\n"
            "Then restart the app."
        )
    uploaded = st.file_uploader("Upload a PDF circular", type=["pdf"], label_visibility="collapsed")

    if uploaded and uploaded.name not in st.session_state.processed_uploads:
        with st.spinner(f"Processing {uploaded.name}..."):
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(uploaded.read())
                tmp_path = tmp.name

            from ingest import ingest_pdf
            result = ingest_pdf(tmp_path, vectorstore)

            Path(tmp_path).unlink(missing_ok=True)

        st.session_state.processed_uploads.add(uploaded.name)

        if result["status"] == "duplicate":
            st.warning(f"Already loaded: **{result['name']}**")
        elif result["status"] == "empty":
            st.error(
                f"No text extracted from **{result['name']}**.\n\n"
                "If it's a scanned PDF, install Tesseract OCR and retry."
            )
        else:
            ocr_note = f" · {result['ocr_pages']} pages OCR'd" if result["ocr_pages"] else ""
            st.success(
                f"Added **{result['name']}**\n\n"
                f"{result['pages']} pages · {result['chunks']} chunks{ocr_note}"
            )
    elif uploaded and uploaded.name in st.session_state.processed_uploads:
        st.info(f"**{uploaded.name}** was added this session.")

    st.divider()

    # ── Circular browser ───────────────────────────────────────────────────────
    from ingest import list_circulars
    all_circulars = list_circulars(vectorstore)

    st.markdown(f"### 📂 Loaded Circulars ({len(all_circulars)})")
    search_term = st.text_input("Filter", placeholder="Search circulars...", label_visibility="collapsed")
    filtered    = [c for c in all_circulars if search_term.lower() in c.lower()] if search_term else all_circulars

    with st.container(height=260):
        for name in filtered:
            label = name.replace(".pdf", "").replace("-", " ")
            if st.button(f"📄 {label}", key=name, use_container_width=True):
                st.session_state.pending = f"Summarise the circular: {name}"

    st.divider()

    # ── Quick prompts ──────────────────────────────────────────────────────────
    st.markdown("### ⚡ Quick Queries")
    quick = [
        "What are the UPI transaction limits?",
        "Explain UPI AutoPay rules",
        "What are merchant onboarding requirements?",
        "Summarise dispute resolution timelines",
        "List all circulars on fraud prevention",
        "Compare interoperability rules across circulars",
        "What are the UPI charges and MDR rules?",
    ]
    for q in quick:
        if st.button(q, use_container_width=True, key=q):
            st.session_state.pending = q

    st.divider()
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pending  = None
        st.rerun()

    st.caption("Answers grounded in loaded circulars only. Never fabricated.")

# ── Main area ──────────────────────────────────────────────────────────────────
st.title("💳 UPI Intelligence Architecture")
st.markdown(
    "Ask anything about your UPI circulars — **summaries**, **explanations**, "
    "**comparisons**, or **specific rules**. Upload new circulars anytime from the sidebar."
)
st.divider()

# ── Chat history ───────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander(f"📄 {len(msg['sources'])} source circular(s)"):
                for src in msg["sources"]:
                    st.markdown(f"- `{src}`")

# ── Input ──────────────────────────────────────────────────────────────────────
pending    = st.session_state.pop("pending", None) if st.session_state.pending else None
user_input = st.chat_input("Ask about any UPI circular...") or pending

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    from rag import get_context_and_sources, format_history

    context, sources = get_context_and_sources(retriever, user_input)
    history_text     = format_history(st.session_state.messages[:-1])

    with st.chat_message("assistant"):
        full_response = st.write_stream(
            chain.stream({"question": user_input, "context": context, "history": history_text})
        )
        if sources:
            with st.expander(f"📄 {len(sources)} source circular(s)"):
                for src in sources:
                    st.markdown(f"- `{src}`")

    st.session_state.messages.append({
        "role":    "assistant",
        "content": full_response,
        "sources": sources,
    })
