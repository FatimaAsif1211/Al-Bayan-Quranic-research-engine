import os

# --- OFFLINE MODE (must be set BEFORE importing transformers/optimum) ---
# Belt-and-suspenders alongside local_files_only=True below: stops any
# accidental network call to huggingface.co on startup.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import streamlit as st
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForFeatureExtraction
from sentence_transformers import CrossEncoder
import torch
import requests
import re
import json
import pandas as pd
import time

# --- CONFIGURATION ---
QDRANT_STORAGE_PATH = "qdrant_storage"
COLLECTION_NAME = "tafsir_ibn_kathir_v2"
MODEL_ID = "BAAI/bge-m3"
LOCAL_ONNX_PATH = "models/bge-m3-onnx"  # our own export, saved locally after first run
OLLAMA_MODEL_NAME = "myllama3"

APP_NAME = "Al-Bayan"
APP_TAGLINE = "Qur'anic Research Engine"

# --- PAGE SETUP ---
st.set_page_config(
    page_title=f"{APP_NAME} · {APP_TAGLINE}",
    page_icon="✒️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------
# STYLE SHEET (CLEAN CONFLICTS & HIGH CONTRAST)
# ------------------------------------------------------------------
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Amiri:wght@400;700&family=Merriweather:wght@400;700;900&family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, sans-serif;
        color: #12151c !important;
    }

    .stApp {
        background-color: #eef1f6;
    }

    /* Kill Streamlit's default top padding so our header sits flush */
    .block-container { padding-top: 2rem; max-width: 1180px; }

    /* ---------- Masthead ---------- */
    .masthead {
        background: #101a33;
        border-radius: 10px;
        padding: 28px 32px;
        margin-bottom: 26px;
        box-shadow: 0 4px 14px rgba(16, 26, 51, 0.18);
    }
    .masthead-name {
        font-family: 'Merriweather', serif;
        font-size: 36px;
        font-weight: 900;
        color: #ffffff !important;
        letter-spacing: 0.3px;
        margin: 0;
    }
    .masthead-name .arabic-mark {
        font-family: 'Amiri', serif;
        font-weight: 700;
        margin-right: 12px;
        color: #f2b705 !important;
    }
    .masthead-tagline {
        font-size: 13.5px;
        text-transform: uppercase;
        letter-spacing: 2px;
        color: #cdd6ea !important;
        margin-top: 6px;
        font-weight: 500;
    }

    /* ---------- Section labels ---------- */
    .section-label {
        font-size: 12.5px;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: #c4870a !important;
        font-weight: 700;
        margin-bottom: 4px;
    }
    .section-title {
        font-family: 'Merriweather', serif;
        font-size: 24px;
        font-weight: 700;
        color: #101a33 !important;
        margin-top: 0;
        margin-bottom: 18px;
    }

    /* ---------- Source / result cards ---------- */
    .record-card {
        background: #ffffff;
        border: 1px solid #d7dce6;
        border-left: 5px solid #1e4fd8;
        border-radius: 8px;
        padding: 22px 26px;
        margin-bottom: 18px;
        box-shadow: 0 2px 8px rgba(16, 26, 51, 0.05);
    }
    .record-card .ref-line {
        display: flex;
        justify-content: space-between;
        align-items: baseline;
        font-size: 13.5px;
        color: #52586b !important;
        margin-bottom: 12px;
        border-bottom: 1px solid #e7eaf1;
        padding-bottom: 10px;
    }
    .record-card .ref-name {
        font-weight: 700;
        color: #101a33 !important;
        letter-spacing: 0.2px;
        font-size: 14.5px;
    }
    .record-card .ref-score {
        font-variant-numeric: tabular-nums;
        color: #c4870a !important;
        font-weight: 600;
    }

    /* ---------- Arabic text ---------- */
    .arabic-render {
        font-family: 'Amiri', serif;
        font-size: 32px;
        font-weight: 700;
        text-align: right;
        direction: rtl;
        line-height: 2.1;
        color: #0d1220 !important;
        background: #f4f6fb;
        border: 1px solid #e0e5f0;
        border-radius: 8px;
        padding: 20px 24px;
        margin: 10px 0 16px 0;
    }

    .translation-line {
        font-size: 15.5px;
        line-height: 1.75;
        color: #1c2130 !important;
        font-weight: 400;
        background: #f7f8fb;
        border-left: 3px solid #1e4fd8;
        padding: 12px 16px;
        border-radius: 0 6px 6px 0;
    }

    .tafsir-block {
        font-size: 15.5px;
        line-height: 1.8;
        color: #1c2130 !important;
        text-align: justify;
    }

    /* ---------- Stat strip ---------- */
    .stat-strip {
        display: flex;
        flex-wrap: wrap;
        gap: 28px;
        font-size: 13px;
        color: #52586b !important;
        background: #f7f8fb;
        border: 1px solid #e7eaf1;
        border-radius: 6px;
        margin-top: 16px;
        padding: 12px 16px;
    }
    .stat-strip b { color: #101a33 !important; }

    /* ---------- Sidebar ---------- */
    section[data-testid="stSidebar"] {
        background-color: #101a33 !important;
        border-right: 1px solid #0a1226;
    }
    section[data-testid="stSidebar"] * {
        color: #e8ecf6 !important;
    }
    .sidebar-heading {
        font-family: 'Merriweather', serif;
        font-size: 17px;
        font-weight: 700;
        color: #ffffff !important;
        margin-bottom: 2px;
    }
    .sidebar-sub {
        font-size: 11.5px;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        color: #f2b705 !important;
        margin-bottom: 14px;
        font-weight: 700;
    }
    section[data-testid="stSidebar"] hr {
        border-color: #2a3654;
    }

    /* Text inputs */
    .stTextInput > div > div > input {
        background-color: #ffffff !important;
        border: 1.5px solid #c7cede !important;
        color: #101a33 !important;
        font-size: 15px;
        border-radius: 6px;
    }

    /* Buttons */
    .stButton > button, .stDownloadButton > button {
        background-color: #1e4fd8 !important;
        color: #ffffff !important;
        border-radius: 6px;
        border: none;
        font-weight: 600;
        padding: 0.5rem 1.1rem;
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
        background-color: #163bab !important;
        color: #ffffff !important;
    }

    /* General labels color corrections */
    h3, h2, h1, p, span, label {
        color: #101a33 !important;
    }

    /* --- SPECIFIC FIXES FOR EXPANDERS / STEP BANNERS --- */
    [data-testid="stExpander"] summary {
        background-color: #ffffff !important;
        color: #101a33 !important;
        border-radius: 6px;
        border: 1px solid #cbd5e1;
    }
    [data-testid="stExpander"] summary * {
        color: #101a33 !important;
    }
    [data-testid="stExpander"] {
        background-color: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 6px;
    }

    /* --- SPECIFIC FIXES FOR FILE UPLOADER CONTAINER --- */
    [data-testid="stFileUploader"] {
        background-color: #ffffff !important;
        border: 1.5px dashed #cbd5e1 !important;
        padding: 20px;
        border-radius: 8px;
    }
    [data-testid="stFileUploader"] * {
        color: #101a33 !important;
    }
    [data-testid="stFileUploader"] button {
        background-color: #e2e8f0 !important;
        color: #101a33 !important;
        border: 1px solid #cbd5e1 !important;
    }

    /* Preserve st.json STYLING FROM OVERRIDES */
    [data-testid="stJson"] * {
        color: inherit !important;
    }
    [data-testid="stJson"] {
        background-color: #0e1117 !important;
        padding: 15px;
        border-radius: 8px;
    }

    /* Footer */
    .site-footer {
        text-align: center;
        font-size: 12px;
        color: #6b7185 !important;
        letter-spacing: 0.5px;
        margin-top: 40px;
        padding-top: 16px;
        border-top: 1px solid #d7dce6;
    }

    .guide-box, [data-testid="stExpander"] div, [data-testid="stExpander"] p, [data-testid="stExpander"] li {
        color: #101a33 !important;
    }
    .guide-box b, [data-testid="stExpander"] b {
        font-weight: 700 !important;
        color: #101a33 !important;
    }
    .guide-box code, [data-testid="stExpander"] code {
        background-color: #f1f5f9 !important;
        padding: 2px 6px !important;
        border-radius: 4px !important;
        color: #b41e1e !important;
        font-family: monospace !important;
        font-weight: 600 !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- MASTHEAD ---
st.markdown(f"""
    <div class="masthead">
        <p class="masthead-name"><span class="arabic-mark">البيان</span>{APP_NAME}</p>
        <p class="masthead-tagline">{APP_TAGLINE} &nbsp;·&nbsp; Semantic Retrieval Over Ibn Kathir's Tafsir</p>
    </div>
""", unsafe_allow_html=True)

# --- LOAD RESOURCES (OFFLINE MODE ACTIVATED) ---
@st.cache_resource
def load_resources():
    client = QdrantClient(path=QDRANT_STORAGE_PATH)

    # IMPORTANT: we do NOT load BAAI/bge-m3's hub-provided ONNX export directly.
    # That pre-built ONNX file names its output differently than what optimum
    # expects internally ("last_hidden_state"), which is exactly what caused
    # the KeyError. Instead, we let optimum do its own export ONE TIME (fully
    # offline, from the already-cached pytorch weights) and save the result
    # to LOCAL_ONNX_PATH. Every run after that loads directly from there --
    # no network calls, no re-export, and correctly-named outputs.
    if os.path.exists(LOCAL_ONNX_PATH) and os.path.exists(os.path.join(LOCAL_ONNX_PATH, "config.json")):
        tokenizer = AutoTokenizer.from_pretrained(LOCAL_ONNX_PATH, local_files_only=True)
        model = ORTModelForFeatureExtraction.from_pretrained(LOCAL_ONNX_PATH, local_files_only=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
        model = ORTModelForFeatureExtraction.from_pretrained(MODEL_ID, export=True, local_files_only=True)
        os.makedirs(LOCAL_ONNX_PATH, exist_ok=True)
        tokenizer.save_pretrained(LOCAL_ONNX_PATH)
        model.save_pretrained(LOCAL_ONNX_PATH)
    try:
        re_ranker = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2",
            local_files_only=True
        )
    except Exception:
        re_ranker = None

    return client, tokenizer, model, re_ranker

try:
    client, tokenizer, model, re_ranker = load_resources()
except Exception as e:
    st.error(f"Unable to initialize the retrieval backend: {e}")
    st.stop()

# --- HELPER FUNCTIONS ---
def get_embedding(text):
    inputs = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        raw_outputs = model(**inputs)

    # The hub-provided ONNX export for bge-m3 doesn't always name its output
    # "last_hidden_state", which is what optimum expects by default and is
    # what caused the KeyError. Grab the underlying tensor positionally
    # instead, so this works regardless of the export's output names.
    if hasattr(raw_outputs, "last_hidden_state") and raw_outputs.last_hidden_state is not None:
        hidden_state = raw_outputs.last_hidden_state
    elif isinstance(raw_outputs, dict) and "last_hidden_state" in raw_outputs:
        hidden_state = raw_outputs["last_hidden_state"]
    else:
        hidden_state = raw_outputs[0]

    return hidden_state[:, 0, :].numpy()[0].tolist()

def stream_ai_response(query, context_chunks):
    context = "\n\n---\n\n".join(context_chunks)
    prompt = f"""
    You are a careful Islamic studies research assistant. Using the query and the provided context records, write a clear, well-sourced summary.
    If the context has exact Qur'anic ayat, quote them precisely. Keep the tone measured and scholarly, highlighting key lessons, translations, and tafsir points.

    User Query: {query}
    Retrieved Context Records:
    {context}

    Your Response:
    """
    try:
        res = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": OLLAMA_MODEL_NAME, "prompt": prompt, "stream": True},
            stream=True,
            timeout=10
        )
        for line in res.iter_lines():
            if line:
                decoded = line.decode('utf-8')
                json_data = json.loads(decoded)
                yield json_data.get("response", "")
    except Exception:
        yield f"**Local model unreachable.** Start it with `ollama run {OLLAMA_MODEL_NAME}` to enable live synthesis."

# --- SIDEBAR ---
st.sidebar.markdown('<p class="sidebar-heading">Navigate</p>', unsafe_allow_html=True)
st.sidebar.markdown('<p class="sidebar-sub">Research Modes</p>', unsafe_allow_html=True)

search_mode = st.sidebar.radio(
    label="Research Modes",
    options=[
        "Semantic Search & Synthesis",
        "Direct Verse Lookup",
        "Keyword Scan",
        "Data Ingestion",
    ],
    label_visibility="collapsed",
)

st.sidebar.markdown("---")
st.sidebar.markdown('<p class="sidebar-heading" style="font-size:15px;">Retrieval Tuning</p>', unsafe_allow_html=True)
top_k_candidates = st.sidebar.slider("Candidate pool (dense search)", min_value=10, max_value=30, value=15)
rerank_display_limit = st.sidebar.slider("Verified results to display", min_value=3, max_value=8, value=5)

# ==============================================================================
# MODE 1: SEMANTIC SEARCH & SYNTHESIS
# ==============================================================================
if search_mode == "Semantic Search & Synthesis":
    st.markdown('<p class="section-label">Research</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-title">Semantic Search &amp; Synthesis</p>', unsafe_allow_html=True)

    with st.expander("📖 Step-by-Step Guide: How to Use Semantic Search & Synthesis", expanded=False):
        st.markdown("""
        <div class="guide-box">
            <b>How to Use:</b>
            <ol>
                <li>Type any conceptual topic or question in the input box below (e.g., <i>"rewards of patience"</i> or <i>"day of judgment"</i>).</li>
                <li>Press <b>Enter</b> to let the system fetch cross-referenced historical documents.</li>
                <li>Read the <b>Scholarly Summary</b> synthesized by your local AI, or download it as an offline record.</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

    user_query = st.text_input(
        "Query",
        placeholder="e.g., What is said about patience (sabr), or the significance of Friday prayer?",
        label_visibility="collapsed",
    )

    if user_query:
        start_time = time.time()

        with st.spinner("Running dense retrieval and cross-encoder re-ranking…"):
            query_vector = get_embedding(user_query)

            raw_results = client.query_points(
                collection_name=COLLECTION_NAME,
                query=query_vector,
                limit=top_k_candidates
            ).points

            if not raw_results:
                st.warning("No matching records were found for this query.")
            else:
                if re_ranker is not None:
                    pairs = [[user_query, (hit.payload.get("english_translation", "") or hit.payload.get("english_tafsir_chunk", ""))] for hit in raw_results]
                    scores = re_ranker.predict(pairs)
                    scored_results = sorted(zip(scores, raw_results), key=lambda x: x[0], reverse=True)
                    top_results = [item[1] for item in scored_results[:rerank_display_limit]]
                else:
                    top_results = raw_results[:rerank_display_limit]

                elapsed_time = time.time() - start_time

                context_for_llm = []
                for hit in top_results:
                    p = hit.payload
                    surah_title = p.get('surah_name') if p.get('surah_name') else f"Surah {p.get('surah_num', 'Unknown')}"
                    if "verse_id" in p:
                        context_for_llm.append(f"Surah: {surah_title} | Verse {p.get('verse_id')}: {p.get('english_translation')}")
                    else:
                        context_for_llm.append(f"Tafsir Block [{surah_title}]: {p.get('english_tafsir_chunk')}")

                col1, col2 = st.columns([1.1, 0.9], gap="large")

                with col1:
                    st.markdown('<p class="section-label">Synthesis</p>', unsafe_allow_html=True)
                    st.markdown('<p class="section-title" style="font-size:19px;">Scholarly Summary</p>', unsafe_allow_html=True)

                    # --- STREAMING ENGINE: fully-formed HTML card on every update ---
                    response_placeholder = st.empty()
                    full_ai_response = ""

                    for response_chunk in stream_ai_response(user_query, context_for_llm):
                        full_ai_response += response_chunk
                        response_placeholder.markdown(f"""
                            <div style="background-color: #ffffff; padding: 20px; border-radius: 8px; border: 1px solid #d7dce6; min-height: 150px; color: #101a33; line-height: 1.7;">
                                {full_ai_response}▌
                            </div>
                        """, unsafe_allow_html=True)

                    # Lock final output without the trailing cursor
                    response_placeholder.markdown(f"""
                        <div style="background-color: #ffffff; padding: 20px; border-radius: 8px; border: 1px solid #d7dce6; min-height: 150px; color: #101a33; line-height: 1.7;">
                            {full_ai_response}
                        </div>
                    """, unsafe_allow_html=True)

                    st.markdown(f"""
                        <div class="stat-strip">
                            <span>Latency: <b>{elapsed_time:.3f}s</b></span>
                            <span>Candidates scanned: <b>{top_k_candidates}</b></span>
                            <span>Verified matches: <b>{len(top_results)}</b></span>
                        </div>
                    """, unsafe_allow_html=True)

                    st.markdown("<div style='margin-top:14px;'></div>", unsafe_allow_html=True)

                    # Only show the download button once streaming has actually produced content
                    if full_ai_response:
                        st.download_button(
                            label="Download summary (.txt)",
                            data=f"{APP_NAME.upper()} RESEARCH SUMMARY\nQuery: {user_query}\n\nSummary:\n{full_ai_response}",
                            file_name=f"al_bayan_summary_{int(time.time())}.txt",
                            mime="text/plain"
                        )

                with col2:
                    st.markdown('<p class="section-label">Evidence</p>', unsafe_allow_html=True)
                    st.markdown('<p class="section-title" style="font-size:19px;">Source Records</p>', unsafe_allow_html=True)

                    for i, hit in enumerate(top_results):
                        p = hit.payload
                        score = hit.score if hit.score is not None else 0.0

                        surah_title = p.get('surah_name') if p.get('surah_name') else f"Surah {p.get('surah_num', 'Tafsir Segment')}"
                        verse_ref = f" — Ayah {p.get('verse_id')}" if "verse_id" in p else ""

                        with st.container():
                            st.markdown(f"""
                                <div class="record-card">
                                    <div class="ref-line">
                                        <span class="ref-name">{surah_title}{verse_ref}</span>
                                        <span class="ref-score">Match #{i+1} · score {score:.3f}</span>
                                    </div>
                            """, unsafe_allow_html=True)

                            if "verse_id" in p:
                                st.markdown(f"<div class=\"arabic-render\">{p.get('arabic_text')}</div>", unsafe_allow_html=True)
                                st.markdown(f"<div class=\"translation-line\">{p.get('english_translation')}</div>", unsafe_allow_html=True)
                            else:
                                st.markdown(f"<div class=\"tafsir-block\">{p.get('english_tafsir_chunk')}</div>", unsafe_allow_html=True)

                            st.markdown("</div>", unsafe_allow_html=True)

# ==============================================================================
# MODE 2: DIRECT VERSE LOOKUP
# ==============================================================================
elif search_mode == "Direct Verse Lookup":
    st.markdown('<p class="section-label">Reference</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-title">Direct Verse Lookup</p>', unsafe_allow_html=True)

    with st.expander("📖 Step-by-Step Guide: How to Use Direct Verse Lookup", expanded=False):
        st.markdown("""
        <div class="guide-box">
            <b>How to Use:</b>
            <ol>
                <li>Input the standard chapter-to-verse coordinate mapping index (e.g., <code>36:2</code> or <code>2:255</code>).</li>
                <li>The system will retrieve the exact matching row string from the database.</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

    ref_input = st.text_input(
        "Reference",
        placeholder="Surah:Ayah — e.g. 36:2, 1:1, 2:255",
        label_visibility="collapsed",
    )

    if ref_input:
        cleaned_ref = ref_input.strip()
        with st.spinner("Locating exact reference…"):
            try:
                res, _ = client.scroll(
                    collection_name=COLLECTION_NAME,
                    scroll_filter=qmodels.Filter(
                        must=[qmodels.FieldCondition(key="verse_id", match=qmodels.MatchValue(value=cleaned_ref))]
                    ),
                    limit=1
                )
                if res:
                    p = res[0].payload
                    surah_title = p.get('surah_name') if p.get('surah_name') else f"Surah {p.get('surah_num', 'Unknown')}"
                    st.markdown(f"""
                        <div class="record-card">
                            <div class="ref-line">
                                <span class="ref-name">{surah_title} — Ayah {cleaned_ref}</span>
                            </div>
                            <div class="arabic-render">{p.get('arabic_text')}</div>
                            <div class="translation-line">{p.get('english_translation')}</div>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.warning(f"No record matches the format: '{cleaned_ref}' in database.")
            except Exception as e:
                st.error(f"Error: {e}")

    # --- LIVE DATA INSPECTOR WITH PAGINATION & DYNAMIC LIMIT ---
    st.markdown("---")
    st.subheader("🔍 Live Database Inspector")
    st.write("Browse and paginate through all the records saved in your Qdrant database:")

    if "db_page" not in st.session_state:
        st.session_state.db_page = 0

    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([1.5, 1.5, 2])
    with col_ctrl1:
        inspect_limit = st.number_input("Records per page", min_value=1, max_value=50, value=5, step=1)

    current_offset = st.session_state.db_page * inspect_limit

    col_btn1, col_btn2, _ = st.columns([0.8, 0.8, 3.4])
    with col_btn1:
        if st.button("⬅️ Previous") and st.session_state.db_page > 0:
            st.session_state.db_page -= 1
            st.rerun()
    with col_btn2:
        if st.button("Next ➡️"):
            st.session_state.db_page += 1
            st.rerun()

    st.write(f"Showing records starting from index: **{current_offset}** (Page: {st.session_state.db_page + 1})")

    if st.button("Fetch / Refresh Records from Qdrant", key="run_inspector"):
        try:
            with st.spinner("Fetching database slice..."):
                debug_res, next_page_offset = client.scroll(
                    collection_name=COLLECTION_NAME,
                    limit=inspect_limit,
                    with_payload=True,
                    with_vectors=False,
                    offset=current_offset if current_offset > 0 else None
                )

                if debug_res:
                    st.success(f"Successfully retrieved {len(debug_res)} records!")
                    for idx, point in enumerate(debug_res):
                        record_num = current_offset + idx + 1
                        st.info(f"**Record #{record_num} (ID: {point.id})**")
                        st.json(point.payload)
                else:
                    st.warning("No more records found at this offset. Try going back to Previous pages.")
                    if st.session_state.db_page > 0:
                        st.session_state.db_page = 0
        except Exception as e:
            st.error(f"Inspection error: {e}")

# ==============================================================================
# MODE 3: KEYWORD SCAN (WITH HIGHLIGHTING)
# ==============================================================================
elif search_mode == "Keyword Scan":
    st.markdown('<p class="section-label">Utility</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-title">Keyword Scan</p>', unsafe_allow_html=True)

    with st.expander("📖 Step-by-Step Guide: How to Use Keyword Scan", expanded=False):
        st.markdown("""
        <div class="guide-box">
            <b>How to Use:</b>
            <ol>
                <li>Enter an exact alphanumeric word pattern inside the console text input.</li>
                <li>The app evaluates matches across the current database scroll window.</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

    keyword = st.text_input(
        "Keyword",
        placeholder="Enter an exact term to scan across indexed text segments",
        label_visibility="collapsed",
    )

    if keyword:
        with st.spinner("Scanning indexed records…"):
            res, _ = client.scroll(collection_name=COLLECTION_NAME, limit=250)
            matched_points = []
            keyword_lower = keyword.lower()

            for point in res:
                p = point.payload
                text_pool = (p.get("english_tafsir_chunk", "") + " " + p.get("english_translation", "")).lower()
                if keyword_lower in text_pool:
                    matched_points.append(p)

            if matched_points:
                st.success(f"Found {len(matched_points)} matching record(s).")

                for i, p in enumerate(matched_points[:10]):
                    surah_label = p.get('surah_name') if p.get('surah_name') else f"Surah {p.get('surah_num', 'Unknown')}"

                    raw_text = p.get('english_tafsir_chunk', p.get('english_translation', ''))

                    try:
                        compiled_regex = re.compile(rf"({re.escape(keyword)})", re.IGNORECASE)
                        highlighted_text = compiled_regex.sub(r"<mark style='background-color: #f2b705; color: #101a33; font-weight: 700; padding: 2px 4px; border-radius: 4px;'>\1</mark>", raw_text)
                    except Exception:
                        highlighted_text = raw_text

                    with st.container():
                        st.markdown(f"""
                            <div class="record-card">
                                <div class="ref-line">
                                    <span class="ref-name">{surah_label}</span>
                                    <span class="ref-score">Match {i+1}</span>
                                </div>
                                <div class="tafsir-block" style="text-align: left; line-height: 1.8;">
                                    {highlighted_text}
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
            else:
                st.info("No records contained that term within the scanned range.")

# ==============================================================================
# MODE 4: DATA INGESTION
# ==============================================================================
elif search_mode == "Data Ingestion":
    st.markdown('<p class="section-label">Admin</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-title">Data Ingestion Pipeline</p>', unsafe_allow_html=True)

    with st.expander("📖 Step-by-Step Guide: How to Use Data Ingestion", expanded=False):
        st.markdown("""
        <div class="guide-box">
            <b>How to Use:</b>
            <ol>
                <li>Upload any updated CSV log containing columns: <code>verse_id</code>, <code>arabic_text</code>, <code>english_translation</code>, and <code>surah_name</code>.</li>
                <li>Review the data table sample preview grid.</li>
                <li>Click <b>Append and index records</b> to invoke live injection.</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload a structured CSV of records to index:", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.dataframe(df.head(3), use_container_width=True)

        if st.button("Append and index records"):
            progress_bar = st.progress(0)
            total_rows = len(df)

            for idx, row in df.iterrows():
                v_id = str(row.get("verse_id", f"gen_{idx}"))
                v_vec = get_embedding(str(row.get("english_translation", "")))

                client.upsert(
                    collection_name=COLLECTION_NAME,
                    points=[
                        qmodels.PointStruct(
                            id=hash(v_id) % 10**8,
                            vector=v_vec,
                            payload={
                                "verse_id": v_id,
                                "arabic_text": str(row.get("arabic_text", "")),
                                "english_translation": str(row.get("english_translation", "")),
                                "surah_name": str(row.get("surah_name", "Unknown"))
                            }
                        )
                    ]
                )
                progress_bar.progress((idx + 1) / total_rows)

            st.success("Records embedded and written to the index.")

# --- FOOTER ---
st.markdown(f"""
    <p class="site-footer">{APP_NAME} · {APP_TAGLINE} · Local Retrieval-Augmented Research Workspace</p>
""", unsafe_allow_html=True)