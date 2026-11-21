"""
LexAI — AI Legal Assistant for Indian Law
Master Streamlit Application
"""

import os
import sys
from pathlib import Path

# Ensure application directory is in Python path
APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import streamlit as st

# Must be the very first Streamlit command
st.set_page_config(
    page_title="LexAI — Indian Legal Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

from core.config import (
    GROQ_API_KEY,
    GROQ_MODEL,
    PINECONE_API_KEY,
    RAG_THRESHOLD,
    RAG_TOP_K,
    SERPAPI_API_KEY,
    save_env_key,
)
from core.llm import call_llm, generate_chat_title
from core.pipeline import run_query
from core.prompts import DRAFT_TEMPLATES, DOCUMENT_SUMMARY
from db.database import Database
from ui.analytics import render_analytics_page
from ui.components import (
    load_css,
    render_bns_ipc_matrix,
    render_glossary_view,
    render_helplines_card,
    render_hero,
    render_pipeline_status,
    render_structured_answer,
)
from utils.export import export_markdown, export_pdf
from utils.helpers import (
    format_timestamp,
    get_width_kw,
    group_chats_by_date,
    sanitize_input,
)


# ── Database Initialization ───────────────────────────────────────────────
@st.cache_resource
def get_db() -> Database:
    return Database()


db = get_db()


# ── Session State Setup ───────────────────────────────────────────────────
def init_session_state() -> None:
    if "theme" not in st.session_state:
        st.session_state["theme"] = "light"
    if "user" not in st.session_state:
        st.session_state["user"] = None  # None = Guest Mode
    if "current_chat_id" not in st.session_state:
        # Create or select initial chat
        chats = db.get_chats(user_id=None)
        if chats:
            st.session_state["current_chat_id"] = chats[0]["id"]
        else:
            new_id = db.create_chat(user_id=None, title="New Consultation")
            st.session_state["current_chat_id"] = new_id
    if "nav_page" not in st.session_state:
        st.session_state["nav_page"] = "💬 Legal Chat"
    if "pending_query" not in st.session_state:
        st.session_state["pending_query"] = None
    if "answer_style" not in st.session_state:
        st.session_state["answer_style"] = "detailed"
    if "rag_threshold" not in st.session_state:
        st.session_state["rag_threshold"] = RAG_THRESHOLD
    if "rag_top_k" not in st.session_state:
        st.session_state["rag_top_k"] = RAG_TOP_K
    if "selected_model" not in st.session_state:
        st.session_state["selected_model"] = GROQ_MODEL
    # Dynamic API keys in session
    if "groq_key" not in st.session_state:
        st.session_state["groq_key"] = GROQ_API_KEY
    if "pinecone_key" not in st.session_state:
        st.session_state["pinecone_key"] = PINECONE_API_KEY
    if "serpapi_key" not in st.session_state:
        st.session_state["serpapi_key"] = SERPAPI_API_KEY


init_session_state()
load_css()


# ── Sidebar Navigation & Conversation History ────────────────────────────
def render_sidebar() -> None:
    with st.sidebar:
        # App Header Brand
        st.markdown(
            """
            <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 14px; padding: 6px 0;">
                <div style="width: 44px; height: 44px; border-radius: 12px; background: linear-gradient(135deg, #1e3a8a, #d97706); display: flex; align-items: center; justify-content: center; font-size: 1.5rem; box-shadow: 0 4px 14px rgba(30, 58, 138, 0.4);">
                    ⚖️
                </div>
                <div>
                    <h2 style="margin: 0; font-size: 1.35rem; font-weight: 800; letter-spacing: -0.5px; color: #ffffff !important;">LexAI</h2>
                    <p style="margin: 0; font-size: 0.74rem; color: #94a3b8 !important; font-weight: 500;">Bharat Legal Intelligence</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        user = st.session_state.get("user")
        user_id = user["id"] if user else None

        # User Authentication status
        with st.expander(f"👤 Account: {user['username'] if user else 'Guest Mode'}", expanded=False):
            if user:
                if st.button("Log Out", key="logout_btn", **get_width_kw()):
                    st.session_state["user"] = None
                    st.rerun()
            else:
                auth_tab1, auth_tab2 = st.tabs(["Sign In", "Register"])
                with auth_tab1:
                    l_user = st.text_input("Username", key="login_u")
                    l_pass = st.text_input("Password", type="password", key="login_p")
                    if st.button("Sign In", key="login_btn", **get_width_kw()):
                        u_record = db.authenticate_user(l_user, l_pass)
                        if u_record:
                            st.session_state["user"] = u_record
                            st.toast(f"Welcome back, {l_user}!", icon="👋")
                            st.rerun()
                        else:
                            st.error("Invalid username or password.")
                with auth_tab2:
                    r_user = st.text_input("New Username", key="reg_u")
                    r_pass = st.text_input("New Password", type="password", key="reg_p")
                    if st.button("Create Account", key="reg_btn", **get_width_kw()):
                        if r_user and r_pass:
                            try:
                                uid = db.create_user(r_user, r_pass)
                                st.session_state["user"] = {"id": uid, "username": r_user}
                                st.success("Account created successfully!")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Username already taken or error: {e}")

        st.markdown("<hr>", unsafe_allow_html=True)

        # New Consultation Button
        if st.button("➕ New Consultation", key="new_chat_btn", type="primary", **get_width_kw()):
            new_id = db.create_chat(user_id=user_id, title="New Consultation")
            st.session_state["current_chat_id"] = new_id
            st.session_state["nav_page"] = "💬 Legal Chat"
            st.rerun()

        # Navigation Selector
        nav_options = [
            "💬 Legal Chat",
            "⚖️ BNS ⇄ IPC Cross-Reference",
            "🔖 Saved Answers",
            "📝 Legal Draft Generator",
            "📄 Document Analyzer",
            "📊 Analytics Dashboard",
            "📖 Legal Glossary",
            "🚨 Emergency Helplines",
            "⚙️ Settings",
        ]
        st.session_state["nav_page"] = st.radio(
            "Navigation",
            options=nav_options,
            index=nav_options.index(st.session_state["nav_page"]) if st.session_state["nav_page"] in nav_options else 0,
            label_visibility="collapsed",
        )

        st.markdown("<hr>", unsafe_allow_html=True)

        # Past Conversations List
        st.markdown("<h5 style='color: #e2e8f0; margin-bottom: 8px;'>📁 Past Consultations</h5>", unsafe_allow_html=True)

        search_query = st.text_input("Search chats...", key="chat_search", placeholder="Filter topics...", label_visibility="collapsed")
        all_chats = db.search_chats(search_query, user_id=user_id) if search_query else db.get_chats(user_id=user_id)

        grouped = group_chats_by_date(all_chats)
        current_cid = st.session_state.get("current_chat_id")

        for group_label, chats_in_group in grouped.items():
            st.caption(f"**{group_label}**")
            for c in chats_in_group:
                cid = c["id"]
                title = c.get("title") or "Consultation"
                is_active = (cid == current_cid)

                col_btn, col_pin, col_del = st.columns([0.74, 0.13, 0.13])
                with col_btn:
                    btn_type = "primary" if is_active else "secondary"
                    if st.button(
                        f"{'📌 ' if c.get('pinned') else ''}{title[:22]}",
                        key=f"chat_select_{cid}",
                        type=btn_type,
                        **get_width_kw(),
                    ):
                        st.session_state["current_chat_id"] = cid
                        st.session_state["nav_page"] = "💬 Legal Chat"
                        st.rerun()

                with col_pin:
                    pinned = bool(c.get("pinned", 0))
                    if st.button("📍" if pinned else "📌", key=f"pin_{cid}", help="Pin or unpin"):
                        db.pin_chat(cid, not pinned)
                        st.rerun()

                with col_del:
                    if st.button("🗑️", key=f"del_{cid}", help="Delete chat"):
                        db.delete_chat(cid)
                        if st.session_state["current_chat_id"] == cid:
                            remaining = db.get_chats(user_id=user_id)
                            st.session_state["current_chat_id"] = remaining[0]["id"] if remaining else None
                        st.rerun()

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown(
            """
            <div style="background: rgba(255,255,255,0.06); padding: 12px 14px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.1);">
                <div style="font-size: 0.78rem; font-weight: 800; color: #60a5fa; text-transform: uppercase; letter-spacing: 0.5px;">
                    🛡️ Privacy Guarantee
                </div>
                <div style="font-size: 0.72rem; color: #94a3b8; margin-top: 4px; line-height: 1.4;">
                    Consultations are stored locally in SQLite. Zero third-party telemetry.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


render_sidebar()


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: 💬 LEGAL CHAT
# ═══════════════════════════════════════════════════════════════════════════
def render_chat_page() -> None:
    current_cid = st.session_state.get("current_chat_id")
    if not current_cid:
        current_cid = db.create_chat(
            user_id=st.session_state["user"]["id"] if st.session_state.get("user") else None,
            title="New Consultation",
        )
        st.session_state["current_chat_id"] = current_cid

    chat_info = db.get_chat(current_cid)
    chat_title = chat_info.get("title", "Legal Consultation") if chat_info else "Legal Consultation"

    # Header Bar
    top_col1, top_col2, top_col3, top_col4 = st.columns([0.48, 0.20, 0.16, 0.16])
    with top_col1:
        st.markdown(f"<h3 style='margin: 0; font-size: 1.35rem; font-weight: 800; color: #0f172a;'>⚖️ {chat_title}</h3>", unsafe_allow_html=True)

    with top_col2:
        # Answer detail toggle
        st.session_state["answer_style"] = st.selectbox(
            "Answer Style",
            options=["detailed", "simple"],
            format_func=lambda x: "Detailed Legal" if x == "detailed" else "Simple (ELIF5)",
            index=0 if st.session_state["answer_style"] == "detailed" else 1,
            label_visibility="collapsed",
        )

    with top_col3:
        with st.popover("🚨 Helplines"):
            st.markdown("#### 🚨 24/7 National Helplines")
            st.markdown("- **Police Emergency**: `112`")
            st.markdown("- **Women in Distress**: `181`")
            st.markdown("- **Cyber Financial Fraud**: `1930`")
            st.markdown("- **Free Legal Aid (NALSA)**: `15100`")
            st.markdown("- **National Consumer Helpline**: `1915`")

    with top_col4:
        with st.popover("⚙️ Options"):
            st.markdown("##### Consultation Options")
            new_title = st.text_input("Rename Consultation", value=chat_title)
            if st.button("Save New Title", key="update_title_btn", **get_width_kw()):
                if new_title.strip():
                    db.update_chat_title(current_cid, new_title.strip())
                    st.rerun()

            st.markdown("---")
            # Export consultation
            curr_msgs = db.get_messages(current_cid)
            if curr_msgs:
                full_md = export_markdown(chat_title, curr_msgs)
                st.download_button(
                    "📄 Export as Markdown",
                    data=full_md,
                    file_name=f"{chat_title.lower().replace(' ', '_')}.md",
                    mime="text/markdown",
                    **get_width_kw(fn=st.download_button),
                )
                full_pdf = export_pdf(chat_title, curr_msgs)
                st.download_button(
                    "📑 Export as PDF Report",
                    data=full_pdf,
                    file_name=f"{chat_title.lower().replace(' ', '_')}.pdf",
                    mime="application/pdf",
                    **get_width_kw(fn=st.download_button),
                )

    messages = db.get_messages(current_cid)

    # If empty chat: Show Hero Landing with suggestion chips
    if not messages:
        render_hero(on_select_query=lambda q: None)

    # Render conversation messages
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", "")

        if role == "user":
            with st.chat_message("user", avatar="👤"):
                st.markdown(content)
        elif role == "assistant":
            with st.chat_message("assistant", avatar="⚖️"):
                render_structured_answer(
                    message=msg,
                    db=db,
                    on_followup=lambda q: None,
                )

    # Process prompt from chat_input OR pending starter chips
    prompt_input = st.chat_input("Ask any Indian law question (e.g. tenant eviction, FIR procedure, POSH, salary dues)...")
    prompt_to_run = None

    if st.session_state.get("pending_query"):
        prompt_to_run = st.session_state["pending_query"]
        st.session_state["pending_query"] = None
    elif prompt_input:
        prompt_to_run = prompt_input

    if prompt_to_run:
        cleaned_prompt = sanitize_input(prompt_to_run)

        # 1. Render and persist user message
        with st.chat_message("user", avatar="👤"):
            st.markdown(cleaned_prompt)

        user_msg_id = db.add_message(
            chat_id=current_cid,
            role="user",
            content=cleaned_prompt,
        )

        # If this is the first message, auto-title the chat
        if len(messages) == 0:
            auto_title = generate_chat_title(cleaned_prompt, api_key=st.session_state.get("groq_key"))
            db.update_chat_title(current_cid, auto_title)

        # 2. Execute multi-stage pipeline with live status events
        with st.chat_message("assistant", avatar="⚖️"):
            status_container = st.container()
            events_log = []

            with status_container:
                with st.status("🧠 Initiating LexAI Multi-Stage Legal Workflow...", expanded=True) as status_box:
                    runner = run_query(
                        query=cleaned_prompt,
                        style=st.session_state.get("answer_style", "detailed"),
                        rag_threshold=st.session_state.get("rag_threshold", 0.5),
                        top_k=st.session_state.get("rag_top_k", 3),
                        groq_api_key=st.session_state.get("groq_key"),
                        pinecone_api_key=st.session_state.get("pinecone_key"),
                        serpapi_api_key=st.session_state.get("serpapi_key"),
                    )

                    final_result = None
                    for event in runner:
                        node = event.get("node")
                        events_log.append(event)
                        st.write(f"• **{event.get('title', node)}**: {event.get('message', '')}")
                        if node == "complete":
                            final_result = event.get("result")

                    status_box.update(label="🎉 Analysis Complete", state="complete", expanded=False)

            if final_result:
                # Save assistant message in DB
                asst_msg_id = db.add_message(
                    chat_id=current_cid,
                    role="assistant",
                    content=final_result.get("raw_answer", ""),
                    metadata=final_result,
                )

                # Reload and render structured output
                st.rerun()


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: 🔖 SAVED ANSWERS
# ═══════════════════════════════════════════════════════════════════════════
def render_saved_page() -> None:
    st.markdown(
        """
        <div class="page-title-banner">
            <h2>🔖 Bookmarked Consultations & Guidance</h2>
            <p>Access all answers you have saved across your legal research sessions.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    user = st.session_state.get("user")
    bookmarks = db.get_bookmarks(user_id=user["id"] if user else None)

    if not bookmarks:
        st.info("No saved answers yet. Click '⭐ Bookmark' on any answer in the chat to save it here.")
        return

    for bm in bookmarks:
        with st.container(border=True):
            b_col1, b_col2 = st.columns([0.85, 0.15])
            with b_col1:
                st.markdown(f"#### ⚖️ From: *{bm.get('chat_title', 'Consultation')}*")
                st.caption(f"Saved on {format_timestamp(bm.get('created_at'))}")
            with b_col2:
                if st.button("🗑️ Remove", key=f"del_bm_{bm['id']}", **get_width_kw()):
                    db.remove_bookmark(bm["id"])
                    st.toast("Bookmark removed.")
                    st.rerun()

            msg_record = {
                "id": bm["message_id"],
                "content": bm.get("content", ""),
                "metadata": bm.get("metadata", {}),
            }
            render_structured_answer(msg_record, db=db)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: 📝 LEGAL DRAFT GENERATOR
# ═══════════════════════════════════════════════════════════════════════════
def render_drafts_page() -> None:
    st.markdown(
        """
        <div class="page-title-banner">
            <h2>📝 Indian Legal Document & Notice Generator</h2>
            <p>Select a standard template, enter the facts, and generate a formal draft compliant with Indian legal procedure.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    template_names = list(DRAFT_TEMPLATES.keys())
    selected_template_name = st.selectbox("Select Template", options=template_names)
    template = DRAFT_TEMPLATES[selected_template_name]

    st.info(f"ℹ️ **{selected_template_name}**: {template['description']}")

    # Render dynamic form fields
    field_values = {}
    with st.form(key="draft_form"):
        for field in template["fields"]:
            label = template["field_labels"].get(field, field.replace("_", " ").title())
            if "details" in field or "sought" in field or "facts" in field:
                field_values[field] = st.text_area(label, height=100)
            else:
                field_values[field] = st.text_input(label)

        submitted = st.form_submit_button("Generate Formal Legal Draft", type="primary", **get_width_kw())

    if submitted:
        # Check required fields
        if not any(field_values.values()):
            st.warning("Please fill in the required fields to draft the document.")
            return

        with st.spinner("⚖️ Drafting formal Indian legal application..."):
            prompt = template["prompt"].format(**field_values)
            draft_text = call_llm(
                prompt=prompt,
                system_prompt="You are a legal document drafting expert enrolled with the Bar Council of India.",
                temperature=0.2,
                api_key=st.session_state.get("groq_key"),
            )

        st.success("Draft Generated Successfully!")
        st.markdown("### 📄 Generated Draft Preview")
        st.text_area("Draft Text", value=draft_text, height=350)

        # Export buttons
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            st.download_button(
                label="📥 Download as Text (.txt)",
                data=draft_text,
                file_name=f"{selected_template_name.lower().replace(' ', '_')}.txt",
                mime="text/plain",
                **get_width_kw(fn=st.download_button),
            )
        with col_d2:
            st.download_button(
                label="📥 Download as Markdown (.md)",
                data=draft_text,
                file_name=f"{selected_template_name.lower().replace(' ', '_')}.md",
                mime="text/markdown",
                **get_width_kw(fn=st.download_button),
            )


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: 📄 DOCUMENT ANALYZER
# ═══════════════════════════════════════════════════════════════════════════
def render_document_analyzer_page() -> None:
    st.markdown(
        """
        <div class="page-title-banner">
            <h2>📄 Legal Document & Contract Analyzer</h2>
            <p>Upload a rent agreement, FIR copy, or legal notice to extract risks, red flags, and plain-language clauses.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader("Upload legal document (PDF or TXT)", type=["pdf", "txt"])

    if uploaded_file is not None:
        doc_text = ""
        try:
            if uploaded_file.name.endswith(".pdf"):
                from pypdf import PdfReader
                reader = PdfReader(uploaded_file)
                pages = [page.extract_text() for page in reader.pages if page.extract_text()]
                doc_text = "\n".join(pages)
            else:
                doc_text = uploaded_file.read().decode("utf-8", errors="replace")
        except Exception as e:
            st.error(f"Error parsing document: {e}")
            return

        if not doc_text.strip():
            st.warning("Could not extract any text from this document.")
            return

        st.caption(f"Loaded {len(doc_text)} characters from {uploaded_file.name}")
        with st.expander("Preview Extracted Text"):
            st.text(doc_text[:1200] + ("..." if len(doc_text) > 1200 else ""))

        if st.button("🔍 Analyze Document for Risks & Rights", type="primary", **get_width_kw()):
            with st.spinner("Analyzing clauses under Indian contract and procedural law..."):
                prompt = DOCUMENT_SUMMARY.format(document_text=doc_text[:12000])
                analysis = call_llm(
                    prompt=prompt,
                    system_prompt="You are a senior Indian advocate reviewing documents for risk analysis.",
                    temperature=0.2,
                    api_key=st.session_state.get("groq_key"),
                )

            st.markdown("### 📋 Document Analysis Report")
            st.markdown(analysis)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: ⚙️ SETTINGS
# ═══════════════════════════════════════════════════════════════════════════
def render_settings_page() -> None:
    st.markdown(
        """
        <div class="page-title-banner">
            <h2>⚙️ Application Settings & Configuration</h2>
            <p>Customize model parameters, retrieval thresholds, API keys, and privacy controls.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab_model, tab_keys, tab_privacy = st.tabs(["🤖 Model & RAG", "🔑 API Keys", "🛡️ Privacy & Data"])

    with tab_model:
        st.markdown("##### LLM & Retrieval Configuration")
        model_options = [
            "qwen/qwen3.8-27b",
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b",
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "mixtral-8x7b-32768",
        ]
        st.session_state["selected_model"] = st.selectbox(
            "Groq LLM Model",
            options=model_options,
            index=model_options.index(st.session_state.get("selected_model", GROQ_MODEL)) if st.session_state.get("selected_model") in model_options else 0,
        )

        st.session_state["rag_threshold"] = st.slider(
            "RAG Similarity Fallback Threshold",
            min_value=0.2,
            max_value=0.9,
            value=float(st.session_state.get("rag_threshold", 0.5)),
            step=0.05,
            help="If Pinecone cosine similarity score falls below this value, the pipeline falls back to SerpAPI web search.",
        )

        st.session_state["rag_top_k"] = st.slider(
            "Knowledge Base Chunks (Top-K)",
            min_value=1,
            max_value=8,
            value=int(st.session_state.get("rag_top_k", 3)),
            help="Number of legal chunks to retrieve from the nearest cluster.",
        )

    with tab_keys:
        st.markdown("##### API Keys (Stored in memory and local .env)")
        st.caption("You can configure your keys here or via the `.env` file.")

        k_groq = st.text_input("Groq API Key", value=st.session_state.get("groq_key", ""), type="password")
        k_pine = st.text_input("Pinecone API Key", value=st.session_state.get("pinecone_key", ""), type="password")
        k_serp = st.text_input("SerpAPI Key", value=st.session_state.get("serpapi_key", ""), type="password")

        if st.button("Save API Keys", **get_width_kw()):
            g_key = k_groq.strip().strip("\"' ")
            p_key = k_pine.strip().strip("\"' ")
            s_key = k_serp.strip().strip("\"' ")
            st.session_state["groq_key"] = g_key
            st.session_state["pinecone_key"] = p_key
            st.session_state["serpapi_key"] = s_key
            save_env_key("GROQ_API_KEY", g_key)
            save_env_key("PINECONE_API_KEY", p_key)
            save_env_key("SERPAPI_API_KEY", s_key)
            st.success("API keys updated and saved to local .env file!")

    with tab_privacy:
        st.markdown("##### Data Retention & Local SQLite Storage")
        st.write(f"- **Local SQLite Database Path**: `{db.db_path}`")
        st.write(f"- **Current Database Size**: `{db.get_db_size()}`")
        st.write("- **Analytics Tracking**: Zero third-party cookies or telemetry. All conversations are stored solely in your local database.")

        st.markdown("<br>", unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown("⚠️ **Danger Zone**")
            st.caption("Permanently delete all consultation history, bookmarks, and saved feedback.")
            if st.button("🗑️ Delete All My Data", type="primary", **get_width_kw()):
                user = st.session_state.get("user")
                db.clear_all_data(user_id=user["id"] if user else None)
                st.session_state["current_chat_id"] = None
                st.success("All conversations and local data have been permanently cleared.")
                st.rerun()


# ── Route to Active Navigation Page ──────────────────────────────────────
page = st.session_state.get("nav_page", "💬 Legal Chat")

if page == "💬 Legal Chat":
    render_chat_page()
elif page == "⚖️ BNS ⇄ IPC Cross-Reference":
    st.markdown(
        """
        <div class="page-title-banner">
            <h2>⚖️ Bharatiya Nyaya Sanhita (BNS) ⇄ IPC & BNSS ⇄ CrPC</h2>
            <p>Interactive search & reference matrix for India's 2024 criminal law transition.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_bns_ipc_matrix(in_expander=False)
elif page == "🔖 Saved Answers":
    render_saved_page()
elif page == "📝 Legal Draft Generator":
    render_drafts_page()
elif page == "📄 Document Analyzer":
    render_document_analyzer_page()
elif page == "📊 Analytics Dashboard":
    render_analytics_page(db=db, user_id=st.session_state["user"]["id"] if st.session_state.get("user") else None)
elif page == "📖 Legal Glossary":
    render_glossary_view()
elif page == "🚨 Emergency Helplines":
    render_helplines_card()
elif page == "⚙️ Settings":
    render_settings_page()
