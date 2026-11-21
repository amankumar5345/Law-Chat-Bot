"""
UI Component library for LexAI.
Renders high-end structured legal answers, risk badges, interactive roadmaps,
starter chips, emergency helplines, pipeline progress, and feedback bars.
"""

from pathlib import Path
from typing import Any, Callable, Optional

import streamlit as st

from core.config import (
    BNS_IPC_MAPPING,
    HELPLINES,
    LEGAL_GLOSSARY,
    STARTER_QUESTIONS,
)
from utils.export import export_markdown, export_pdf
from utils.helpers import (
    format_confidence,
    risk_description,
    risk_emoji,
)


def load_css() -> None:
    """Inject custom modern styles into Streamlit."""
    css_path = Path(__file__).parent / "styles.css"
    if css_path.exists():
        with open(css_path, "r", encoding="utf-8") as f:
            css = f.read()
        theme = st.session_state.get("theme", "light")
        st.markdown(
            f"<style>\nbody[data-theme='{theme}'], .stApp[data-theme='{theme}'] {{\n{css}\n}}\n{css}\n</style>",
            unsafe_allow_html=True,
        )


def render_bns_ipc_matrix() -> None:
    """Renders an interactive BNS 2024 vs IPC 1860 & BNSS vs CrPC conversion widget."""
    with st.expander("⚖️ Quick Statute Cross-Reference: Bharatiya Nyaya Sanhita (BNS) ⇄ IPC / BNSS ⇄ CrPC", expanded=False):
        st.markdown("Search or compare the new 2024 criminal codes with legacy Indian Penal Code & CrPC provisions:")
        filter_text = st.text_input("🔍 Search offence or section...", placeholder="e.g. Cheating, Murder, FIR, Theft, 420, 302, 154", key="bns_search")

        filtered = [
            item for item in BNS_IPC_MAPPING
            if not filter_text or (
                filter_text.lower() in item["topic"].lower()
                or filter_text.lower() in item["old_law"].lower()
                or filter_text.lower() in item["new_law"].lower()
                or filter_text.lower() in item["desc"].lower()
            )
        ]

        for item in filtered:
            st.markdown(
                f"""
                <div class="matrix-row-item">
                    <div>
                        <div style="font-weight: 800; font-size: 0.95rem; margin-bottom: 3px;">{item['topic']}</div>
                        <div style="font-size: 0.8rem; color: #64748b;">{item['desc']}</div>
                        <div style="font-size: 0.72rem; color: #059669; font-weight: 700; margin-top: 4px;">• {item['severity']}</div>
                    </div>
                    <div style="display: flex; align-items: center; gap: 12px;">
                        <span class="matrix-law-old">{item['old_law']}</span>
                        <span class="matrix-arrow">➔</span>
                        <span class="matrix-law-new">{item['new_law']}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_hero(on_select_query: Optional[Callable[[str], None]] = None) -> None:
    """Render executive hero landing screen with colorful cards and 4-step pipeline visual."""

    # 1. Top Announcement Bar
    st.markdown(
        """
        <div class="top-statute-banner">
            <div>🇮🇳 Bharatiya Nyaya Sanhita (BNS) 2024 Active • 16,000+ Verified Vectors</div>
            <div class="top-badge-pill">⚡ LIVE AI WORKFLOW</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 2. Executive Hero Card
    st.markdown(
        """
        <div class="lexai-super-hero">
            <div class="hero-mesh-ball-1"></div>
            <div class="hero-mesh-ball-2"></div>
            <div class="hero-tag-badge">⚖️ LexAI • Bharat Legal Intelligence • 2026 Edition</div>
            <div class="hero-title-giant">
                Your Rights, Your Procedure. <br><span class="gradient-span">Decoded.</span>
            </div>
            <div class="hero-description-lead">
                Instant, context-aware legal guidance built specifically for India — cross-referencing
                the Bharatiya Nyaya Sanhita (BNS), BNSS / CrPC, Consumer Protection Act, and Labour Codes.
            </div>
            <div class="hero-chips-row">
                <span class="hero-chip-pill">⚡ Groq Llama 3.3 (70B) Inference</span>
                <span class="hero-chip-pill">📚 16,000+ Pinecone Legal Chunks</span>
                <span class="hero-chip-pill">🌐 SerpAPI Live Web Fallback</span>
                <span class="hero-chip-pill">🔒 Encrypted Local SQLite Storage</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 3. How It Works Pipeline Visual
    st.markdown(
        """
        <div class="dag-flow-container">
            <div class="dag-step-card">
                <div class="dag-circle-icon" style="background: #2563eb;">1</div>
                <div class="dag-step-name">Classify & Route</div>
                <div class="dag-step-info">Detects Indian legal domain vs general inquiry</div>
            </div>
            <div class="dag-step-card">
                <div class="dag-circle-icon" style="background: #7c3aed;">2</div>
                <div class="dag-step-name">Refine & Risk Score</div>
                <div class="dag-step-info">Extracts statutes & assigns Low/Med/High severity</div>
            </div>
            <div class="dag-step-card">
                <div class="dag-circle-icon" style="background: #d97706;">3</div>
                <div class="dag-step-name">Cluster RAG Search</div>
                <div class="dag-step-info">Vector DB search with live web fallback</div>
            </div>
            <div class="dag-step-card">
                <div class="dag-circle-icon" style="background: #059669;">4</div>
                <div class="dag-step-name">Action Roadmap</div>
                <div class="dag-step-info">Structured advice, citizen rights & advocate flag</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 4. Interactive BNS vs IPC Navigator
    render_bns_ipc_matrix()

    # 5. Colorful Category Starter Cards
    st.markdown('<div class="scenario-section-header">💡 Frequently Explored Legal Scenarios (One-Click Inquiries)</div>', unsafe_allow_html=True)
    cols = st.columns(2)

    cat_class_map = {
        "🏠 Tenant & Property": "tenant",
        "👩 Women's Rights": "women",
        "🛡️ Criminal & FIR": "criminal",
        "💻 Cyber Crime": "cyber",
        "🛒 Consumer Rights": "consumer",
        "👨‍👩‍👧 Family Law": "family",
        "👷 Labour & Employment": "labour",
        "📜 General Legal": "general",
    }

    categories = list(STARTER_QUESTIONS.items())
    for idx, (cat_name, questions) in enumerate(categories):
        col = cols[idx % 2]
        css_cls = cat_class_map.get(cat_name, "general")
        with col:
            st.markdown(
                f"""
                <div class="category-banner-card {css_cls}">
                    <div style="font-weight: 800; font-size: 1.05rem; margin-bottom: 8px;">{cat_name}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            for q in questions:
                if st.button(q, key=f"starter_{idx}_{q[:20]}", use_container_width=True):
                    if on_select_query:
                        on_select_query(q)
                    st.session_state["pending_query"] = q
                    st.rerun()


def render_pipeline_status(status_events: list[dict]) -> None:
    """Render real-time node-by-node execution pipeline in st.status."""
    with st.status("🧠 Orchestrating Indian Legal Analysis DAG...", expanded=True) as status_box:
        for ev in status_events:
            title = ev.get("title", ev.get("node", ""))
            msg = ev.get("message", "")
            if ev.get("status") == "done":
                st.write(f"✅ **{title}** — {msg}")
            else:
                st.write(f"⏳ **{title}** — {msg}")
        status_box.update(label="🎉 Multi-Stage Legal Workflow Completed", state="complete", expanded=False)


def render_pipeline_expander(metadata: dict) -> None:
    """Renders a 'How I Answered This' transparent DAG execution inspection."""
    refined = metadata.get("refined_query", "")
    cluster = metadata.get("matched_cluster")
    score = metadata.get("confidence_score")
    source = metadata.get("source", "RAG")
    latency = metadata.get("latency")

    cluster_text = f"Cluster #{cluster}" if cluster is not None else "Direct semantic match"
    score_text = f"{score:.2f}" if score is not None else "N/A"
    latency_text = f"{latency}s" if latency else "fast"

    with st.expander("🔍 How I Answered This (Pipeline DAG Transparency)", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Source Engine", "Pinecone Vector DB" if source == "RAG" else "SerpAPI Web Search")
        with c2:
            st.metric("Vector Similarity", score_text)
        with c3:
            st.metric("Total Latency", latency_text)

        st.markdown(f"**Refined Indian Law Query**: `{refined}`")
        if cluster is not None:
            st.caption(f"Scoped to nearest semantic document cluster: **{cluster_text}**")


def render_structured_answer(
    message: dict,
    db: Any,
    on_followup: Optional[Callable[[str], None]] = None,
) -> None:
    """
    Renders an executive, modern structured legal response card:
      - Top Segmented Meta Panel (Risk pill, Source pill, Confidence bar)
      - Lawyer Advisory Banner
      - Tabbed Deep Dive (Summary, Relevant Statutes, Interactive Checklist Roadmap, Rights, Sources)
      - Action Toolbar (Feedback, Bookmark, Markdown, PDF)
      - Follow-up recommendation chips
      - Disclaimer notice
    """
    msg_id = message.get("id")
    content = message.get("content", "")
    meta = message.get("metadata", {}) or {}
    sections = meta.get("answer_sections", {}) or {}

    risk = meta.get("risk", "low").lower()
    source = meta.get("source", "RAG")
    confidence = meta.get("confidence_score", 0.75)
    lawyer_needed = meta.get("lawyer_needed", sections.get("lawyer_needed", "Optional"))
    lawyer_reason = meta.get("lawyer_reason", sections.get("lawyer_reason", ""))

    summary = sections.get("legal_summary") or content
    relevant_laws = sections.get("relevant_laws", [])
    roadmap_steps = sections.get("action_roadmap", [])
    rights = sections.get("important_rights", [])
    final_note = sections.get("final_note", "")
    disclaimer = sections.get("disclaimer", "Information is for general guidance only.")

    # 1. Top Segmented Meta Panel
    conf_pct = int((confidence if confidence is not None else 0.5) * 100)

    st.markdown(
        f"""
        <div class="answer-box-card">
            <div class="stripe-accent-bar"></div>
            <div class="answer-status-pill-row">
                <div style="display: flex; flex-wrap: wrap; align-items: center; gap: 10px;">
                    <span class="risk-halo-pill {risk}">
                        {risk_emoji(risk)} RISK: {risk.upper()} • {risk_description(risk)}
                    </span>
                    <span class="source-halo-pill">
                        {'📚 Pinecone Legal VectorDB' if source == 'RAG' else '🌐 Live Indian Web Search'}
                    </span>
                </div>
                <div class="confidence-fill-box">
                    <span>Confidence: <b>{conf_pct}%</b></span>
                    <div class="confidence-rail">
                        <div class="confidence-slider" style="width: {conf_pct}%;"></div>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 2. Lawyer Advisory Banner
    lawyer_cls = lawyer_needed.lower().replace(" ", "-")
    lawyer_icon = "🚨" if "require" in lawyer_cls else ("⚖️" if "optional" in lawyer_cls else "✅")
    st.markdown(
        f"""
        <div class="premium-lawyer-banner {lawyer_cls}">
            <div class="lawyer-badge-large-icon">{lawyer_icon}</div>
            <div>
                <div class="lawyer-banner-main">Advocate / Lawyer: {lawyer_needed}</div>
                <div class="lawyer-banner-sub">{lawyer_reason or 'Formal legal consultation is advised before executing filings.'}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 3. Tabs: Deep-Dive Presentation
    tabs = st.tabs(["📋 Legal Summary", "📜 Relevant Statutes", "🗺️ Action Roadmap", "🛡️ Citizen Rights", "📑 Raw Sources"])

    with tabs[0]:
        st.markdown(summary)
        if final_note:
            st.info(f"💡 **Practical Advisory**: {final_note}")

    with tabs[1]:
        if relevant_laws:
            st.markdown("##### 🏛️ Applicable Statutory Provisions")
            for item in relevant_laws:
                sec = item.get("section", "")
                desc = item.get("description", "")
                st.markdown(
                    f"""
                    <span class="statute-code-chip">§ {sec}</span> {desc}
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.markdown("General procedural doctrines of Indian Law apply to this scenario.")

    with tabs[2]:
        st.markdown("##### 🗺️ Step-by-Step Action Plan")
        st.caption("Check off items as you proceed; your progress is automatically saved.")

        if roadmap_steps:
            progress = db.get_roadmap_progress(msg_id) if (db and msg_id) else {}
            completed_count = sum(1 for v in progress.values() if v)
            total_count = len(roadmap_steps)
            progress_pct = completed_count / total_count if total_count > 0 else 0.0

            st.progress(progress_pct, text=f"Progress: {completed_count}/{total_count} steps completed")

            for step_idx, step_text in enumerate(roadmap_steps, 1):
                col_check, col_text = st.columns([0.08, 0.92])
                with col_check:
                    is_done = progress.get(step_idx, False)
                    checked = st.checkbox(
                        label=f"Step {step_idx}",
                        value=is_done,
                        key=f"step_{msg_id}_{step_idx}",
                        label_visibility="collapsed",
                    )
                    if checked != is_done and db and msg_id:
                        db.update_roadmap_step(msg_id, step_idx, checked)
                with col_text:
                    st.markdown(f"**Step {step_idx}:** {step_text}")
        else:
            st.markdown("1. File formal written grievance with jurisdictional authority.\n2. Retain date-stamped acknowledgments.")

    with tabs[3]:
        if rights:
            st.markdown("##### 🛡️ Citizen Legal Rights Under Indian Law")
            for r in rights:
                st.markdown(f"• **{r}**")
        else:
            st.markdown("Guaranteed fundamental rights under Part III of the Constitution of India apply.")

    with tabs[4]:
        chunks = meta.get("retrieved_chunks", [])
        web_res = meta.get("web_results", [])

        if chunks:
            st.markdown("**Retrieved Knowledge Base Chunks (Pinecone):**")
            for i, chunk in enumerate(chunks, 1):
                with st.expander(f"Chunk {i} — Relevance Score: {chunk.get('score', 'N/A')}"):
                    st.text(chunk.get("text", ""))

        if web_res:
            st.markdown("**Live Web Context Sources (SerpAPI):**")
            for w in web_res:
                st.markdown(f"- [{w.get('title')}]({w.get('link')}) — *{w.get('snippet')}*")

        if not chunks and not web_res:
            st.caption("Statutory context synthesized from general legal knowledge base.")

    # 4. Pipeline Transparency Expander
    render_pipeline_expander(meta)

    # 5. Action Toolbar
    st.markdown("<hr style='margin: 14px 0 10px 0;'>", unsafe_allow_html=True)
    col1, col2, col3, col4, col5 = st.columns([0.22, 0.22, 0.22, 0.17, 0.17])

    with col1:
        if st.button("👍 Helpful", key=f"thumb_up_{msg_id}", use_container_width=True):
            if db and msg_id:
                db.add_feedback(msg_id, 1)
            st.toast("Thank you for your feedback!", icon="🙏")

    with col2:
        if st.button("👎 Inaccurate", key=f"thumb_down_{msg_id}", use_container_width=True):
            if db and msg_id:
                db.add_feedback(msg_id, -1)
            st.toast("Feedback recorded.", icon="📝")

    with col3:
        is_bm = db.is_bookmarked(msg_id) if (db and msg_id) else False
        btn_label = "🔖 Saved" if is_bm else "⭐ Bookmark"
        if st.button(btn_label, key=f"bm_{msg_id}", use_container_width=True):
            if db and msg_id:
                if not is_bm:
                    db.add_bookmark(msg_id)
                    st.toast("Added to Saved Answers!", icon="⭐")
                    st.rerun()

    with col4:
        md_data = export_markdown("Legal Consultation", [message])
        st.download_button(
            label="📄 MD",
            data=md_data,
            file_name="lexai_consultation.md",
            mime="text/markdown",
            key=f"dl_md_{msg_id}",
            use_container_width=True,
        )

    with col5:
        pdf_bytes = export_pdf("Legal Consultation", [message])
        st.download_button(
            label="📑 PDF",
            data=pdf_bytes,
            file_name="lexai_consultation.pdf",
            mime="application/pdf",
            key=f"dl_pdf_{msg_id}",
            use_container_width=True,
        )

    # 6. Follow-up Suggestions
    followups = meta.get("followups", [])
    if followups:
        st.markdown("##### 🔍 Explore Related Questions:")
        f_cols = st.columns(len(followups))
        for f_idx, f_q in enumerate(followups):
            with f_cols[f_idx]:
                if st.button(f_q, key=f"fup_{msg_id}_{f_idx}", use_container_width=True):
                    if on_followup:
                        on_followup(f_q)
                    st.session_state["pending_query"] = f_q
                    st.rerun()

    # 7. Compliance Disclaimer
    st.markdown(
        f"""
        <div style="margin-top: 1.5rem; padding: 12px 16px; border-radius: 12px; background: rgba(148, 163, 184, 0.08); border-left: 4px solid #d97706; font-size: 0.78rem; color: #64748b; line-height: 1.5;">
            ⚖️ <b>Legal Disclaimer</b>: {disclaimer}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_helplines_card() -> None:
    """Render Emergency Helplines quick card with dial pill badges."""
    st.markdown("### 🚨 Emergency Legal & Crisis Helplines (India)")
    st.caption("Official 24/7 national toll-free helplines for immediate legal assistance and intervention.")

    for h in HELPLINES:
        st.markdown(
            f"""
            <div class="sos-helpline-card">
                <div>
                    <div style="font-size: 1.15rem; font-weight: 800;">{h['icon']} {h['name']}</div>
                    <div style="font-size: 0.82rem; color: #64748b; margin-top: 2px;">{h['desc']}</div>
                </div>
                <div class="sos-dial-badge">{h['number']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_glossary_view() -> None:
    """Render interactive legal jargon glossary with pill badges."""
    st.markdown("### 📖 Indian Legal Terms Glossary")
    st.caption("Plain-language definitions for court terms, criminal procedure, and statutory phrases.")

    search = st.text_input("🔍 Search term or abbreviation...", placeholder="e.g. FIR, BNS, Bail, PIL, Caveat")
    filtered = {
        k: v for k, v in LEGAL_GLOSSARY.items()
        if not search or search.lower() in k.lower() or search.lower() in v.lower()
    }

    cols = st.columns(2)
    for i, (term, definition) in enumerate(filtered.items()):
        with cols[i % 2]:
            with st.container(border=True):
                st.markdown(f"**⚖️ {term}**")
                st.markdown(f"<span style='font-size: 0.9rem; color: #475569;'>{definition}</span>", unsafe_allow_html=True)
