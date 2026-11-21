"""
Analytics dashboard page for LexAI.
Uses Plotly to visualize query volumes, risk distribution, RAG vs Web ratios,
confidence trends, and user satisfaction metrics.
"""

from typing import Any, Optional
import streamlit as st


def render_analytics_page(db: Any, user_id: Optional[int] = None) -> None:
    """Render comprehensive analytics dashboard with Plotly charts."""
    st.markdown("## 📊 LexAI Intelligence & Usage Analytics")
    st.caption("Insights into legal query patterns, risk severity, retrieval sources, and system confidence.")

    try:
        import plotly.express as px
        import plotly.graph_objects as go
    except ImportError:
        st.warning("Plotly is required for the analytics dashboard. Please run: pip install plotly")
        return

    # 1. Top KPI Summary Cards
    total_queries = db.get_total_queries(user_id=user_id)
    feedback_stats = db.get_feedback_stats(user_id=user_id)
    conf_data = db.get_confidence_over_time(user_id=user_id, limit=100)

    avg_confidence = 0.0
    if conf_data:
        scores = [d["score"] for d in conf_data if d.get("score") is not None]
        avg_confidence = (sum(scores) / len(scores)) if scores else 0.0

    pos = feedback_stats.get("positive", 0)
    neg = feedback_stats.get("negative", 0)
    tot_feedback = pos + neg
    satisfaction_rate = round((pos / tot_feedback * 100), 1) if tot_feedback > 0 else 100.0

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.metric(label="Total Queries Answered", value=total_queries, delta=f"{total_queries} queries")
    with kpi2:
        st.metric(label="User Satisfaction Rate", value=f"{satisfaction_rate}%", delta=f"{pos} positive")
    with kpi3:
        st.metric(label="Avg RAG Confidence", value=f"{avg_confidence:.2f}", delta="Pinecone Top-k")
    with kpi4:
        st.metric(label="Saved Bookmarks", value=len(db.get_bookmarks(user_id=user_id)))

    st.markdown("---")

    # 2. Charts Row 1: Risk Distribution & Source Ratio
    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.markdown("##### ⚠️ Risk Severity Breakdown")
        risk_data = db.get_risk_distribution(user_id=user_id)
        if risk_data:
            labels = [r["risk"].capitalize() if r.get("risk") else "Unclassified" for r in risk_data]
            values = [r["count"] for r in risk_data]
            colors_map = {
                "High": "#ef4444",
                "Medium": "#f59e0b",
                "Low": "#10b981",
                "Unclassified": "#94a3b8",
            }
            colors = [colors_map.get(l, "#64748b") for l in labels]

            fig_donut = go.Figure(
                data=[
                    go.Pie(
                        labels=labels,
                        values=values,
                        hole=0.55,
                        marker=dict(colors=colors),
                        textinfo="label+percent",
                    )
                ]
            )
            fig_donut.update_layout(
                margin=dict(t=10, b=10, l=10, r=10),
                height=280,
                showlegend=True,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_donut, use_container_width=True)
        else:
            st.info("No risk classification data yet. Ask questions in the chat to see insights.")

    with col_chart2:
        st.markdown("##### 📚 Retrieval Source: RAG vs. Web Fallback")
        source_data = db.get_source_distribution(user_id=user_id)
        if source_data:
            labels = [
                "Knowledge Base (Pinecone)" if s["source"] == "RAG" else ("Live Web Search" if s["source"] == "WEB" else "General LLM")
                for s in source_data
            ]
            values = [s["count"] for s in source_data]
            colors = ["#1e3a8a", "#0284c7", "#64748b"]

            fig_pie = go.Figure(
                data=[
                    go.Pie(
                        labels=labels,
                        values=values,
                        marker=dict(colors=colors),
                        textinfo="label+percent",
                    )
                ]
            )
            fig_pie.update_layout(
                margin=dict(t=10, b=10, l=10, r=10),
                height=280,
                showlegend=True,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("No source distribution data yet. Ask questions in the chat to populate.")

    st.markdown("---")

    # 3. Charts Row 2: Queries over time & Confidence Trend
    col_chart3, col_chart4 = st.columns(2)

    with col_chart3:
        st.markdown("##### 📈 Query Volume Timeline")
        timeline_data = db.get_queries_by_date(user_id=user_id, days=30)
        if timeline_data:
            dates = [d["date"] for d in timeline_data]
            counts = [d["count"] for d in timeline_data]

            fig_line = px.bar(
                x=dates,
                y=counts,
                labels={"x": "Date", "y": "Queries"},
                color_discrete_sequence=["#2563eb"],
            )
            fig_line.update_layout(
                margin=dict(t=10, b=10, l=10, r=10),
                height=260,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_line, use_container_width=True)
        else:
            st.info("Query volume timeline will populate as messages are sent.")

    with col_chart4:
        st.markdown("##### 🎯 Knowledge Base Confidence Trend")
        if conf_data:
            x_vals = [i for i in range(1, len(conf_data) + 1)]
            y_vals = [d["score"] for d in reversed(conf_data)]

            fig_conf = px.line(
                x=x_vals,
                y=y_vals,
                labels={"x": "Recent Queries", "y": "Similarity Score"},
                color_discrete_sequence=["#10b981"],
            )
            fig_conf.add_hline(y=0.5, line_dash="dash", line_color="#f59e0b", annotation_text="Fallback Threshold (0.5)")
            fig_conf.update_layout(
                margin=dict(t=10, b=10, l=10, r=10),
                height=260,
                yaxis_range=[0, 1],
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_conf, use_container_width=True)
        else:
            st.info("Confidence score timeline will display once RAG queries run.")
