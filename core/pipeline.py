"""
LangGraph orchestration pipeline for LexAI.
Preserves the exact DAG logic from agent.ipynb while exposing a generator
run_query() that yields real-time node progress events, and returns
a structured dict with parsed sections and metadata.
"""

import time
from typing import Any, Generator, Optional, TypedDict

try:
    from langgraph.graph import END, StateGraph
    _HAS_LANGGRAPH = True
except ImportError:
    StateGraph = None
    END = "__end__"
    _HAS_LANGGRAPH = False

from core.config import LAW_KEYWORDS, RAG_THRESHOLD, RAG_TOP_K
from core.llm import (
    call_llm,
    generate_followup_questions,
)
from core.prompts import (
    GENERAL_RESPONSE,
    LEGAL_EXPERT,
    QUERY_REFINER,
    RISK_CLASSIFIER,
    ROADMAP_GENERATOR,
    STYLE_DETAILED,
    STYLE_SIMPLE,
)
from core.rag import search_rag
from core.search import search_web
from utils.helpers import parse_risk_response, parse_structured_answer


# ── TypedDict for LangGraph State ─────────────────────────────────────────

class GraphState(TypedDict):
    query: str
    domain: str
    refined_query: str
    risk: str
    risk_reason: str
    context: str
    score: float
    matched_cluster: Optional[int]
    retrieved_chunks: list[dict]
    web_results: list[dict]
    source: str
    roadmap: str
    final_answer: str
    structured_sections: dict[str, Any]
    style: str
    groq_api_key: Optional[str]
    pinecone_api_key: Optional[str]
    serpapi_api_key: Optional[str]


# ── Individual Nodes (Preserving exact logic from agent.ipynb) ────────────

def domain_classifier_node(state: GraphState) -> GraphState:
    """Classifies if query is law-related or general."""
    query = state["query"].lower()
    domain = "law" if any(k in query for k in LAW_KEYWORDS) else "general"
    return {**state, "domain": domain}


def domain_router(state: GraphState) -> str:
    return "law" if state.get("domain") == "law" else "general"


def general_llm_node(state: GraphState) -> GraphState:
    """Direct LLM response for non-law queries."""
    prompt = GENERAL_RESPONSE.format(query=state["query"])
    answer = call_llm(prompt, api_key=state.get("groq_api_key"))
    return {
        **state,
        "final_answer": answer,
        "source": "GENERAL",
        "structured_sections": {
            "legal_summary": answer,
            "disclaimer": "This is a general inquiry response, not legal information.",
            "lawyer_needed": "Not Needed",
            "lawyer_reason": "Non-legal matter.",
            "relevant_laws": [],
            "action_roadmap": [],
            "important_rights": [],
            "final_note": "Feel free to ask any Indian law questions if you need legal guidance.",
        },
    }


def refine_query_node(state: GraphState) -> GraphState:
    """Converts user situation into a precise Indian legal search query."""
    prompt = QUERY_REFINER.format(query=state["query"])
    refined = call_llm(
        prompt=prompt,
        system_prompt="You are an Indian Legal Research Expert. Output only the refined query string.",
        temperature=0.2,
        api_key=state.get("groq_api_key"),
    )
    # Fallback to original query if LLM fails or repeats preamble
    refined = refined.strip().strip('"\'')
    if not refined or "Demo Mode" in refined:
        refined = f"{state['query']} under Indian Law"
    return {**state, "refined_query": refined}


def risk_classifier_node(state: GraphState) -> GraphState:
    """Classifies risk level: LOW / MEDIUM / HIGH with reasoning."""
    prompt = RISK_CLASSIFIER.format(query=state["query"])
    raw_response = call_llm(
        prompt=prompt,
        system_prompt="You are a Legal Risk Assessment Expert.",
        temperature=0.2,
        api_key=state.get("groq_api_key"),
    )
    level, reason = parse_risk_response(raw_response)
    return {**state, "risk": level, "risk_reason": reason}


def rag_node(state: GraphState) -> GraphState:
    """Retrieves top legal chunks from Pinecone using cluster-aware search."""
    search_query = state.get("refined_query") or state["query"]
    rag_result = search_rag(
        query=search_query,
        top_k=RAG_TOP_K,
        api_key=state.get("pinecone_api_key"),
    )
    return {
        **state,
        "context": rag_result.get("context", ""),
        "score": rag_result.get("score", 0.0),
        "matched_cluster": rag_result.get("cluster_id"),
        "retrieved_chunks": rag_result.get("chunks", []),
        "source": "RAG",
    }


def rag_router(state: GraphState) -> str:
    """Routes to SerpAPI if RAG score is below threshold; otherwise to roadmap."""
    score = state.get("score", 0.0)
    threshold = RAG_THRESHOLD
    if score < threshold or not state.get("context"):
        return "serp"
    return "roadmap"


def serp_node(state: GraphState) -> GraphState:
    """Fetches live web context restricted to Indian Law."""
    refined = state.get("refined_query") or state["query"]
    search_query = f"{refined} under Indian Law BNS IPC"
    serp_result = search_web(
        query=search_query,
        api_key=state.get("serpapi_api_key"),
    )

    existing = state.get("context", "")
    web_text = serp_result.get("web_context", "")
    new_context = f"{existing}\n\n=== INDIAN WEB CONTEXT ===\n{web_text}".strip()

    return {
        **state,
        "context": new_context,
        "web_results": serp_result.get("results", []),
        "source": "WEB",
    }


def roadmap_node(state: GraphState) -> GraphState:
    """Generates a practical legal roadmap based on context."""
    context = state.get("context") or "No specific statute context found."
    prompt = ROADMAP_GENERATOR.format(query=state["query"], context=context)
    roadmap = call_llm(
        prompt=prompt,
        system_prompt="You are a legal procedure advisor for Indian citizens.",
        temperature=0.3,
        api_key=state.get("groq_api_key"),
    )
    return {**state, "roadmap": roadmap}


def legal_expert_node(state: GraphState) -> GraphState:
    """Synthesizes all context into structured Indian legal advice."""
    style = state.get("style", "detailed")
    style_instruction = STYLE_SIMPLE if style == "simple" else STYLE_DETAILED
    detail_level = "simple, plain-language" if style == "simple" else "thorough and well-reasoned"

    context = state.get("context") or "No direct statute context found."
    risk_level = (state.get("risk") or "low").upper()

    prompt = LEGAL_EXPERT.format(
        context=context,
        query=state["query"],
        risk=risk_level,
        style_instruction=style_instruction,
        detail_level=detail_level,
    )

    final_answer = call_llm(
        prompt=prompt,
        system_prompt="You are a STRICT Indian legal assistant. Follow the exact section headers requested.",
        temperature=0.3,
        api_key=state.get("groq_api_key"),
    )

    structured = parse_structured_answer(final_answer)
    return {
        **state,
        "final_answer": final_answer,
        "structured_sections": structured,
    }


# ── Build LangGraph DAG (Backward Compatible) ──────────────────────────────

def build_graph() -> Any:
    """Compiles and returns the LangGraph workflow DAG."""
    if not _HAS_LANGGRAPH or StateGraph is None:
        return None
    graph = StateGraph(GraphState)

    graph.add_node("domain", domain_classifier_node)
    graph.add_node("general", general_llm_node)
    graph.add_node("refine", refine_query_node)
    graph.add_node("risk", risk_classifier_node)
    graph.add_node("rag", rag_node)
    graph.add_node("serp", serp_node)
    graph.add_node("roadmap", roadmap_node)
    graph.add_node("final", legal_expert_node)

    graph.set_entry_point("domain")

    graph.add_conditional_edges(
        "domain",
        domain_router,
        {"law": "refine", "general": "general"},
    )
    graph.add_edge("refine", "risk")
    graph.add_edge("risk", "rag")

    graph.add_conditional_edges(
        "rag",
        rag_router,
        {"roadmap": "roadmap", "serp": "serp"},
    )

    graph.add_edge("serp", "roadmap")
    graph.add_edge("roadmap", "final")
    graph.add_edge("final", END)
    graph.add_edge("general", END)

    return graph.compile()


# Global compiled app handle for direct invocation
langgraph_app = build_graph()


# ── Interactive Generator Pipeline for Streamlit ──────────────────────────

def run_query(
    query: str,
    style: str = "detailed",
    rag_threshold: float = RAG_THRESHOLD,
    top_k: int = RAG_TOP_K,
    groq_api_key: Optional[str] = None,
    pinecone_api_key: Optional[str] = None,
    serpapi_api_key: Optional[str] = None,
) -> Generator[dict[str, Any], None, dict[str, Any]]:
    """
    Step-by-step pipeline execution yielding progress events.

    Yields:
      {"node": str, "status": "running"|"done", "data": dict}

    Final return/yield:
      {"node": "complete", "status": "done", "result": dict}
    """
    start_time = time.time()

    state: GraphState = {
        "query": query,
        "domain": "",
        "refined_query": "",
        "risk": "low",
        "risk_reason": "",
        "context": "",
        "score": 0.0,
        "matched_cluster": None,
        "retrieved_chunks": [],
        "web_results": [],
        "source": "RAG",
        "roadmap": "",
        "final_answer": "",
        "structured_sections": {},
        "style": style,
        "groq_api_key": groq_api_key,
        "pinecone_api_key": pinecone_api_key,
        "serpapi_api_key": serpapi_api_key,
    }

    # 1. Domain classification
    yield {
        "node": "domain",
        "status": "running",
        "title": "Classifying Domain",
        "message": "Checking if query pertains to Indian law...",
    }
    t0 = time.time()
    state = domain_classifier_node(state)
    domain_time = round(time.time() - t0, 2)
    yield {
        "node": "domain",
        "status": "done",
        "title": f"Domain: {state['domain'].upper()}",
        "message": f"Identified as {state['domain']} domain ({domain_time}s)",
        "data": {"domain": state["domain"], "duration": domain_time},
    }

    if state["domain"] != "law":
        # General response branch
        yield {
            "node": "general",
            "status": "running",
            "title": "Generating Answer",
            "message": "Providing direct response for general inquiry...",
        }
        t0 = time.time()
        state = general_llm_node(state)
        gen_time = round(time.time() - t0, 2)

        followups = generate_followup_questions(
            query=query,
            summary=state["final_answer"],
            api_key=groq_api_key,
        )

        total_time = round(time.time() - start_time, 2)
        final_result = {
            "query": query,
            "refined_query": query,
            "domain": "general",
            "risk": "low",
            "risk_reason": "Non-legal matter",
            "source": "GENERAL",
            "confidence_score": 1.0,
            "matched_cluster": None,
            "retrieved_chunks": [],
            "web_results": [],
            "answer_sections": state["structured_sections"],
            "raw_answer": state["final_answer"],
            "lawyer_needed": "Not Needed",
            "lawyer_reason": "General question.",
            "followups": followups,
            "latency": total_time,
        }
        yield {
            "node": "complete",
            "status": "done",
            "title": "Response Ready",
            "message": f"Completed in {total_time}s",
            "result": final_result,
        }
        return final_result

    # 2. Query refinement
    yield {
        "node": "refine",
        "status": "running",
        "title": "Refining Legal Query",
        "message": "Mapping situation to Indian statutes (IPC/BNS, CrPC/BNSS)...",
    }
    t0 = time.time()
    state = refine_query_node(state)
    refine_time = round(time.time() - t0, 2)
    yield {
        "node": "refine",
        "status": "done",
        "title": "Query Refined",
        "message": f"Search terms: '{state['refined_query'][:60]}...' ({refine_time}s)",
        "data": {"refined_query": state["refined_query"], "duration": refine_time},
    }

    # 3. Risk classification
    yield {
        "node": "risk",
        "status": "running",
        "title": "Assessing Legal Risk",
        "message": "Evaluating urgency, rights violation, or imminent arrest danger...",
    }
    t0 = time.time()
    state = risk_classifier_node(state)
    risk_time = round(time.time() - t0, 2)
    yield {
        "node": "risk",
        "status": "done",
        "title": f"Risk Level: {state['risk'].upper()}",
        "message": f"{state['risk_reason']} ({risk_time}s)",
        "data": {"risk": state["risk"], "reason": state["risk_reason"], "duration": risk_time},
    }

    # 4. RAG Node
    yield {
        "node": "rag",
        "status": "running",
        "title": "Querying Knowledge Base",
        "message": "Scanning Pinecone vector clusters for Indian statutes and precedents...",
    }
    t0 = time.time()
    state = rag_node(state)
    rag_time = round(time.time() - t0, 2)
    cluster_str = f"Cluster {state['matched_cluster']}" if state['matched_cluster'] is not None else "No cluster"
    yield {
        "node": "rag",
        "status": "done",
        "title": f"Vector Search: Score {state['score']:.2f}",
        "message": f"Matched {cluster_str} with {len(state['retrieved_chunks'])} chunks ({rag_time}s)",
        "data": {
            "score": state["score"],
            "cluster_id": state["matched_cluster"],
            "chunks_count": len(state["retrieved_chunks"]),
            "duration": rag_time,
        },
    }

    # 5. Routing: Check if SERP needed
    if state["score"] < rag_threshold or not state["context"]:
        yield {
            "node": "serp",
            "status": "running",
            "title": "Live Web Search Fallback",
            "message": f"Confidence ({state['score']:.2f}) < threshold ({rag_threshold:.2f}). Querying SerpAPI...",
        }
        t0 = time.time()
        state = serp_node(state)
        serp_time = round(time.time() - t0, 2)
        yield {
            "node": "serp",
            "status": "done",
            "title": "Web Search Complete",
            "message": f"Retrieved {len(state['web_results'])} Indian legal web references ({serp_time}s)",
            "data": {"results_count": len(state["web_results"]), "duration": serp_time},
        }

    # 6. Roadmap generation
    yield {
        "node": "roadmap",
        "status": "running",
        "title": "Building Action Roadmap",
        "message": "Formulating procedural legal roadmap...",
    }
    t0 = time.time()
    state = roadmap_node(state)
    roadmap_time = round(time.time() - t0, 2)
    yield {
        "node": "roadmap",
        "status": "done",
        "title": "Roadmap Prepared",
        "message": f"Action plan formulated ({roadmap_time}s)",
        "data": {"duration": roadmap_time},
    }

    # 7. Final Legal Expert Synthesis
    yield {
        "node": "final",
        "status": "running",
        "title": "Synthesizing Legal Advice",
        "message": "Generating structured response with statutory citations & rights...",
    }
    t0 = time.time()
    state = legal_expert_node(state)
    expert_time = round(time.time() - t0, 2)
    yield {
        "node": "final",
        "status": "done",
        "title": "Synthesis Complete",
        "message": f"Advice drafted ({expert_time}s)",
        "data": {"duration": expert_time},
    }

    # 8. Follow-up suggestions
    sections = state.get("structured_sections", {})
    summary_text = sections.get("legal_summary", state["final_answer"])
    followups = generate_followup_questions(
        query=query,
        summary=summary_text,
        api_key=groq_api_key,
    )

    total_time = round(time.time() - start_time, 2)
    final_result = {
        "query": query,
        "refined_query": state.get("refined_query", query),
        "domain": state.get("domain", "law"),
        "risk": state.get("risk", "low"),
        "risk_reason": state.get("risk_reason", ""),
        "source": state.get("source", "RAG"),
        "confidence_score": round(state.get("score", 0.0), 2),
        "matched_cluster": state.get("matched_cluster"),
        "retrieved_chunks": state.get("retrieved_chunks", []),
        "web_results": state.get("web_results", []),
        "answer_sections": state.get("structured_sections", {}),
        "raw_answer": state.get("final_answer", ""),
        "lawyer_needed": sections.get("lawyer_needed", "Optional"),
        "lawyer_reason": sections.get("lawyer_reason", ""),
        "followups": followups,
        "latency": total_time,
    }

    yield {
        "node": "complete",
        "status": "done",
        "title": "Analysis Ready",
        "message": f"All nodes executed in {total_time}s",
        "result": final_result,
    }

    return final_result
