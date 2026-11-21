"""
Groq LLM integration and helper utilities for LexAI.
Supports both non-streaming and streaming inference, title generation,
and follow-up generation. Gracefully handles missing keys.
"""

import os
from typing import Any, Generator, Optional

try:
    from groq import Groq
    _HAS_GROQ = True
except ImportError:
    Groq = None
    _HAS_GROQ = False

from core.config import GROQ_API_KEY, GROQ_MODEL, LLM_TEMPERATURE
from core.prompts import (
    FOLLOWUP_GENERATOR,
    TITLE_GENERATOR,
)


def get_groq_client(api_key: Optional[str] = None) -> Optional[Any]:
    """Return an initialized Groq client or None if API key is not set."""
    if not _HAS_GROQ or Groq is None:
        return None
    key = api_key or GROQ_API_KEY or os.getenv("GROQ_API_KEY")
    if not key or key.strip() in ("", "your_groq_api_key_here"):
        return None
    try:
        return Groq(api_key=key.strip())
    except Exception as e:
        print(f"Error initializing Groq client: {e}")
        return None


def call_llm(
    prompt: str,
    system_prompt: str = "You are a helpful legal AI assistant specializing in Indian law.",
    model: Optional[str] = None,
    temperature: float = LLM_TEMPERATURE,
    api_key: Optional[str] = None,
) -> str:
    """
    Standard LLM call using Groq.
    Falls back to a clear mock message if client cannot be initialized.
    """
    client = get_groq_client(api_key)
    if not client:
        return (
            "[Demo Mode] Groq API key is not configured. Please add GROQ_API_KEY in "
            "the Settings tab or in your .env file to generate live legal answers."
        )

    selected_model = model or GROQ_MODEL
    try:
        response = client.chat.completions.create(
            model=selected_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error communicating with Groq API ({selected_model}): {str(e)}"


def stream_llm(
    prompt: str,
    system_prompt: str = "You are a helpful legal AI assistant specializing in Indian law.",
    model: Optional[str] = None,
    temperature: float = LLM_TEMPERATURE,
    api_key: Optional[str] = None,
) -> Generator[str, None, None]:
    """
    Stream tokens from Groq LLM.
    Yields chunks of text as they arrive.
    """
    client = get_groq_client(api_key)
    if not client:
        demo_msg = (
            "### Disclaimer\n"
            "This information is provided for educational and general guidance only and should not be considered "
            "as formal legal advice. Please consult a qualified legal professional for your specific situation.\n\n"
            "### Lawyer Needed\n"
            "**Optional**\n"
            "Consulting an advocate is recommended to evaluate the facts and review documentary evidence.\n\n"
            "### Legal Summary\n"
            "Under Indian law, citizens and residents are protected by statutory frameworks and constitutional "
            "guarantees. In civil and criminal matters, timely documentation and appropriate forum selection are vital.\n\n"
            "### Relevant Laws\n"
            "- **Bharatiya Nyaya Sanhita (BNS) / IPC**: Substantive criminal provisions\n"
            "- **Bharatiya Nagarik Suraksha Sanhita (BNSS) / CrPC**: Procedure for reporting and inquiry\n"
            "- **Constitution of India**: Fundamental rights protection\n\n"
            "### Action Roadmap\n"
            "1. Gather and document all relevant communications, notices, and receipts.\n"
            "2. Send a formal legal notice or approach the concerned jurisdictional authority.\n"
            "3. Seek consultation with an advocate specializing in this subject if needed.\n\n"
            "### Important Rights\n"
            "- Right to fair hearing and due process\n"
            "- Right to receive acknowledgment for written complaints\n\n"
            "### Final Note\n"
            "Keep written records of every correspondence. Please configure your GROQ_API_KEY for live inferences."
        )
        for word in demo_msg.split(" "):
            yield word + " "
        return

    selected_model = model or GROQ_MODEL
    try:
        stream = client.chat.completions.create(
            model=selected_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            stream=True,
        )
        for chunk in stream:
            content = chunk.choices[0].delta.content or ""
            if content:
                yield content
    except Exception as e:
        yield f"\n\n[Error streaming response: {str(e)}]"


def generate_chat_title(query: str, api_key: Optional[str] = None) -> str:
    """Generate a 3-6 word descriptive chat title based on the first query."""
    prompt = TITLE_GENERATOR.format(query=query)
    title = call_llm(
        prompt=prompt,
        system_prompt="You generate concise 3 to 6 word titles for legal conversations.",
        temperature=0.2,
        api_key=api_key,
    )
    cleaned = title.strip().strip('"\'#*').replace("\n", " ")
    if len(cleaned) > 50 or "Demo Mode" in cleaned or "Error" in cleaned:
        words = query.strip().split()
        return " ".join(words[:5]) + ("..." if len(words) > 5 else "")
    return cleaned


def generate_followup_questions(
    query: str,
    summary: str,
    api_key: Optional[str] = None,
) -> list[str]:
    """Generate 3 contextual follow-up questions."""
    prompt = FOLLOWUP_GENERATOR.format(query=query, summary=summary[:400])
    raw = call_llm(
        prompt=prompt,
        system_prompt="You suggest 3 short, relevant legal follow-up questions.",
        temperature=0.4,
        api_key=api_key,
    )
    questions = []
    for line in raw.split("\n"):
        line = line.strip()
        if line and any(line.startswith(prefix) for prefix in ("1.", "2.", "3.", "-", "*")):
            cleaned = line.lstrip("1234567890.-*• ").strip()
            if cleaned and len(cleaned) > 5:
                questions.append(cleaned)
    if not questions:
        questions = [
            "What documents or evidence do I need to collect?",
            "What is the limitation period or deadline to take action?",
            "What are the expected court fees or costs involved?",
        ]
    return questions[:3]
