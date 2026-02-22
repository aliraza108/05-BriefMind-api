"""
BriefMind - AI Executive Briefing Generator
FastAPI + OpenAI Agents SDK
"""

import os
import json
import uuid
from typing import Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from agents import Agent, Runner, set_default_openai_api, set_default_openai_client, set_tracing_disabled, AsyncOpenAI
from briefmind_tools import (
    chunk_and_extract_tool,
    merge_insights_tool,
    build_concept_map_tool,
    extract_decisions_tool,
    generate_executive_summary_tool,
    answer_question_tool,
)
from document_store import DocumentStore

# ──────────────────────────────────────────────
# Configuration — Gemini via OpenAI-compatible API
# ──────────────────────────────────────────────
import os

API_KEY = os.getenv("API_KEY")

if not API_KEY:
    raise ValueError("API_KEY is not set in environment variables")
BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
MODEL    = "gemini-2.5-flash"

client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY)
set_default_openai_client(client=client)
set_default_openai_api("chat_completions")

set_tracing_disabled(True)

# ──────────────────────────────────────────────
# In-memory document store (swap for Redis/DB in prod)
# ──────────────────────────────────────────────
store = DocumentStore()

# ──────────────────────────────────────────────
# Agents
# ──────────────────────────────────────────────
briefing_agent = Agent(
    name="BriefMind Briefing Agent",
    model=MODEL,
    instructions="""
You are BriefMind, an elite AI knowledge distillation engine.
Your job is to transform long documents into structured executive intelligence.

When processing a document you MUST call tools in this order:
1. chunk_and_extract  → break document into semantic chunks & extract raw insights
2. merge_insights     → hierarchically merge and rank insights by impact
3. extract_decisions  → pull out actions, risks, opportunities, strategic implications
4. build_concept_map  → identify relationships between core concepts
5. generate_executive_summary → produce the final 1-2 min executive summary

Return ALL tool results combined as a single JSON object with keys:
  executive_summary, key_insights, topic_breakdown, decision_intelligence, concept_map

Be precise, executive-focused, and ruthlessly concise.
""",
    tools=[
        chunk_and_extract_tool,
        merge_insights_tool,
        extract_decisions_tool,
        build_concept_map_tool,
        generate_executive_summary_tool,
    ],
)

chat_agent = Agent(
    name="BriefMind Chat Agent",
    model=MODEL,
    instructions="""
You are BriefMind's document Q&A assistant.
You have access to a processed knowledge base of a previously analyzed document.
Use the answer_question tool to retrieve relevant context, then answer clearly and concisely.
Always cite which section of the document your answer comes from.
""",
    tools=[answer_question_tool],
)

# ──────────────────────────────────────────────
# FastAPI App
# ──────────────────────────────────────────────
app = FastAPI(
    title="BriefMind API",
    description="AI Executive Briefing Generator — transforms massive documents into structured intelligence",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────
# Schemas
# ──────────────────────────────────────────────
class ChatRequest(BaseModel):
    doc_id: str
    question: str

class BriefingResponse(BaseModel):
    doc_id: str
    executive_summary: str
    key_insights: list
    topic_breakdown: dict
    decision_intelligence: dict
    concept_map: dict
    status: str = "success"


# ──────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────

@app.get("/", tags=["Health"])
async def root():
    return {
        "name": "BriefMind",
        "tagline": "Knowledge Distillation Engine",
        "version": "1.0.0",
        "endpoints": {
            "POST /upload":    "Upload a document file for processing",
            "POST /process":   "Process raw text directly",
            "GET  /briefing/{doc_id}": "Retrieve a generated briefing",
            "POST /chat":      "Ask questions about a processed document",
            "GET  /docs":      "List all processed documents",
            "DELETE /docs/{doc_id}": "Delete a document",
        }
    }


@app.post("/upload", tags=["Processing"])
async def upload_document(
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
):
    """
    Upload a document file (.txt, .pdf text, .md, .docx text).
    Returns a doc_id and the full executive briefing.
    """
    allowed = {".txt", ".md", ".csv", ".json"}
    ext = os.path.splitext(file.filename)[1].lower()

    # Read content
    raw_bytes = await file.read()
    try:
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="File must be UTF-8 text. For PDFs, extract text first.")

    if len(text.strip()) < 100:
        raise HTTPException(status_code=400, detail="Document too short to process (min 100 chars).")

    doc_id = str(uuid.uuid4())[:8]
    doc_title = title or file.filename

    return await _run_briefing_pipeline(doc_id, doc_title, text)


@app.post("/process", tags=["Processing"])
async def process_text(
    text: str = Form(...),
    title: str = Form("Untitled Document"),
):
    """
    Paste raw text directly for processing.
    Returns a doc_id and the full executive briefing.
    """
    if len(text.strip()) < 100:
        raise HTTPException(status_code=400, detail="Text too short (min 100 chars).")

    doc_id = str(uuid.uuid4())[:8]
    return await _run_briefing_pipeline(doc_id, title, text)


@app.get("/briefing/{doc_id}", tags=["Retrieval"])
async def get_briefing(doc_id: str):
    """Retrieve a previously generated briefing by doc_id."""
    doc = store.get(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found.")
    return doc


@app.post("/chat", tags=["Chat"])
async def chat_with_document(req: ChatRequest):
    """
    Ask any question about a processed document.
    The agent uses the stored knowledge base to answer.
    """
    doc = store.get(req.doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document '{req.doc_id}' not found.")

    # Inject doc context into tool via store
    store.set_active(req.doc_id)

    prompt = f"""
Document ID: {req.doc_id}
Document Title: {doc.get('title', 'Unknown')}

Briefing Context:
{json.dumps(doc.get('briefing', {}), indent=2)[:3000]}

User Question: {req.question}
"""
    result = await Runner.run(chat_agent, prompt)
    return {
        "doc_id": req.doc_id,
        "question": req.question,
        "answer": result.final_output,
        "status": "success"
    }


@app.get("/docs", tags=["Management"])
async def list_documents():
    """List all processed documents."""
    docs = store.list_all()
    return {"documents": docs, "count": len(docs)}


@app.delete("/docs/{doc_id}", tags=["Management"])
async def delete_document(doc_id: str):
    """Delete a document and its briefing from the store."""
    if not store.get(doc_id):
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found.")
    store.delete(doc_id)
    return {"message": f"Document '{doc_id}' deleted.", "status": "success"}


# ──────────────────────────────────────────────
# Internal: Briefing Pipeline
# ──────────────────────────────────────────────
async def _run_briefing_pipeline(doc_id: str, title: str, text: str) -> dict:
    """Run the full briefing agent pipeline on a document."""

    # Store raw text for chat mode
    store.store_raw(doc_id, title, text)

    word_count = len(text.split())
    char_count = len(text)

    prompt = f"""
Process this document and generate a complete executive briefing.

Document ID: {doc_id}
Title: {title}
Word Count: {word_count:,}
Characters: {char_count:,}

--- DOCUMENT START ---
{text[:50000]}
--- DOCUMENT END ---

Call all 5 tools in order and return a complete JSON briefing.
"""

    result = await Runner.run(briefing_agent, prompt)
    raw_output = result.final_output

    # Try to parse structured output
    briefing = _parse_briefing_output(raw_output)

    # Store for later retrieval
    store.store_briefing(doc_id, title, text, briefing)

    return {
        "doc_id": doc_id,
        "title": title,
        "word_count": word_count,
        "briefing": briefing,
        "status": "success",
        "message": f"Document '{title}' processed successfully. Use doc_id '{doc_id}' for chat queries."
    }


def _parse_briefing_output(raw: str) -> dict:
    """Extract structured JSON from agent output."""
    # Try direct JSON parse
    try:
        # Find JSON block if wrapped in markdown
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0].strip()
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0].strip()
        return json.loads(raw)
    except Exception:
        pass

    # Return raw as summary if parsing fails
    return {
        "executive_summary": raw[:2000],
        "key_insights": [],
        "topic_breakdown": {},
        "decision_intelligence": {"actions": [], "risks": [], "opportunities": []},
        "concept_map": {"nodes": [], "edges": []},
    }