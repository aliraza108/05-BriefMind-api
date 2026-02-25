"""
BriefMind - AI Executive Briefing Generator
FastAPI + OpenAI Agents SDK
"""

import os
import json
import uuid
from typing import Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Request
from fastapi.middleware.cors import CORSMiddleware
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
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
_ai_configured = False
MODEL = os.getenv("MODEL", DEFAULT_GEMINI_MODEL)


def _get_provider_and_key() -> tuple[Optional[str], Optional[str]]:
    gemini_key = os.getenv("API_KEY") or os.getenv("GEMINI_API_KEY")
    if gemini_key:
        return "gemini", gemini_key

    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        return "openai", openai_key

    return None, None


def _ensure_ai_ready() -> None:
    global _ai_configured
    if _ai_configured:
        return

    provider, api_key = _get_provider_and_key()
    if not api_key or not provider:
        raise HTTPException(
            status_code=500,
            detail="Missing API key. Set API_KEY/GEMINI_API_KEY (Gemini) or OPENAI_API_KEY (OpenAI).",
        )

    global MODEL
    if provider == "gemini":
        MODEL = os.getenv("MODEL", DEFAULT_GEMINI_MODEL)
        client = AsyncOpenAI(base_url=GEMINI_BASE_URL, api_key=api_key)
    else:
        MODEL = os.getenv("MODEL", DEFAULT_OPENAI_MODEL)
        client = AsyncOpenAI(api_key=api_key)

    set_default_openai_client(client=client)
    set_default_openai_api("chat_completions")
    set_tracing_disabled(True)

    # Agents are instantiated at import time; keep model aligned to selected provider.
    briefing_agent.model = MODEL
    chat_agent.model = MODEL
    _ai_configured = True

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

class ProcessTextRequest(BaseModel):
    text: str
    title: str = "Untitled Document"

class ProposalRequest(BaseModel):
    job_request: Optional[str] = None
    job: Optional[str] = None
    job_description: Optional[str] = None
    description: Optional[str] = None
    prompt: Optional[str] = None
    context: Optional[str] = None
    company: Optional[str] = None
    sender_name: Optional[str] = None

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
    request: Request,
):
    """
    Paste raw text directly for processing.
    Returns a doc_id and the full executive briefing.
    """
    content_type = request.headers.get("content-type", "").lower()
    text = ""
    title = "Untitled Document"

    if "application/json" in content_type:
        payload = ProcessTextRequest(**(await request.json()))
        text = payload.text
        title = payload.title
    else:
        form = await request.form()
        text = str(form.get("text", "")).strip()
        title = str(form.get("title", "Untitled Document"))

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
    _ensure_ai_ready()
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


@app.post("/emails/generate/proposal", tags=["Generation"])
async def generate_proposal(request: Request):
    """
    Generate a professional email proposal from a job request or description.
    """
    content_type = request.headers.get("content-type", "").lower()
    data = {}
    if "application/json" in content_type:
        data = await request.json()
    else:
        form = await request.form()
        data = dict(form)

    req = ProposalRequest(**data)
    source_text = req.job_request or req.job or req.job_description or req.description or req.prompt or req.context

    if not source_text or len(source_text.strip()) < 20:
        raise HTTPException(
            status_code=422,
            detail="Provide a job request/description with at least 20 characters.",
        )

    _ensure_ai_ready()
    company = req.company or "the client"
    sender_name = req.sender_name or "the team"
    proposal_agent = Agent(
        name="Proposal Email Generator",
        model=MODEL,
        instructions=(
            "Write clear, concise, professional proposal emails. "
            "Return plain text only."
        ),
    )
    proposal_prompt = f"""
Create a professional proposal email.

Client: {company}
Sender: {sender_name}
Job Request:
{source_text.strip()}

Requirements:
- Include a short subject line on first line as: Subject: ...
- Keep tone confident and practical
- Include brief scope, timeline, cost-placeholder, and next step call-to-action
- Keep total length under 250 words
"""
    result = await Runner.run(proposal_agent, proposal_prompt)
    return {
        "status": "success",
        "proposal": result.final_output,
    }


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
    _ensure_ai_ready()
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
