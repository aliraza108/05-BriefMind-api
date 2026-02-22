"""
BriefMind Tools — registered as agent tools via @function_tool decorator
Each tool performs one step in the knowledge distillation pipeline.
"""

import json
import re
from agents import function_tool


# ──────────────────────────────────────────────
# Tool 1: Chunk & Extract
# ──────────────────────────────────────────────
@function_tool
def chunk_and_extract_tool(document_text: str, chunk_size: int = 1000) -> str:
    """
    Split a document into semantic chunks and extract key points from each chunk.

    Args:
        document_text: The full document text to process
        chunk_size: Approximate words per chunk (default 1000)

    Returns:
        JSON string with chunks and extracted key points per chunk
    """
    words = document_text.split()
    total_words = len(words)
    chunks = []

    step = chunk_size
    for i in range(0, total_words, step):
        chunk_words = words[i:i + step]
        chunk_text = " ".join(chunk_words)
        chunks.append({
            "chunk_id": len(chunks) + 1,
            "word_range": f"{i}–{min(i + step, total_words)}",
            "text": chunk_text,
            "word_count": len(chunk_words),
        })

    result = {
        "total_chunks": len(chunks),
        "total_words": total_words,
        "chunks": chunks,
        "instruction": (
            "For EACH chunk, extract: main_topic, key_points (list of 3-5 bullets), "
            "named_entities (people/orgs/places/dates), sentiment, and importance_score (1-10). "
            "Return as structured JSON with chunk_extractions list."
        )
    }
    return json.dumps(result)


# ──────────────────────────────────────────────
# Tool 2: Merge Insights
# ──────────────────────────────────────────────
@function_tool
def merge_insights_tool(chunk_extractions_json: str) -> str:
    """
    Hierarchically merge insights from all chunks, deduplicate, and rank by impact.

    Args:
        chunk_extractions_json: JSON string of extracted insights from all chunks

    Returns:
        JSON with merged insights ranked by impact, grouped by theme
    """
    try:
        data = json.loads(chunk_extractions_json)
    except Exception:
        data = {"raw": chunk_extractions_json}

    instruction = {
        "task": "merge_and_rank_insights",
        "input_data": data,
        "output_format": {
            "key_insights": [
                {
                    "rank": 1,
                    "insight": "string — clear actionable finding",
                    "impact_score": "1-10",
                    "evidence": "brief supporting evidence from document",
                    "theme": "category/theme this belongs to"
                }
            ],
            "topic_breakdown": {
                "theme_name": {
                    "summary": "2-3 sentence theme summary",
                    "key_points": ["list of points"],
                    "chunk_references": ["chunk_ids that contribute to this theme"]
                }
            },
            "named_entities": {
                "people": [],
                "organizations": [],
                "locations": [],
                "dates": [],
                "metrics": []
            }
        },
        "rules": [
            "Deduplicate similar insights, keeping the most specific version",
            "Rank by business/strategic impact (10 = critical, 1 = minor)",
            "Group into 3-7 major themes maximum",
            "Include quantitative data points where available"
        ]
    }
    return json.dumps(instruction)


# ──────────────────────────────────────────────
# Tool 3: Extract Decisions
# ──────────────────────────────────────────────
@function_tool
def extract_decisions_tool(merged_insights_json: str) -> str:
    """
    Extract decision intelligence: actions, risks, opportunities, and strategic implications.

    Args:
        merged_insights_json: JSON string of merged and ranked insights

    Returns:
        JSON with decision_intelligence structure
    """
    try:
        data = json.loads(merged_insights_json)
    except Exception:
        data = {"raw": merged_insights_json}

    instruction = {
        "task": "extract_decision_intelligence",
        "input_data": data,
        "output_format": {
            "decision_intelligence": {
                "suggested_actions": [
                    {
                        "action": "specific actionable step",
                        "priority": "HIGH/MEDIUM/LOW",
                        "timeline": "immediate/short-term/long-term",
                        "owner": "who should own this (role/department)",
                        "rationale": "why this action matters"
                    }
                ],
                "risks": [
                    {
                        "risk": "description of risk",
                        "severity": "CRITICAL/HIGH/MEDIUM/LOW",
                        "likelihood": "HIGH/MEDIUM/LOW",
                        "mitigation": "suggested mitigation strategy"
                    }
                ],
                "opportunities": [
                    {
                        "opportunity": "description",
                        "potential_impact": "quantified or described impact",
                        "effort_required": "HIGH/MEDIUM/LOW",
                        "time_sensitivity": "description"
                    }
                ],
                "strategic_implications": [
                    "list of broader strategic considerations"
                ],
                "warnings": [
                    "critical warnings or red flags from the document"
                ],
                "key_decisions_required": [
                    {
                        "decision": "what needs to be decided",
                        "deadline": "urgency/deadline if known",
                        "stakeholders": ["who needs to be involved"]
                    }
                ]
            }
        }
    }
    return json.dumps(instruction)


# ──────────────────────────────────────────────
# Tool 4: Build Concept Map
# ──────────────────────────────────────────────
@function_tool
def build_concept_map_tool(document_summary_json: str) -> str:
    """
    Build a concept map (knowledge graph) showing relationships between main ideas.

    Args:
        document_summary_json: JSON string with insights and topic breakdown

    Returns:
        JSON with nodes and edges representing the concept map
    """
    try:
        data = json.loads(document_summary_json)
    except Exception:
        data = {"raw": document_summary_json}

    instruction = {
        "task": "build_concept_map",
        "input_data": data,
        "output_format": {
            "concept_map": {
                "nodes": [
                    {
                        "id": "unique_id",
                        "label": "Concept Name",
                        "type": "theme|entity|action|risk|opportunity",
                        "weight": "1-10 (importance)",
                        "description": "brief description"
                    }
                ],
                "edges": [
                    {
                        "source": "node_id",
                        "target": "node_id",
                        "relationship": "causes|supports|contradicts|requires|leads_to|part_of",
                        "strength": "strong|moderate|weak",
                        "description": "brief edge description"
                    }
                ],
                "central_concept": "the single most important concept in the document",
                "key_clusters": [
                    {
                        "cluster_name": "name",
                        "node_ids": ["ids of nodes in this cluster"],
                        "summary": "what this cluster represents"
                    }
                ]
            }
        },
        "rules": [
            "Create 8-20 nodes maximum for clarity",
            "Focus on the most impactful relationships",
            "Use consistent node IDs (snake_case)",
            "Identify 2-4 major clusters"
        ]
    }
    return json.dumps(instruction)


# ──────────────────────────────────────────────
# Tool 5: Generate Executive Summary
# ──────────────────────────────────────────────
@function_tool
def generate_executive_summary_tool(
    full_briefing_json: str,
    target_read_time_minutes: int = 2
) -> str:
    """
    Generate the final executive summary (1-2 minute read) from all processed intelligence.

    Args:
        full_briefing_json: JSON string containing all processed briefing components
        target_read_time_minutes: Target reading time (default 2 minutes ≈ 500 words)

    Returns:
        JSON with the complete executive briefing including summary
    """
    try:
        data = json.loads(full_briefing_json)
    except Exception:
        data = {"raw": full_briefing_json}

    target_words = target_read_time_minutes * 250  # ~250 wpm executive reading speed

    instruction = {
        "task": "generate_executive_summary",
        "input_data": data,
        "target_words": target_words,
        "output_format": {
            "executive_summary": {
                "headline": "One powerful sentence capturing the document's core message",
                "context": "1-2 sentences on what this document is and why it matters",
                "core_message": "The single most important takeaway (2-3 sentences)",
                "top_3_findings": [
                    "Finding 1 — most impactful",
                    "Finding 2",
                    "Finding 3"
                ],
                "bottom_line": "What decision-maker MUST know and do right now (2-3 sentences)",
                "read_time_estimate": f"{target_read_time_minutes} minute(s)"
            }
        },
        "quality_standards": [
            "Write for a C-suite executive with 2 minutes to read",
            "No jargon — clarity over cleverness",
            "Lead with implications, not process",
            "Quantify wherever possible",
            "Every sentence must earn its place"
        ]
    }
    return json.dumps(instruction)


# ──────────────────────────────────────────────
# Tool 6: Answer Question (Chat Mode)
# ──────────────────────────────────────────────
@function_tool
def answer_question_tool(question: str, context_json: str) -> str:
    """
    Answer a user question using the document's stored knowledge base.

    Args:
        question: The user's question about the document
        context_json: JSON string with the document briefing as context

    Returns:
        JSON with the answer and source citations
    """
    try:
        context = json.loads(context_json)
    except Exception:
        context = {"raw": context_json}

    instruction = {
        "task": "answer_question",
        "question": question,
        "available_context": context,
        "output_format": {
            "answer": "Direct answer to the question (2-5 sentences)",
            "confidence": "HIGH/MEDIUM/LOW",
            "sources": [
                {
                    "section": "which part of the briefing supports this answer",
                    "evidence": "specific supporting text or data point"
                }
            ],
            "follow_up_questions": [
                "Suggested related question 1",
                "Suggested related question 2"
            ],
            "caveat": "Any important nuance or limitation to the answer (if applicable)"
        }
    }
    return json.dumps(instruction)