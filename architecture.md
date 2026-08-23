# LangGraph Research Agent — Architecture Design (v2)

## 1. Architecture Overview

The system is an evidence-based research agent orchestrated using LangGraph. It accepts a research question, understands intent, decomposes the question into sub-tasks, retrieves evidence from a curated RAG knowledge base and/or external web sources, evaluates evidence quality, decides what can legitimately be concluded, synthesizes a structured research report, and validates that report before returning it to the user.

This version consolidates the original eight logical stages into **four pipeline nodes**, each covering one coherent responsibility, plus **two bounded retry loops** that handle the two ways the pipeline can come back with a weak result: insufficient evidence and unverified claims.

1. Understand & Plan
2. Retrieve Evidence
3. Evaluate & Assess
4. Synthesize & Validate

The system still does not retry blindly or invent information to fill gaps. Retries are targeted, bounded, and only fire on a specific, named failure condition.

---

## 2. High-Level Architecture

```text
                               USER QUESTION
                                     |
                                     v
                      +-----------------------------+
                      |    1. Understand & Plan     |
                      |     Intent + sub-tasks      |
                      |      + source routing       |
                      +-----------------------------+
                                     |
                                     v
                      +-----------------------------+
          +---------->|    2. Retrieve Evidence     |
          |           |    RAG + Web (parallel)     |
          |           +-----------------------------+
          |                          |
          |                          v
          |           +-----------------------------+
          |           |    3. Evaluate & Assess     |
          |           |      Score evidence +       |
          |           |     sufficiency verdict     |
          |           +-----------------------------+
          |      insufficient        | sufficient / partial
          +--------------------------+
            (bounded retry, rewritten queries)
                                     v
                      +-----------------------------+
          +---------->|  4. Synthesize & Validate   |
          |           |  Draft report + fact-check  |
          |           |      against evidence       |
          |           +-----------------------------+
          |      failed              | validation passed
          +--------------------------+
            (bounded retry, feedback-corrected)
                                     v
                               FINAL REPORT
```

---

## 3. Stage 1 — Understand & Plan

### Purpose

Turn one messy natural-language question into a small set of answerable sub-questions, each with a known destination for evidence.

### Input

```text
User Question (raw natural language)
```

Example:

```text
What are the latest developments in AI agents
and how do they compare with the approaches
described in our knowledge base?
```

### Process

1. Parse intent: question type, key entities, research goal, explicit constraints.
2. Determine whether the question needs current/external information, curated/internal information, or both.
3. Decompose the question into independent research tasks. Each task is itself a clear, answerable sub-question — no separate "objective" wording layer.
4. For each task, route it to a source: `rag`, `web`, or `rag + web` (used when the task requires comparing internal and external information).
5. Does **not** retrieve anything and does **not** generate multiple phrasings of each task upfront — that only happens later, on-demand, if retrieval comes back insufficient.

### Output

```json
{
  "question_type": "comparison",
  "requires_current_information": true,
  "requires_internal_information": true,
  "research_tasks": [
    {
      "id": "RQ1",
      "question": "What are the latest developments in AI agents?",
      "sources": ["web"]
    },
    {
      "id": "RQ2",
      "question": "What agent architectures are described in the knowledge base?",
      "sources": ["rag"]
    },
    {
      "id": "RQ3",
      "question": "How do current agent approaches compare to those in the knowledge base?",
      "sources": ["rag", "web"]
    }
  ]
}
```

### Responsibility

> **Decide what needs to be answered, and where to look for each piece.**

---

## 4. Stage 2 — Retrieve Evidence

### Purpose

Execute retrieval for every research task against its assigned source(s), and collect results into one unified evidence structure.

### Input

```json
{
  "research_tasks": [
    { "id": "RQ1", "question": "...", "sources": ["web"] },
    { "id": "RQ2", "question": "...", "sources": ["rag"] },
    { "id": "RQ3", "question": "...", "sources": ["rag", "web"] }
  ]
}
```

### Process

1. Fan out: dispatch one retrieval call per `(task_id, source)` pair in parallel (LangGraph `Send` API), so a task routed to both RAG and web doesn't wait on itself.
2. Each call returns raw results — RAG returns chunks + document metadata, web returns pages + source metadata.
3. Normalize both into one common evidence shape, regardless of source.
4. Fan in: merge all results into a single evidence list and deduplicate — by URL for web, by document/chunk ID for RAG, or by content similarity across sources.
5. Does **not** judge evidence quality — only collects it.

### Output

```json
{
  "evidence": [
    {
      "evidence_id": "E1",
      "research_id": "RQ1",
      "source_type": "web",
      "title": "Example AI Agent Research",
      "url": "https://example.com/article",
      "content": "Relevant extracted content...",
      "metadata": { "published_date": "2026-08-10" }
    },
    {
      "evidence_id": "E2",
      "research_id": "RQ2",
      "source_type": "rag",
      "document_id": "DOC-001",
      "title": "Agent Architecture Research",
      "content": "Relevant document content...",
      "metadata": { "document_type": "research_paper" }
    }
  ]
}
```

### Responsibility

> **Collect evidence for every task without deciding whether it is good enough.**

### Retry Retrieval (bounded loop)

**Trigger:** not decided here — it is decided in Stage 3 (Evaluate & Assess), when a task's sufficiency verdict comes back `insufficient`. A conditional edge then routes control back to this stage, for that task only.

**What changes on a retry:**

1. Only the flagged task(s) are re-run — tasks that already succeeded are untouched.
2. The LLM generates 2–3 alternate phrasings of the failed task's question (this is where query rewriting actually applies — not upfront).
3. The source set may widen (e.g. a RAG-only task also tries web on retry).
4. New results are merged and deduplicated against evidence already collected for that task.

**State tracking (prevents infinite loops):**

```json
{ "retrieval_attempts": { "RQ1": 0, "RQ3": 1 } }
```

**Conditional edge logic:**

```text
if status == "insufficient" AND retrieval_attempts[task_id] < max_retries:
    → back to Retrieve Evidence (rewritten queries, that task only)
else:
    → proceed to Synthesize & Validate with the gap documented
```

Recommended `max_retries`: 2. If two rewrites still fail, the gap is likely real, not a phrasing problem.

---

## 5. Stage 3 — Evaluate & Assess

### Purpose

Determine how good the retrieved evidence is, and how far it can legitimately support a conclusion for each research task — including deciding when it's worth retrieving again.

### Input

```json
{
  "research_tasks": [
    { "id": "RQ1", "question": "..." }
  ],
  "evidence": [
    { "evidence_id": "E1", "research_id": "RQ1", "source_type": "web", "content": "...", "metadata": {} }
  ]
}
```

### Process

**Sub-phase A — Evidence Evaluation (per evidence item):**

Score every evidence item against its own research task:

- **Relevance** — does it directly address the task?
- **Authority** — is the source credible for this claim?
- **Recency** — current enough for the question?
- **Specificity** — concrete detail vs. vague assertion?
- **Corroboration** — supported by other independent evidence?

Also tag which claims each item supports or contradicts, and flag weak, irrelevant, or duplicate items.

**Sub-phase B — Evidence Sufficiency (per research task, rolled up):**

Aggregate all evaluated evidence for a task into one verdict:

- `sufficient` — reliable, corroborated evidence exists → proceed.
- `partial` — some angle answered, real gap remains → proceed, gap gets documented.
- `insufficient` — no reliable evidence to support a conclusion → triggers the retry loop back to Stage 2.

### Output

```json
{
  "evaluated_evidence": [
    {
      "evidence_id": "E1",
      "research_id": "RQ1",
      "evaluation": {
        "relevance": "high",
        "authority": "high",
        "recency": "high",
        "specificity": "high",
        "corroboration": "medium"
      },
      "supports": ["claim_1"],
      "contradicts": [],
      "confidence": "high"
    }
  ],
  "research_assessment": [
    { "research_id": "RQ1", "status": "sufficient", "confidence": "high", "evidence_gap": null },
    { "research_id": "RQ3", "status": "insufficient", "confidence": "low", "evidence_gap": "No direct comparative evidence found" }
  ]
}
```

### Responsibility

> **Decide how good the evidence is, and what it can legitimately support — including when it's worth going back for more.**

### Conditional edge (this is where Retry Retrieval actually fires)

```text
for each research_task:
    if status == "insufficient" AND retrieval_attempts[task_id] < max_retries:
        → route this task_id back to Retrieve Evidence
    else:
        → mark task final (even if insufficient), continue to Synthesize & Validate
```

Only insufficient tasks loop back — sufficient and partial tasks flow straight through. Re-evaluation on retry only scores the newly retrieved items; it doesn't re-score evidence already evaluated.

---

## 6. Stage 4 — Synthesize & Validate

### Purpose

Turn evaluated evidence into a coherent, grounded report, then fact-check that report against the evidence before it ever reaches the user.

### Input

```json
{
  "original_question": "...",
  "research_tasks": [ { "id": "RQ1", "question": "..." } ],
  "evaluated_evidence": [ { "evidence_id": "E1", "research_id": "RQ1", "evaluation": {}, "confidence": "high" } ],
  "research_assessment": [ { "research_id": "RQ1", "status": "sufficient", "evidence_gap": null } ]
}
```

### Process

**Sub-phase A — Research Synthesis:**

1. Answer the original question, organized around the research tasks.
2. Tie every non-trivial claim to a specific `evidence_id`.
3. State conflicting evidence explicitly instead of silently picking a side.
4. Name gaps for any task marked `insufficient` or `partial` — never smooth them over.
5. Treat the evaluated evidence as the source of truth, not the LLM's general knowledge.

**Sub-phase B — Report Validation:**

Fact-check the draft against the evidence it claims to rely on:

- **Claim support** — does every finding trace back to a real `evidence_id`?
- **Citation correctness** — does the cited evidence actually say what the claim says?
- **Unsupported information** — did synthesis introduce anything not present in the evidence?
- **Gap disclosure** — are insufficient/partial tasks visibly flagged?
- **Conclusion consistency** — does the conclusion follow from the findings, or overreach?

### Output

Draft (sub-phase A):

```json
{
  "draft_report": {
    "executive_summary": "...",
    "key_findings": [
      { "finding": "...", "supporting_evidence": ["E1", "E4"], "research_id": "RQ1" }
    ],
    "conflicting_evidence": [
      { "claim": "...", "supports": ["E2"], "contradicts": ["E5"] }
    ],
    "limitations": [
      { "research_id": "RQ3", "gap": "Limited direct comparison evidence" }
    ],
    "conclusion": "..."
  }
}
```

Validation result (sub-phase B):

```json
{
  "validation_status": "failed",
  "validation_issues": [
    {
      "type": "unsupported_claim",
      "claim": "AI agents adopt memory in 80% of production systems",
      "reason": "No evidence item supports this statistic"
    }
  ]
}
```

Final output, once passed:

```json
{
  "validation_status": "passed",
  "validation_issues": [],
  "final_report": {
    "research_question": "...",
    "executive_summary": "...",
    "key_findings": [],
    "detailed_analysis": [],
    "conflicting_evidence": [],
    "limitations": [],
    "conclusion": "...",
    "sources": []
  }
}
```

### Responsibility

> **Produce a report that says only what the evidence supports, and never present an unverified answer as a verified one.**

### Retry Synthesis (bounded loop)

**Trigger:** `validation_status == "failed"` at the end of sub-phase B.

**What changes on a retry:** `validation_issues` are fed back into Research Synthesis as explicit correction instructions (e.g. "remove the unsupported 80% claim, or replace it with what E1–E5 actually support"). This is targeted repair, not a blind re-run.

**State tracking:**

```json
{ "synthesis_attempts": 1 }
```

**Conditional edge logic:**

```text
if validation_status == "failed" AND synthesis_attempts < max_retries:
    → back to Research Synthesis, with validation_issues as feedback
elif validation_status == "failed" AND synthesis_attempts >= max_retries:
    → return final_report with an explicit "unverified" flag
else:
    → return final_report
```

Recommended `max_retries`: 1–2. If the model still can't produce a grounded report after being told exactly what's wrong twice, more retries won't fix it — surface the `unverified` flag honestly instead of looping forever.

---

## 7. Shared Graph State

```json
{
  "user_question": "string",
  "question_type": "string",
  "requires_current_information": "bool",
  "requires_internal_information": "bool",
  "research_tasks": [],
  "evidence": [],
  "evaluated_evidence": [],
  "research_assessment": [],
  "draft_report": {},
  "validation_status": "string",
  "validation_issues": [],
  "final_report": {},
  "retrieval_attempts": {},
  "synthesis_attempts": 0
}
```

Notes for implementation:

- `evidence` and `evaluated_evidence` should use an `operator.add` reducer, since parallel retrieval fan-out writes to them from multiple concurrent branches.
- `retrieval_attempts` is keyed by `research_id` so retries are tracked per task, not globally.
- `synthesis_attempts` is a single counter, since synthesis operates on the whole report, not per-task.

---

## 8. Core Design Principle

| Stage                | Responsibility                                              |
| -------------------- | ------------------------------------------------------------ |
| Understand & Plan     | What needs to be answered, and where should we look?        |
| Retrieve Evidence     | What did we find?                                            |
| Evaluate & Assess     | How good is the evidence, and what can we conclude?          |
| Synthesize & Validate | Is the report grounded, and does it say only what's supported? |

> **Understand → Plan → Retrieve → Evaluate → Assess → Synthesize → Validate**, with two bounded feedback loops: retry retrieval on insufficient evidence, retry synthesis on failed validation.

---

## 9. Final System Responsibility

> **Transform a complex research question into a structured, evidence-backed research report by combining curated knowledge and external information, evaluating evidence quality, explicitly handling uncertainty and evidence gaps, retrying in a targeted and bounded way when evidence or grounding falls short, and never presenting an unverified answer as a reliable one.**
