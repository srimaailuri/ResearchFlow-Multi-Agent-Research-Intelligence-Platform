# ResearchFlow — Project Overview

**ResearchFlow** is a stateful, multi-stage research and RAG pipeline built with **LangGraph**. It takes a natural-language research question, breaks it into sub-tasks, retrieves evidence from an internal PDF corpus and/or the web, evaluates the retrieved evidence with an LLM, synthesizes a structured report, validates the result, and selectively retries when evidence or validation is insufficient.

The design focuses on **source-grounded generation, explicit state management, bounded execution, parallel retrieval, validation, and operational controls** such as cost limits, concurrency controls, logging, and checkpoints.

ResearchFlow is primarily a **controlled AI workflow**, rather than a fully autonomous tool-using agent. Its major stages and routing logic are explicitly defined in the LangGraph workflow.

Repository: [ResearchFlow-Multi-Agent-Research-Intelligence-Platform](https://github.com/srimaailuri/ResearchFlow-Multi-Agent-Research-Intelligence-Platform)

Related design docs: [architecture.md](architecture.md), [learning.md](learning.md).

---

## 1. Problem Statement

Traditional LLM-based research flows can hallucinate, overlook relevant internal documents, or produce answers without sufficient supporting evidence.

ResearchFlow treats research as a structured pipeline:

1. **Plan** — classify the question and split it into independent sub-questions with explicit source requirements (`rag`, `web`, or both).
2. **Retrieve** — retrieve internal document chunks and web evidence in parallel using LangGraph `Send`.
3. **Evaluate** — score evidence for relevance, authority, recency, specificity, and corroboration, and determine whether each task has sufficient evidence.
4. **Synthesize** — generate a structured research report using the evaluated evidence.
5. **Validate** — check citations and disclosure of gaps or limitations.
6. **Retry** — selectively retry retrieval or synthesis when configured validation or evidence conditions are not satisfied.

The shared `ResearchState` carries research tasks, evidence, assessments, drafts, validation results, and the final report through the workflow.

---

## 2. Design Principles

ResearchFlow is designed around the following principles:

### Source-grounded generation

Final responses should be based on retrieved evidence rather than relying only on the model's internal knowledge.

### Explicit state management

Research progress and intermediate results are maintained in a shared `ResearchState`, allowing different graph nodes to operate on the same execution state.

### Parallel execution

Independent retrieval jobs are executed concurrently using LangGraph fan-out/fan-in patterns.

### Bounded execution

Retrieval and synthesis retries have configurable limits to prevent uncontrolled execution.

### Evidence management

Evidence is deduplicated, ranked, and capped before expensive evaluation and synthesis steps.

### Validation before completion

The generated report is validated before the workflow terminates.

### Operational control

Concurrency limits, retry bounds, evidence caps, structured logging, and checkpoints provide basic production-oriented execution controls.

---

## 3. Workflow Architecture

ResearchFlow follows a controlled workflow:

```text
                    ┌──────────────┐
                    │    START     │
                    └──────┬───────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │ Plan / Decompose│
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │ Parallel        │
                  │ Retrieval       │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │ Finalize /      │
                  │ Deduplicate     │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │ Evaluate        │
                  │ Evidence        │
                  └────────┬────────┘
                           │
                    ┌──────┴──────┐
                    │             │
              insufficient     sufficient
                    │             │
                    ▼             ▼
               Retry          Synthesize
                    │             │
                    └──────┐      ▼
                           │   Validate
                           │      │
                           │ ┌────┴────┐
                           │ │         │
                           │ failed   passed
                           │ │         │
                           │ ▼         ▼
                           │ Retry     END
                           │
                           └──────────────
```

The compiled graph is defined in `src/researchflow/graph/builder.py`.

Nodes are Python functions, while graph edges control execution and conditional routing.

---

## 4. LangGraph Control Flow

The graph uses:

- `StateGraph` for workflow orchestration
- Shared state through `ResearchState`
- Conditional edges for routing
- `Send` for parallel retrieval
- Reducers for merging evidence from parallel workers
- Checkpoints for optional run persistence and replay

The major workflow stages are:

```text
START
  ↓
understand_and_plan
  ↓
fan_out_retrieval
  ↓
retrieve_one
  ↓
finalize_retrieval
  ↓
evaluate_and_assess
  ↓
route_after_evaluate
  ├── retry retrieval
  └── synthesize_report
          ↓
      validate_report
          ├── retry synthesis
          └── END
```

### Parallel Retrieval

For every `(research_task, source)` pair, the planner produces a retrieval job.

LangGraph `Send` creates independent retrieval workers. Each worker contributes evidence to the shared `evidence` state through a reducer.

`finalize_retrieval` then:

- deduplicates evidence
- applies evidence limits
- ranks available evidence
- prepares the evidence set for evaluation

### Conditional Routing

ResearchFlow uses conditional routing to determine whether additional work is required.

---

## 5. Retry and Failure Recovery

ResearchFlow uses bounded retries at different levels.

| Failure / Condition | Recovery |
|---|---|
| LLM transient API failure | Tenacity retry |
| Rate limiting | Retry with configured backoff |
| Insufficient retrieval evidence | Retry retrieval for affected tasks |
| Failed report validation | Retry synthesis with validation feedback |
| Excessive evidence | Apply evidence caps |
| Excessive concurrent requests | Semaphore-based concurrency control |

### Retrieval Retry

If a task is marked insufficient and retrieval attempts remain, the workflow performs another retrieval attempt using broader queries.

### Synthesis Retry

If report validation fails and synthesis attempts remain, the report is regenerated using the available validation feedback.

All workflow retries are bounded using configurable limits.

---

## 6. Implemented Components

### 6.1 Core Pipeline Nodes

Located under `src/researchflow/nodes/`.

| Node | Responsibility |
|---|---|
| `plan.py` | Gemini structured output → research tasks, question type, RAG/web flags |
| `retrieve.py` | Retrieval job creation, `Send` fan-out, RAG/web workers, finalization and evidence caps |
| `evaluate.py` | LLM-based evidence evaluation; one batch per task with parallel task processing |
| `routing.py` | Retry scheduling and conditional routing |
| `synthesize.py` | Structured report generation |
| `validate.py` | Rule-based validation of citations, gaps, and limitations |

---

## 7. Retrieval

ResearchFlow supports two primary evidence sources.

### 7.1 Internal RAG

The internal retrieval pipeline uses:

- Chroma
- Google `gemini-embedding-001` embeddings
- PDF documents
- Persistent vector index
- Batched indexing
- Rate-limit backoff
- Thread-safe vector-store initialization

The retrieval pipeline supports configurable retrieval breadth through `top_k` and candidate settings.

### 7.2 Web Retrieval

Web evidence can be retrieved using:

- Tavily when configured
- DuckDuckGo as a fallback

### 7.3 Evidence Deduplication

RAG and web results use separate deduplication keys to prevent duplicate evidence from propagating through later stages.

### 7.4 Evidence Caps

ResearchFlow supports:

- `max_evidence_per_task`
- `max_total_evidence`

Evidence is ranked before applying caps so that expensive LLM evaluation and synthesis operate on a bounded evidence set.

### 7.5 Preferred RAG Documents

Requests and benchmark cases can optionally provide preferred documents.

ResearchFlow can use these documents to:

- filter retrieval
- boost their scores
- provide query hints

This supports controlled benchmark evaluation while retaining the general retrieval workflow.

---

## 8. LLM Integration and Resilience

ResearchFlow uses Google Gemini through LangChain integrations.

LLMs are used for:

- research planning
- evidence evaluation
- report synthesis

Structured outputs are used where appropriate to produce predictable task and report structures.

Transient model/API failures are handled with `tenacity` retries.

Model configuration is controlled through environment variables, allowing planning/synthesis and evaluation models to be configured independently.

---

## 9. State and Data Models

`ResearchState` is the central execution state.

It contains information such as:

- research tasks
- retrieved evidence
- evidence assessments
- generated drafts
- validation status
- retry state
- final report

The evidence collection uses a custom reducer so results from parallel retrieval workers can be safely merged.

Pydantic models define structured representations for:

- research tasks
- evidence
- evaluation scores
- assessments
- drafts
- final reports

This provides explicit data contracts between workflow stages.

---

## 10. API and Execution

ResearchFlow exposes both command-line and HTTP interfaces.

| Entry Point | Purpose |
|---|---|
| CLI | Execute a research question directly |
| `/research/run` | Synchronous research execution |
| `/research/jobs` | Asynchronous research execution |
| `/research/runs/{run_id}` | Inspect checkpointed run state |
| `/health` | Service health |
| `/ready` | Service readiness |

The FastAPI service includes:

- request ID middleware
- maximum concurrent run control
- optional RAG warm-up
- configurable retrieval/synthesis retry limits
- preferred document configuration

---

## 11. Observability and Debugging

ResearchFlow provides lightweight operational observability.

### Structured Logging

JSON logs include identifiers such as:

- `request_id`
- `run_id`

Node lifecycle events are recorded for important workflow stages.

### Checkpointing

LangGraph checkpointing can operate in:

- memory mode
- SQLite mode
- no-checkpoint mode

The LangGraph `thread_id` is mapped to the research `run_id`, allowing a run to be inspected or replayed when checkpointing is enabled.

---

## 12. Performance and Resource Controls

ResearchFlow includes several controls to prevent unnecessary resource consumption.

### Evidence Controls

- Maximum evidence per task
- Maximum total evidence
- Retrieval top-K configuration

### Execution Controls

- Maximum concurrent research jobs
- Maximum retrieval attempts
- Maximum synthesis attempts

### API/LLM Controls

- Batched indexing
- Rate-limit retry/backoff
- Optional RAG warm-up
- Configurable model selection

These controls help bound execution time, evidence volume, API usage, and LLM processing.

---

## 13. Benchmark and Evaluation

ResearchFlow contains a deterministic benchmark suite for workflow-level regression testing.

The benchmark contains approximately 20 research questions and evaluates results against expected information.

Current scoring includes:

- Expected topic coverage
- Expected answer similarity
- Expected PDF/document retrieval
- Per-question scores
- Aggregate benchmark scores

The benchmark does not attempt to become a complete external evaluation framework.

More detailed evaluation of retrieval quality, agent behavior, answer quality, and performance is handled by the separate **AgentEval** project.

This keeps the responsibilities separated:

```text
ResearchFlow
    → executes the research workflow
    → provides local deterministic benchmark scoring
    → exports execution traces

AgentEval
    → evaluates ResearchFlow and AgentOS
    → provides broader retrieval/agent/answer evaluation
    → performs regression analysis
```

---

## 14. Testing

ResearchFlow contains approximately 29 pytest tests covering areas such as:

- graph execution paths
- conditional routing
- retrieval deduplication
- evidence caps and boosting
- API behavior
- LLM retry behavior
- observability
- benchmark scoring

External APIs are not required for unit tests.

The test suite uses mocks where appropriate so tests remain deterministic and do not depend on live model or web-search services.

---

## 15. Engineering Concepts

ResearchFlow brings together the following engineering concepts:

### AI / GenAI

- Retrieval-Augmented Generation
- Embeddings
- Vector search
- Structured LLM outputs
- Evidence evaluation
- Source-grounded generation
- Prompt-based task decomposition

### LangGraph

- StateGraph
- Shared state
- Conditional routing
- `Send`
- Fan-out/fan-in execution
- Reducers
- Checkpointing
- Retry loops

### Backend Engineering

- FastAPI
- REST APIs
- Synchronous and asynchronous execution
- Request identification
- Concurrency control
- Configuration management

### Reliability

- Retry strategies
- Rate-limit handling
- Bounded execution
- Validation
- Failure recovery
- Resource controls

### Evaluation and Observability

- Deterministic benchmark scoring
- Structured logging
- Run IDs
- Request IDs
- Checkpoint inspection
- Regression testing

---

## 16. Technology Stack

| Category | Technology | Role |
|---|---|---|
| Language | Python ≥ 3.11 | Application implementation |
| Orchestration | LangGraph | Stateful workflow execution |
| LLM Integration | LangChain Core + `langchain-google-genai` | Gemini integration and structured outputs |
| Embeddings | Gemini Embeddings | RAG vector representation |
| Vector Store | Chroma | Persistent RAG index |
| Document Loading | PyPDF | PDF processing |
| Web Search | Tavily / DuckDuckGo | External evidence retrieval |
| API | FastAPI + Uvicorn | REST API |
| Configuration | Pydantic Settings | Environment-based configuration |
| Retry | Tenacity | Transient failure handling |
| Checkpointing | LangGraph SQLite checkpoint | Optional run persistence |
| Testing | pytest | Automated tests |
| Linting | Ruff | Development tooling |
| Packaging | setuptools | Python package management |

---

## 17. Configuration

Important environment variables include:

| Variable | Purpose |
|---|---|
| `GOOGLE_API_KEY` | Gemini and embedding access |
| `MODEL_NAME` | Planning/synthesis model |
| `EVALUATE_MODEL_NAME` | Evidence evaluation model |
| `RETRIEVAL_TOP_K` | Retrieval result count |
| `RETRIEVAL_RAG_CANDIDATE_K` | RAG candidate breadth |
| `MAX_EVIDENCE_PER_TASK` | Per-task evidence limit |
| `MAX_TOTAL_EVIDENCE` | Global evidence limit |
| `RAG_PREFERRED_DOCUMENT_BOOST` | Preferred document score boost |
| `CHECKPOINTER_MODE` | Checkpoint configuration |
| `CHECKPOINT_SQLITE_PATH` | SQLite checkpoint location |
| `LOG_LEVEL` | Logging level |
| `LOG_JSON` | Structured log configuration |
| `MAX_CONCURRENT_RESEARCH_JOBS` | Concurrent run limit |
| `MAX_RETRIEVAL_ATTEMPTS` | Retrieval retry limit |
| `MAX_SYNTHESIS_ATTEMPTS` | Synthesis retry limit |
| `BENCHMARK_TOPIC_PASS_RECALL` | Topic benchmark threshold |
| `BENCHMARK_ANSWER_PASS_SIMILARITY` | Answer benchmark threshold |

---

## 18. Data Layout

```text
Data/
├── test_documents/
│   └── PDFs used for RAG
│
├── candidate_documents/
│   └── Extended reference corpus
│
└── evaluate_json/
    └── research_questions_memory_ragchecker.json
```

---

## 19. Out of Scope

The current version intentionally does not include:

- External distributed job queues
- Redis/database-backed job management
- LangSmith or third-party tracing
- A full external RAG evaluation framework
- Distributed worker infrastructure
- Multi-agent collaboration
- Long-term agent memory
- Human-in-the-loop execution
- MCP tool integration

These capabilities are outside the current ResearchFlow scope and can be explored in separate systems where they provide a clearer architectural purpose.

---

## 20. Relationship With the Portfolio

ResearchFlow is the foundation for the broader project set.

```text
ResearchFlow
     │
     │ Stateful RAG research workflow
     │
     ▼
AgentOS
     │
     │ General-purpose tool-using agent
     │
     ▼
AgentEval
     │
     │ Evaluation of AI systems
     ▼
Quality + Reliability + Performance
```

The projects intentionally have different responsibilities.

### ResearchFlow

Builds a structured research workflow using LangGraph, RAG, web retrieval, evidence evaluation, synthesis, and validation.

### AgentOS

Builds a more general-purpose agent capable of selecting and executing tools, maintaining state, handling multi-step tasks, and integrating with external systems.

### AgentEval

Evaluates the behavior and output quality of ResearchFlow, AgentOS, and potentially other AI systems using standardized execution traces.

---

## 21. Suggested Code Reading Order

1. `architecture.md` — workflow intent and stage design
2. `src/researchflow/graph/builder.py` — graph wiring
3. `src/researchflow/state/research_state.py` — state contract
4. `src/researchflow/state/models.py` — structured data models
5. `src/researchflow/nodes/plan.py`
6. `src/researchflow/nodes/retrieve.py`
7. `src/researchflow/nodes/evaluate.py`
8. `src/researchflow/nodes/routing.py`
9. `src/researchflow/nodes/synthesize.py`
10. `src/researchflow/nodes/validate.py`
11. `src/researchflow/graph/invoke.py`
12. `src/researchflow/api/app.py`
13. `src/researchflow/eval/scoring.py`

---

## 22. Quick Run

```powershell
python -m researchflow "What is retrieval-augmented generation?"

python -m researchflow.eval_benchmark --ids Q001 --limit 1

researchflow-api

pytest
```