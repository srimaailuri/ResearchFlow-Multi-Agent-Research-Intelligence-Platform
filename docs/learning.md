# ResearchFlow ù Project Learnings

Notes from building this codebase: concepts, decisions, and enterprise-style patterns worth keeping.

Related docs: [architecture.md](./architecture.md), [Project-definition.md](./Project-definition.md).

---

## LangGraph `Send` (fan-out / map-reduce)

### What we learned

`Send(node_name, arg)` tells LangGraph to invoke **that node again** with a **separate payload** (`arg`), not the full graph state. Returning a **list of `Send`** from a routing function is the **map** step: one independent unit of work per `(research task, source)`.

In this project:

- **`fan_out_retrieval`** builds jobs from `research_tasks` and returns `[Send("retrieve_one", job), ...]`.
- **`retrieve_one`** runs once per job (worker); each run performs retrieval for a single `(task, source)` pair.
- **`finalize_retrieval`** is the **join** after workers finish (sync point + dedupe on merged evidence).

Same logical work as a nested `for task / for source` loop in one node; different **orchestration** so the runtime can run workers concurrently.

### Example (conceptual)

After planning, suppose:

- RQ1 ? `sources: ["web"]`
- RQ2 ? `sources: ["rag"]`
- RQ3 ? `sources: ["rag", "web"]`

That yields **four** jobs and **four** `Send("retrieve_one", ...)` calls. Each worker gets a small `RetrieveJob`: `{ task, source, sequence }`.

### Is this real parallelism or only ùon paperù?

Only inside **`retrieve_one`** (worker), based on `job["source"]`.  
`fan_out_retrieval` only builds jobs; `finalize_retrieval` only dedupes.

---

## Parallelism and concurrency (important)

### Is `Send` ùmockù parallelism?

**No.** `Send` is real **workflow fan-out**: LangGraph schedules **multiple executions** of the same node with different inputs.

You do **not** write `ThreadPoolExecutor` in application code for that map step; the **framework runtime** runs tasks concurrently.

### Division of responsibility

```text
Send                    ?  WHAT to run in parallel (one job per task ù source)
LangGraph Pregel        ?  WHEN steps run, merge state, checkpoints
LangGraph executor      ?  HOW concurrent tasks run (threads or asyncio)
retrieve_one + stubs    ?  YOUR business logic per job
merge_evidence_state    ?  Safe merge + dedupe across branches
```

### What LangGraph uses under the hood (sync vs async)

When you call **`graph.invoke(...)`**, LangGraphùs sync path uses a **background executor** that delegates work to a **thread pool** (`concurrent.futures`), via LangChainùs `get_executor_for_config`.

When you call **`graph.ainvoke(...)`**, the async path uses **asyncio** tasks (and can respect **`max_concurrency`** in config).

So: **you didnùt add ThreadPoolExecutor in `retrieve.py` because LangGraph already uses one (or asyncio) inside its pregel executor.**

### Stubs hide the benefit

In-memory stubs finish in microseconds. Parallelism is still **scheduled**, but **wall-clock speedup** only shows up when retrievers do slow I/O (HTTP, embedding search, DB). That is expected.

**Optional lab:** add a temporary `time.sleep(1)` in stubs and compare total runtime for 4 jobs (roughly ~1s parallel vs ~4s sequential).

---

## Enterprise-style rules (use as we build)

### 1. Workflow parallelism (prefer LangGraph)

Use when work is **independent** across branches:

- Multiple `(research_id, source)` retrievals ? **`Send`** (current design).
- Phase 5: retry **only** failed tasks ? `Send` again for those jobs.

Avoid hand-rolling a thread pool that duplicates what the graph already expresses.

### 2. In-node / external concurrency (you still own this)

LangGraph does **not** remove the need for good I/O design inside a node:

| Concern | Typical approach |
|---------|------------------|
| API rate limits | `max_concurrency` on invoke config, semaphores, backoff |
| HTTP to web search | `httpx` async client or shared session; timeouts |
| Vector DB / RAG | Connection pool, batch size limits |
| CPU-heavy PDF parsing | Process pool, separate worker service, or queue |
| Shared mutable globals in nodes | Avoid; use thread-safe clients |

**Rule:** **Between branches** ? `Send` + graph config. **Inside a node** or **shared external systems** ? async clients, pools, limits.

### 3. State and reducers

- **`evidence`** uses **`merge_evidence_state`** (concat + dedupe), not plain `operator.add`, so parallel branches and finalize do not duplicate rows incorrectly.
- **`evaluated_evidence`** (later) will use `operator.add` when parallel evaluate workers append scores.

### 4. Observability (production habit)

Log per job: `research_id`, `source`, `sequence`, thread/async context.  
Later: LangSmith / OpenTelemetry spans per `Send` branch.

### 5. Idempotency

Each **`retrieve_one`** job should be safe to retry (Phase 5): same job ? same logical fetch, dedupe by URL / `document_id`.

---

## Mental model for this repo

```text
Plan:     WHAT + WHERE (sources on each task)
Send:     many retrieve workers
Workers:  retrieve_web / retrieve_rag (stub ? real)
Reducer:  merge evidence safely
Finalize: join + dedupe barrier
```

---

## Questions we answered in chat (index)

- **`StateGraph(ResearchState)`** ù passes the **state schema** (TypedDict), not a graph instance; the graph comes from `.compile()`.
- **`pip install -e .`** ù editable install; code stays in `src/`, imports resolve via `.venv`.
- **`ResearchPlan` fields** ù filled by Gemini at plan time; copied into `ResearchState`.
- **`with_structured_output`** ù validates response **structure**; semantic quality still depends on prompts and tests.

---

## Next learning topic

**Phase 3 ù Evaluate & assess:** score each evidence item, roll up per-task `sufficient` / `partial` / `insufficient` (stub rules first, then LLM).

---

## LangChain `with_structured_output` (Pydantic)

Revision notes for structured LLM responses (used in **Understand & Plan**).

### Definition (memorize)

| Term | Meaning |
|------|--------|
| **`with_structured_output(Model)`** | Wraps the chat model so the **successful** reply is parsed into Pydantic **`Model`**. |
| **Schema** | Pydantic model fields become JSON schema the provider is steered toward (tool / JSON mode). |
| **Parse + validate** | Response text ? JSON ? **`Model.model_validate`**; bad shape ? **error**, not a loose dict. |
| **`ResearchPlan`** | Plan-stage schema: `question_type`, `requires_*`, `research_tasks`. |
| **`ResearchTask`** | Nested in plan: `id`, `question`, `sources` (`rag` / `web`). |

### In this repo

```text
get_chat_model().with_structured_output(ResearchPlan)
plan: ResearchPlan = llm.invoke([SystemMessage, HumanMessage(user_question)])
return partial state update (fields copied from plan, tasks as model_dump JSON)
```

- **`ResearchPlan`** exists only for **one LLM call**; graph state stores **top-level keys**, not a nested `ResearchPlan` object.
- **`user_question`** stays on state; plan node **updates** `question_type`, `requires_current_information`, `requires_internal_information`, `research_tasks`.

### What it guarantees vs what it does not

| Guaranteed (call succeeds) | Not guaranteed |
|----------------------------|----------------|
| Fields match Pydantic types and required keys | Correct planning (flags, task split, sources) |
| Nested `ResearchTask` / enums valid where enforced | Same wording every run |
| No extra top-level fields on the returned object | Optimal number of sub-questions |

**Two failure modes:**

1. **Schema / parse failure** ù exception; no valid `ResearchPlan` instance.
2. **Schema OK, content wrong** ù valid object, bad judgment (fix with **prompt**, enums, tests).

### Flow (revise in order)

```text
user_question  -->  LLM + schema(ResearchPlan)  -->  ResearchPlan instance
       -->  model_dump (tasks)  -->  merge into ResearchState
```

### Recall checks

**Q: Does `with_structured_output` force only those variables?**  
A: On **success**, yes ù you get a **`ResearchPlan`** (those fields only). Values inside are still the model's choice.

**Q: Is it the same as "reply in JSON" in the prompt?**  
A: **Stronger** ù provider binding + **Pydantic validation**; failures surface as errors, not manual `json.loads` + hope.

**Q: Where does `ResearchPlan` live in graph state?**  
A: It **doesn't** persist as one blob; its fields are **copied** onto **`ResearchState`**.

**Q: Can `question_type` be wrong but valid?**  
A: **Yes** if it stays a **str**; tightening with a **`QuestionType` enum** fails invalid labels at validate time.

### Don't confuse

| Mistake | Correct |
|---------|---------|
| Structured output = always correct plan | It enforces **shape**, not **quality** |
| Store `ResearchPlan` on state | Store **flattened** plan fields on **`ResearchState`** |
| Structured output = 0% parse errors | **Low** on simple schemas, **not zero**; handle retries / errors in production |
| Plan node retrieves evidence | Plan sets **`research_tasks`**; retrieval is **later** |

### Production reminders

- Add **`Field(description=...)`** on schema fields to steer the model via schema docs.
- **Retry once** on validation errors for transient provider glitches.
- **Tests:** mock LLM returning fixed `ResearchPlan` JSON for stable CI (no API key).
- Log **`question_type`** and task count; assert invariants in code (e.g. every task has non-empty `sources` when policy requires it).

### One-line summary

**`with_structured_output` = typed LLM contract; Pydantic validates shape; prompts and tests validate meaning.**

---

## Live retrieval (Phase 6)

### What changed

Retrieval is always **live**: `retrieval/rag.py` (Chroma + PDFs) and `retrieval/web.py` (Tavily or DuckDuckGo). The graph calls them through `retrieve_one`; there is no stub retrieval path.

### RAG path

- PDFs under `RAG_DOCUMENTS_DIR` (default `Data/test_documents/`).
- First run: load with `PyPDFLoader`, chunk, embed with **`models/text-embedding-004`**, persist **Chroma** under `.chroma/`.
- Later runs: reopen persisted store (no re-index unless the folder is empty).
- Each hit becomes an `Evidence` row with `document_id`, `content`, and `source_type=rag`.

### Web path

- If **`TAVILY_API_KEY`** is set ? Tavily via LangChain community wrapper.
- Else ? **DuckDuckGo** (`duckduckgo-search`).
- Up to **`retrieval_top_k`** results per job; mapped to `Evidence` with `url` and truncated `content`.

### Retry query widening

`retrieval/query.py` ? **`retrieval_query_for_task`**: when `retrieval_attempts[task_id] > 0`, the search string adds guidance to use synonyms and related terms (architecture ùwiden on retryù). The same `query` is passed on every `RetrieveJob` and into live retrievers.

### Enterprise pattern: swap backends without touching the graph

| Layer | Responsibility |
|-------|------------------|
| `retrieve_one` | One `(task, source)` job; calls `retrieve_rag` / `retrieve_web` |
| `retrieval/__init__.py` | Re-exports `retrieve_rag` / `retrieve_web` |
| `rag.py` / `web.py` | I/O, normalization to `Evidence` |

**Evaluate** uses Gemini + structured output per evidence item (no stub scores).

### Recall checks

**Q: Does the plan node call RAG or web?**  
A: **No.** Plan only sets `research_tasks[].sources`. Retrieval runs in **`retrieve_one`**.

**Q: Where does parallelism matter for live mode?**  
A: Same as stubs: multiple **`Send("retrieve_one", job)`** branches; wall-clock wins when HTTP and embedding search overlap.

**Q: What if Chroma index is stale after adding PDFs?**  
A: Delete **`.chroma/`** (or point `CHROMA_PERSIST_DIR` elsewhere) to force re-index.
