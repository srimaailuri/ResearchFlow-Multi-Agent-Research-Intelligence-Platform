# ResearchFlow



Multi-agent research pipeline built with **LangGraph**: plan sub-questions, retrieve evidence in parallel (`Send`), evaluate, synthesize a draft report, validate citations, and retry with bounded loops.



Design docs: [docs/architecture.md](docs/architecture.md), [docs/Project-definition.md](docs/Project-definition.md). Learning notes: [docs/learning.md](docs/learning.md).



## Requirements



- Python 3.11+

- [Google AI Studio](https://aistudio.google.com/apikey) API key (Gemini chat + embeddings for RAG)



## Setup



```powershell

python -m venv .venv

.venv\Scripts\Activate.ps1

pip install -e ".[dev]"

copy .env.example .env

# Edit .env and set GOOGLE_API_KEY

```



Place PDFs for RAG under `Data/test_documents/` (default). The first RAG run indexes PDFs into `.chroma/` (gitignored).



## Configuration



| Variable | Default | Purpose |

|----------|---------|---------|

| `GOOGLE_API_KEY` | (required) | Gemini + embeddings |

| `RAG_DOCUMENTS_DIR` | `Data/test_documents` | PDF corpus |

| `CHROMA_PERSIST_DIR` | `.chroma` | Vector store on disk |

| `TAVILY_API_KEY` | optional | Tavily web search; if unset, DuckDuckGo is used |

| `MODEL_NAME` | `gemini-3.8-flash` | Chat model (plan, synthesize) |

| `EVALUATE_MODEL_NAME` | (same as `MODEL_NAME`) | Optional separate model for evaluate batches |

| `LLM_MAX_RETRIES` | `4` | Retries on 429/503/quota errors |

| `LLM_RETRY_MIN_WAIT_SECONDS` | `2` | Initial backoff for LLM retries |

| `EMBEDDING_MODEL` | `gemini-embedding-001` | RAG embeddings |

| `MAX_RETRIEVAL_ATTEMPTS` | `2` | Per-task retrieval retries |

| `MAX_SYNTHESIS_ATTEMPTS` | `2` | Synthesis retries after validation failure |

| `MAX_EVALUATE_WORKERS` | `4` | Parallel LLM batch calls (one batch per research task) |

| `MAX_CONCURRENT_RESEARCH_JOBS` | `2` | Max pipeline runs at once (API + sync `/research/run`) |
| `RETRIEVAL_TOP_K` | `3` | Web/RAG hits kept per (task, source) before dedupe/caps |
| `RETRIEVAL_RAG_CANDIDATE_K` | `8` | RAG chunks considered before ranking/cap |
| `RAG_PREFERRED_DOCUMENT_BOOST` | `0.2` | Score boost when chunk matches `preferred_rag_documents` |
| `MAX_EVIDENCE_PER_TASK` | `5` | Evidence cap per research task after retrieval |
| `MAX_TOTAL_EVIDENCE` | `15` | Global evidence cap before evaluate/synthesize |
| `CHECKPOINTER_MODE` | `memory` | LangGraph checkpoints: `memory`, `sqlite`, or `none` |
| `CHECKPOINT_SQLITE_PATH` | `.checkpoints/researchflow.db` | SQLite checkpoint file when mode is `sqlite` |
| `LOG_LEVEL` | `INFO` | Root log level (JSON or text) |
| `LOG_JSON` | `true` | Emit structured JSON logs with `request_id` / `run_id` |
| `BENCHMARK_TOPIC_PASS_RECALL` | `0.5` | Min topic recall for benchmark `auto_pass` |
| `BENCHMARK_ANSWER_PASS_SIMILARITY` | `0.35` | Min answer similarity for benchmark `auto_pass` |



## Run



```powershell

python -m researchflow "What is retrieval-augmented generation?"

```



Output is JSON with top-level `run_id` and `state` (`final_report`, `validation_status`, counters, etc.).

### Benchmark JSON (`Data/evaluate_json/research_questions_memory_ragchecker.json`)

```powershell
python -m researchflow.eval_benchmark --limit 1
python -m researchflow.eval_benchmark --ids Q001,Q002 --full-state
python -m researchflow.eval_benchmark
python -m researchflow.eval_benchmark --no-score   # skip auto-scoring
```

Reports are written under `Data/evaluate_json/runs/`. Each run includes **auto-scoring** (unless `--no-score`):

- **Topic recall** — fraction of `expected_topics` found in generated report text (summary, findings, conclusion).
- **Answer similarity** — token/sequence overlap vs `expected_answer` (no extra LLM call).
- **Source document recall** — expected PDF stems vs RAG `document_id`s retrieved.
- **Overall score** — weighted mix of topics, answer, sources, and validation pass.
- **`auto_pass`** — validation passed and topic/answer thresholds met (tune via `BENCHMARK_*` env vars).

The report JSON includes per-run `scores` and top-level `aggregate_scores` (means and pass rates).

## HTTP API (FastAPI)

Requires `GOOGLE_API_KEY` in `.env` (same as CLI).

```powershell
pip install -e .
researchflow-api
# or: uvicorn researchflow.api.app:app --host 127.0.0.1 --port 8000
```

- **GET** `/health` — liveness (respects / returns `X-Request-ID`)  
- **GET** `/ready` — graph loaded + concurrency settings  
- **POST** `/research/jobs` — **recommended** async run (returns `202` + `job_id`; job `run_id` matches `job_id`)  
- **GET** `/research/jobs/{job_id}` — poll until `status` is `completed` or `failed`  
- **POST** `/research/run` — synchronous run (response includes `run_id`; shares concurrency limit)  
- **GET** `/research/runs/{run_id}` — checkpoint snapshot for debugging (requires checkpointer enabled)

Optional JSON body field `preferred_rag_documents` (PDF names/stems) boosts RAG for that run, same as benchmark `source_documents`.

Async example:

```powershell
$r = Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/research/jobs" `
  -ContentType "application/json" -Body '{"question":"What is RAG?"}'
Invoke-RestMethod "http://127.0.0.1:8000/research/jobs/$($r.job_id)"
```

Limit concurrent pipeline runs with `MAX_CONCURRENT_RESEARCH_JOBS` (default `2`). RAG index is warmed on API startup when possible.

Interactive docs: `http://127.0.0.1:8000/docs`

## Tests



```powershell

pytest

```



Unit tests mock retrieval and LLM calls (no network or API key required).



## Graph (high level)



```text

plan → fan_out (Send) → retrieve_one* → finalize → evaluate

                          ↑__________________________|  (insufficient + attempts left)

evaluate → synthesize → validate → END

              ↑___________|  (validation failed + attempts left)

```



## Project layout



```text

src/researchflow/

  graph/builder.py      # StateGraph wiring

  nodes/                # plan, retrieve, evaluate, synthesize, validate, routing

  retrieval/            # RAG (Chroma), web (Tavily/DDG), dedupe

  state/                # ResearchState + Pydantic models

  config/settings.py    # env-based settings
  eval/scoring.py       # benchmark auto-scoring
  observability/        # structured logs, request_id / run_id context
  graph/checkpoint.py   # LangGraph checkpointer setup

```


