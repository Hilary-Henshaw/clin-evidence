# ClinEvidence Developer Guide

---

## Project Structure

```
clinevidence/
├── src/clinevidence/
│   ├── __init__.py           # Version and author
│   ├── settings.py           # Pydantic settings
│   ├── main.py               # FastAPI app factory
│   ├── middleware.py         # Request tracing
│   ├── dependencies.py       # FastAPI DI functions
│   ├── models/
│   │   ├── requests.py       # Pydantic request schemas
│   │   └── responses.py      # Pydantic response schemas
│   ├── agents/
│   │   ├── state.py          # LangGraph WorkflowState
│   │   ├── orchestrator.py   # LangGraph workflow
│   │   ├── safety_filter.py  # Input/output guardrails
│   │   ├── conversation.py   # General clinical Q&A
│   │   ├── rag/
│   │   │   ├── pipeline.py       # KnowledgeBase class
│   │   │   ├── document_extractor.py
│   │   │   ├── document_formatter.py
│   │   │   ├── knowledge_store.py
│   │   │   ├── query_enricher.py
│   │   │   ├── result_ranker.py
│   │   │   └── answer_synthesizer.py
│   │   ├── search/
│   │   │   ├── tavily_client.py
│   │   │   ├── pubmed_client.py
│   │   │   ├── evidence_searcher.py
│   │   │   └── search_processor.py
│   │   └── imaging/
│   │       ├── modality_detector.py
│   │       ├── brain_mri.py
│   │       ├── chest_xray.py
│   │       ├── skin_lesion.py
│   │       └── router.py
│   ├── api/
│   │   ├── chat.py
│   │   ├── media.py
│   │   └── speech.py
│   └── scripts/
│       └── ingest.py
├── tests/
│   ├── conftest.py
│   ├── unit/
│   └── integration/
├── docs/
├── examples/
└── .github/workflows/
```

---

## Adding a New Agent

1. **Create the agent module** in `src/clinevidence/agents/`:
   ```python
   # src/clinevidence/agents/my_new_agent.py
   from __future__ import annotations
   import logging
   from clinevidence.settings import Settings

   logger = logging.getLogger(__name__)

   class MyNewAgent:
       def __init__(self, settings: Settings) -> None:
           self._settings = settings

       def process(self, query: str) -> str:
           # Implementation here
           ...
   ```

2. **Add the agent to `WorkflowState`** in `state.py` if new
   state fields are needed.

3. **Register in the orchestrator** (`orchestrator.py`):
   - Add instance variable in `__init__`
   - Add node method `_run_my_new_agent`
   - Register with `builder.add_node`
   - Add routing case in `_route_after_selection`
   - Add `builder.add_edge` connecting to next node
   - Add the agent key to the routing prompt in
     `_select_agent`

4. **Add the agent string to constants**:
   - The agent selection strings are: CONVERSATION,
     KNOWLEDGE_BASE, WEB_EVIDENCE, BRAIN_MRI,
     CHEST_XRAY, SKIN_LESION

5. **Write tests** in `tests/unit/test_my_new_agent.py`

---

## Adding a New Imaging Model

1. Create a new analyser in `src/clinevidence/agents/imaging/`:
   - Define the PyTorch model architecture as `nn.Module`
   - Implement `analyse(image_path: Path) -> ImagingResult`
   - Import `ImagingResult` from `chest_xray.py`
   - Implement lazy model loading with `_get_model()`

2. Add a new model path to `Settings` in `settings.py`:
   ```python
   my_model_path: str = "./models/my_model.pth"
   ```

3. Add the image type to `ModalityDetector` constants:
   - Add to `_VALID_TYPES` frozenset
   - Update the detection prompt

4. Register in `ImagingRouter`:
   - Add an instance of your analyser
   - Add a routing case in `route_and_analyse`

5. Add a new agent node in `WorkflowOrchestrator`

---

## Running Tests

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run all tests
pytest tests/ -v

# Run unit tests only
pytest tests/unit/ -v

# Run integration tests only
pytest tests/integration/ -v

# Run with coverage report
pytest tests/ --cov=src/clinevidence --cov-report=html
open htmlcov/index.html

# Run a specific test
pytest tests/unit/test_safety_filter.py::TestSafetyFilterInput::test_input_allows_valid_medical_query -v
```

Tests use `unittest.mock.patch` for all external dependencies
(LLM, Qdrant, Tavily, etc.) and do not require API keys.

---

## Code Style

ClinEvidence uses Ruff for linting and formatting.

```bash
# Check for issues
ruff check src/ tests/

# Format code
ruff format src/ tests/

# Type checking
mypy src/clinevidence
```

Key style requirements:
- Maximum 79 characters per line
- `from __future__ import annotations` at top of every file
- Full type hints including return types
- `logging.getLogger(__name__)` only — no `print()`
- No TODO comments or stub functions in production code

---

## How the LangGraph Workflow Works

The `WorkflowOrchestrator._build_graph()` method constructs
a `StateGraph[WorkflowState]`. Key concepts:

**Nodes** are Python methods that receive the current state
and return a dict of updates. LangGraph merges the returned
dict into the state using the defined reducers.

**Conditional edges** call a routing function that returns a
string key, which maps to the next node. This enables dynamic
routing based on LLM decisions or state values.

**Interrupts** (`langgraph.types.interrupt`) pause execution
at a specific node (before `await_validation`). The graph
state is persisted by `MemorySaver`. Calling
`graph.invoke(Command(resume=value), config)` resumes from
the interrupt point.

**The `add_messages` reducer** in `WorkflowState.messages`
appends new messages rather than replacing the list, enabling
conversation history tracking across nodes.

---

## Contributing a New Imaging Model: Checklist

- [ ] Model architecture defined as `nn.Module` subclass
- [ ] Model loads from file using `torch.load(..., weights_only=True)`
- [ ] Device auto-detection (`cuda` if available, else `cpu`)
- [ ] Lazy loading on first `analyse()` call
- [ ] `FileNotFoundError` raised with helpful message if weights missing
- [ ] `ImagingResult` TypedDict returned
- [ ] Explanation string built with confidence label
- [ ] Unit tests with mocked `torch.load`
- [ ] Model path added to `Settings`
- [ ] Router updated
- [ ] Orchestrator node added
