# Contributing to ClinEvidence

Thank you for your interest in contributing. This guide covers
the setup, workflow, and standards for contributors.

---

## Development Setup

```bash
# Fork and clone the repository
git clone https://github.com/your-username/clinevidence.git
cd clinevidence

# Create a virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install with dev dependencies
pip install -e ".[dev]"

# Set up environment
cp .env.example .env
# Fill in test values (real API keys needed for integration tests)
```

---

## Branch Strategy

| Branch | Purpose |
|---|---|
| `main` | Stable releases only; protected branch |
| `dev` | Integration branch; all features merge here first |
| `feature/<name>` | New features branched from `dev` |
| `fix/<name>` | Bug fixes branched from `dev` |
| `docs/<name>` | Documentation updates |

**Workflow:**
1. Branch from `dev`: `git checkout -b feature/my-feature dev`
2. Develop and test locally
3. Open a PR targeting `dev`
4. After review and CI passes, merge to `dev`
5. Periodic releases merge `dev` to `main`

---

## Commit Conventions

ClinEvidence follows Conventional Commits:

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `refactor`: Code change that is neither fix nor feature
- `test`: Adding or updating tests
- `chore`: Build process, CI, or tooling updates
- `perf`: Performance improvement

**Examples:**
```
feat(rag): add cross-encoder reranking to result ranker
fix(imaging): handle missing model weights gracefully
docs(api): add curl examples to api-reference.md
test(safety): add output safety filter unit tests
```

---

## Pull Request Process

1. Ensure all tests pass: `make test`
2. Ensure code is formatted: `make format`
3. Ensure lint passes: `make lint`
4. Ensure type checks pass: `make type-check`
5. Write or update tests for any changed behaviour
6. Update documentation if adding new settings or endpoints
7. Fill in the PR template with a description, test plan,
   and any breaking changes

**PR Title Format:** Use the same Conventional Commits format.

---

## Test Requirements

- All new code must have corresponding tests
- Minimum 80% coverage is enforced by CI
- Unit tests must mock all external services
  (LLM, Qdrant, Tavily, Eleven Labs)
- Tests must be able to run without real API keys

```bash
# Run all tests with coverage
make test

# Run unit tests only
make test-unit

# Run integration tests only
make test-integration
```

---

## Code Style

- Maximum 79 characters per line (Ruff enforced)
- `from __future__ import annotations` in every Python file
- Full type hints on all functions and methods
- `logging.getLogger(__name__)` only — never `print()`
- No TODO comments in committed code
- No stub functions or placeholder implementations
- Docstrings on all public classes and methods

```bash
# Check style
make lint

# Auto-format
make format

# Type check
make type-check
```

---

## Clinical Safety Guidelines

Since ClinEvidence is used in clinical contexts:

- Never weaken safety filter logic without clinical review
- All imaging model changes require clinical validation
- Medical disclaimers must be present in all response types
- Do not raise `kb_min_confidence` above 0.8 without testing
- Human-in-the-loop validation for imaging must not be bypassed

---

## Questions?

Open an issue on GitHub for questions, bug reports, or
feature requests. Tag appropriately:
- `bug` — Something is broken
- `enhancement` — New feature request
- `question` — General question
- `documentation` — Docs improvement needed
