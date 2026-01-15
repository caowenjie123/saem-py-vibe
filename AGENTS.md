<!-- OPENSPEC:START -->
# OpenSpec Instructions

These instructions are for AI assistants working in this project.

Always open `@/openspec/AGENTS.md` when the request:
- Mentions planning or proposals (words like proposal, spec, change, plan)
- Introduces new capabilities, breaking changes, architecture shifts, or big performance/security work
- Sounds ambiguous and you need the authoritative spec before coding

Use `@/openspec/AGENTS.md` to learn:
- How to create and apply change proposals
- Spec format and conventions
- Project structure and guidelines

Keep this managed block so 'openspec update' can refresh the instructions.

<!-- OPENSPEC:END -->
# AGENTS.md
- Editor rules: no `.cursorrules`, `.cursor/rules/`, or `.github/copilot-instructions.md`.
- Install/build: `pip install -r requirements.txt`; `pip install -e .`; `pip install ".[dev]"` / `".[plot]"`; `python -m build`.
- Tests: `pytest`; `pytest tests/test_data.py`; `pytest tests/test_data.py::TestSaemixData::test_basic_creation`; `HYPOTHESIS_PROFILE=ci|dev|debug pytest tests/`.
- Lint/style/type: `black saemix/`; `isort saemix/`; `ruff check saemix/`; `mypy saemix/`; imports stdlib→third-party→local, no wildcard; Black line length 88, 4-space indent, no trailing whitespace; annotate signatures, `Optional[...]` for nullable arrays; naming `snake_case`/`PascalCase`/`UPPER_SNAKE_CASE`; `TypeError` for wrong types, `ValueError` for invalid values; use `numpy.random.Generator` passed through with seed in `saemix_control()`; model functions use 0-based `id`/`xidep`/`psi`.
