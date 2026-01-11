# AGENTS.md

## OVERVIEW
Test suite for saemix package.

## STRUCTURE
tests/
├── __init__.py
├── conftest.py        # Pytest configuration
├── test_*.py          # Individual test files
└── test_*.py

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Test config | conftest.py | Fixtures and setup |
| Data tests | test_data.py | SaemixData tests |
| Model tests | test_model.py | SaemixModel tests |
| Integration | test_integration*.py | Full pipeline tests |

## CONVENTIONS
Use pytest fixtures. Test functions named test_*.

## ANTI-PATTERNS
Do not skip tests without justification.