# AGENTS.md

## OVERVIEW
Core SAEM algorithm implementation modules.

## STRUCTURE
algorithm/
├── __init__.py
├── saem.py          # Main SAEM algorithm
├── estep.py         # E-step implementation
├── mstep.py         # M-step implementation
├── initialization.py# Parameter initialization
├── likelihood.py    # Likelihood calculations
├── fim.py           # Fisher information matrix
├── map_estimation.py# MAP estimation
├── predict.py       # Prediction functions
└── conddist.py      # Conditional distributions

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Main algorithm | saem.py | Core SAEM loop |
| E-step | estep.py | Expectation step |
| M-step | mstep.py | Maximization step |
| Initialization | initialization.py | Starting values |
| Likelihood | likelihood.py | LL calculations |

## CONVENTIONS
Mathematical functions with detailed docstrings.

## ANTI-PATTERNS
Avoid modifying core algorithm without tests.