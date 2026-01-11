# AGENTS.md

## OVERVIEW
Core saemix package containing main modules for data, model, control, results, and utilities.

## STRUCTURE
saemix/
├── __init__.py          # Package initialization
├── data.py              # SaemixData class
├── model.py             # SaemixModel class
├── control.py           # Control parameters
├── results.py           # Results container
├── main.py              # Main saemix() function
├── diagnostics.py       # Diagnostic plots
├── utils.py             # Utility functions
├── compare.py           # Model comparison
├── export.py            # Export functions
├── plot_options.py      # Plot options
├── simulation.py        # Simulation functions
├── stepwise.py          # Stepwise procedures
└── algorithm/           # Core algorithm implementations

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Data handling | data.py | SaemixData class |
| Model definition | model.py | SaemixModel class |
| Run estimation | main.py | saemix() function |
| Results analysis | results.py | SaemixObject class |
| Diagnostics | diagnostics.py | Plot functions |
| Algorithm details | algorithm/ | SAEM implementation |

## CONVENTIONS
Follow parent Python conventions. All functions use snake_case.

## ANTI-PATTERNS
None specific to this directory.