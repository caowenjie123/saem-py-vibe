# AGENTS.md

## OVERVIEW
Test data and R package reference files.

## STRUCTURE
saemix-main/
├── data/             # Test datasets
├── inst/             # R package installation files
├── man/              # R documentation
├── R/                # R source code
├── AGENTS.md         # This file
├── CHANGES           # Change log
├── DESCRIPTION       # R package description
├── LICENSE           # License
├── NAMESPACE         # R namespace
└── README.md         # R package README

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Test data | data/ | Sample datasets for testing |
| R code | R/ | Original R implementation |
| Documentation | man/ | R help files |

## CONVENTIONS
R-style naming in R/ directory.

## ANTI-PATTERNS
Do not modify R files unless porting changes.
