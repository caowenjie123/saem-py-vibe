# Repository Guidelines

## Project Structure & Module Organization
This repository is an R package. Core implementation lives in `R/` (algorithm steps, helpers, and S3/S4-style class logic). User-facing documentation is in `man/` as `.Rd` files. Sample datasets are stored in `data/` (e.g., `*.saemix.tab`). Package metadata and build configuration are in `DESCRIPTION`, `NAMESPACE`, `LICENSE`, and `CHANGES`. The `inst/` directory holds package assets such as `inst/CITATION`.

## Build, Test, and Development Commands
Run commands from the package root (`saemix-main`):
- `R CMD build .` — build a source tarball for distribution.
- `R CMD check .` — run CRAN-style checks (R CMD check is the primary QA gate here).
- `R -e "devtools::check()"` — optional, if you have `devtools` installed, for a developer-friendly check workflow.
- `R -e "roxygen2::roxygenise()"` — optional if you update roxygen comments in `R/` and need to regenerate `man/`.

## Coding Style & Naming Conventions
Follow existing R style in `R/`: use `<-` for assignment, keep indentation consistent (2 spaces in most files), and preserve naming patterns such as `func_*.R`, `main_*.R`, and `saemix.*` function prefixes. Keep roxygen-style comments (`#'`) aligned with functions and update documentation when signatures or behavior change.

## Testing Guidelines
There is no `tests/` directory in this checkout. The package suggests `testthat`, but automated tests are not currently part of the repo. Use `R CMD check` to validate changes. If you add tests, follow standard R conventions: place them under `tests/testthat/` and name files `test-*.R`.

## Commit & Pull Request Guidelines
This repository copy does not include Git history, so commit message conventions cannot be inferred. Use concise, imperative summaries (e.g., “Fix covariance validation”). For PRs, include a clear description, list any affected functions or datasets, and note the `R CMD check` results. Add or update documentation in `man/` when behavior changes.
