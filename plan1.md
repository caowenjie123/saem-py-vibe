# SAEM-Py Improvement Plan

## Prioritized Roadmap

### P0 — Correctness & Reproducibility (High impact, low/med effort)
1. **Thread RNG end-to-end**
   - Pass `control["rng"]` from `saemix()` to `run_saem()` and downstream calls.
   - Replace global RNG use (`np.random.*`, `scipy.stats.*.rvs`) with the shared `Generator`.
2. **Fix data mutation for exponential error**
   - Avoid mutating `SaemixData` in `saemix/main.py`; apply log transform only within likelihood calculations.
3. **Replace `np.linalg.inv` with stable solves**
   - Use `solve` / Cholesky for omega/precision matrices across `estep.py`, `likelihood.py`, `map_estimation.py`.

### P1 — Performance Hotspots (High impact, med effort)
4. **E-step kernel optimizations** (`saemix/algorithm/estep.py`)
   - Precompute `U_eta`, avoid per-iteration copies, vectorize quadratic forms.
5. **Quadrature & MAP stability**
   - Reuse decompositions, reduce redundant evaluations, tighten numerical steps.

### P2 — Feature Completeness (Med impact, med effort)
6. **Implement full FIM** (`saemix/algorithm/fim.py`)
   - Port algorithm from R `saemix` or implement numerical differentiation-based FIM.

### P3 — Developer Experience & Quality (Med impact, low effort)
7. **Documentation accessibility**
   - Add English README/Quickstart and expand `saemix_control` + error model docs.
8. **Quality tools**
   - Add `mypy` config, Ruff config in `pyproject.toml`, and CI enforcement.
9. **Diagnostics UX**
   - Centralize plot styling via `plot_options.py` and remove hard-coded styles.

---

## Task Plan

### Phase 1: Reproducibility & Stability
- [ ] Identify all RNG entry points and replace global RNG usage
- [ ] Update `saemix()` → `run_saem()` → algorithms to pass `rng`
- [ ] Replace `np.linalg.inv` in E-step, MAP, likelihood with `solve`/Cholesky
- [ ] Ensure exponential error models do not mutate `SaemixData`

### Phase 2: Performance Wins
- [ ] Optimize E-step kernels (precompute, avoid copies)
- [ ] Vectorize gradient calculations where possible
- [ ] Reduce redundant likelihood evaluations

### Phase 3: Algorithm Completeness
- [ ] Port/implement full FIM calculation
- [ ] Add regression tests or golden comparisons for FIM

### Phase 4: Docs & DX
- [ ] English README + quickstart
- [ ] Full `saemix_control` parameter reference
- [ ] Add `mypy`/Ruff config and CI checks
- [ ] Normalize plotting style system
