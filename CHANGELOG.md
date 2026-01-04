# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-01-04

### Added

- Initial release of saemix Python package
- SAEM (Stochastic Approximation Expectation Maximization) algorithm implementation
- Support for nonlinear mixed effects models parameter estimation
- Core modules:
  - `SaemixData`: Data handling and validation
  - `SaemixModel`: Model definition with structural and covariate models
  - `SaemixControl`: Algorithm control parameters
  - `SaemixResults`: Results storage and access
- Algorithm components:
  - E-step implementation with MCMC sampling
  - M-step for parameter updates
  - Fisher Information Matrix computation
  - MAP estimation for individual parameters
  - Likelihood computation (Gaussian quadrature and importance sampling)
- Diagnostic tools:
  - Individual and population predictions
  - Residual calculations (IWRES, PWRES, NPDE)
  - Conditional distribution sampling
- Export functionality for results to CSV files
- Simulation capabilities for model-based data generation
- Stepwise covariate selection (forward and backward procedures)
- Model comparison utilities
- Optional plotting support with matplotlib

### Dependencies

- numpy >= 1.20.0
- pandas >= 1.3.0
- scipy >= 1.7.0
- matplotlib >= 3.4.0 (optional, for plotting)
