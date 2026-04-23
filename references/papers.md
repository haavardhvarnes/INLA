# References — annotated bibliography

Key papers and books for the port. Grouped by role in the project.

## Method foundation (read first)

**Rue, H., Martino, S., & Chopin, N. (2009).** Approximate Bayesian
inference for latent Gaussian models by using integrated nested Laplace
approximations. *JRSS-B*, 71(2), 319–392.
> The INLA paper. Defines latent Gaussian models, the Laplace approximation
> scheme, CCD for hyperparameter integration. Read §2–3 for the spine of
> the algorithm.

**Lindgren, F., Rue, H., & Lindström, J. (2011).** An explicit link
between Gaussian fields and Gaussian Markov random fields: the stochastic
partial differential equation approach. *JRSS-B*, 73(4), 423–498.
> SPDE. The Matérn–GMRF link via Hilbert space projection. §2–4 are the
> theory; §5 gives the FEM matrices you need to reproduce (C, G₁, G₂).

**Lindgren, F., & Rue, H. (2015).** Bayesian Spatial Modelling with
R-INLA. *Journal of Statistical Software*, 63(19).
> The R-INLA SPDE interface paper. Reference for how the API was
> shaped.

**Lindgren, F., Bolin, D., & Rue, H. (2022).** The SPDE approach for
Gaussian and non-Gaussian fields: 10 years and still running. *Spatial
Statistics*, 50, 100599.
> The review of the SPDE approach. §4 covers non-stationarity and
> non-Gaussian extensions we probably won't implement in v1 but must
> understand to avoid painting ourselves into a corner.

## Priors and parameterization

**Simpson, D., Rue, H., Riebler, A., Martins, T., & Sørbye, S. (2017).**
Penalising model component complexity: A principled, practical approach to
constructing priors. *Statistical Science*, 32(1), 1–28.
> PC priors. This defines the prior parameterization used for precisions,
> ranges, and mixing parameters throughout R-INLA. Required reading.

**Fuglstad, G.-A., Simpson, D., Lindgren, F., & Rue, H. (2019).**
Constructing priors that penalize the complexity of Gaussian random
fields. *JASA*, 114(525), 445–452.
> PC priors specifically for SPDE–Matérn (range and σ). Phase 4 deliverable.

**Sørbye, S. H., & Rue, H. (2014).** Scaling intrinsic Gaussian Markov
random field priors in spatial modelling. *Spatial Statistics*, 8, 39–51.
> Why and how to scale intrinsic GMRFs. `scale.model = TRUE` comes from
> here. Hyperparameter interpretability across graphs depends on this.

**Riebler, A., Sørbye, S. H., Simpson, D., & Rue, H. (2016).** An
intuitive Bayesian spatial model for disease mapping that accounts for
scaling. *SMMR*, 25(4), 1145–1165.
> BYM2. The reparameterization that cleaned up BYM's identifiability
> issues. Our Phase 2 headline model.

**Freni-Sterrantino, A., Ventrucci, M., & Rue, H. (2018).** A note on
intrinsic conditional autoregressive models for disconnected graphs.
*SSTE*, 26, 25–34.
> The disconnected-components fix. Without this you silently compute
> wrong posteriors on graphs with islands/isolated nodes.

## Algorithmic & computational

**Martins, T. G., Simpson, D., Lindgren, F., & Rue, H. (2013).** Bayesian
computing with INLA: New features. *CSDA*, 67, 68–83.
> The PC prior case studies, the `control.mode` and copy mechanism, multiple
> likelihoods. Useful for understanding R-INLA's feature surface.

**Lindgren, F., Bakka, H., Bolin, D., Krainski, E., & Rue, H. (2024).** A
diffusion-based spatio-temporal extension of Gaussian Matérn fields.
*SORT*, 48(1), 3–66.
> Non-separable space-time SPDEs. Phase 5/6.

**Bolin, D., & Kirchner, K. (2020).** The rational SPDE approach for
Gaussian random fields with general smoothness. *JCGS*, 29(2), 274–285.
> Fractional-α SPDE via rational approximations. Phase 6 optimization.

**Van Niekerk, J., Krainski, E., Rustand, D., & Rue, H. (2023).** A new
avenue for Bayesian inference with INLA. *CSDA*, 181, 107692.
> Recent developments in the INLA algorithm itself, including the current
> parallelization scheme. Useful to know before Phase 6 perf work.

## Textbooks (validation oracles)

**Moraga, P. (2024).** *Geospatial Health Data: Modeling and Visualization
with R-INLA and Shiny* (2nd ed.). Chapman & Hall/CRC.
> Phase 4 tier-4 test suite. Every chapter is implicitly a regression test.
> Code online.

**Blangiardo, M., & Cameletti, M. (2015).** *Spatial and Spatio-Temporal
Bayesian Models with R-INLA*. Wiley.
> The canonical textbook. Datasets and code online.

**Gómez-Rubio, V. (2020).** *Bayesian Inference with INLA*. Chapman &
Hall/CRC.
> More advanced: INLA-within-MCMC, missing data, mixture models. Many
> useful test cases for Phase 5.

**Krainski, E. T., Gómez-Rubio, V., Bakka, H., Lenzi, A.,
Castro-Camilo, D., Simpson, D., Lindgren, F., & Rue, H. (2019).** *Advanced
Spatial Modeling with Stochastic Partial Differential Equations Using R
and INLA*. Chapman & Hall/CRC.
> SPDE cookbook. Phase 4 reference.

## Code references

- `hrue/r-inla` — the reference implementation.
  - `gmrflib/` for the sparse GMRF core
  - `inlaprog/` for the INLA loop
  - `fmesher/` for meshing
  - `rinla/R/` for the R wrapper layer (model definitions, formula parsing)
- `hrue/r-inla-testing` — developer scratch tests.
- `spatialstatisticsupna/Comparing-R-INLA-and-NIMBLE` — NIMBLE cross-checks.
- `ConnorDonegan/Stan-IAR` — Stan ICAR/BYM/BYM2 implementations.
- `inlabru-org/fmesher` — Lindgren's standalone C++ meshing (cleaner than
  the version inside r-inla).

## Related Julia packages (for interop and comparison)

- `JuliaGaussianProcesses/AbstractGPs.jl` — GP infrastructure, dense.
- `JuliaEarth/GeoStats.jl` — classical geostatistics, Meshes.jl-based.
- `SciML/NonlinearSolve.jl`, `LinearSolve.jl`, `Optimization.jl`.
- `TuringLang/Turing.jl` — downstream bridge via LogDensityProblems.
- `rafaqz/Rasters.jl` — gridded covariates and prediction surfaces.
