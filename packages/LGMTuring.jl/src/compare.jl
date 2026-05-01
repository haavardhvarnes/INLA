"""
    compare_posteriors(inla_fit::INLAResult, chain::Chains;
                       model::LatentGaussianModel,
                       tol_mean = 0.10, tol_sd = 0.20)
        -> Vector{@NamedTuple{name, inla_mean, nuts_mean, mean_abs_diff,
                              inla_sd, nuts_sd, sd_rel_diff, flagged}}

Side-by-side hyperparameter posterior summary diff between INLA and a
NUTS chain produced by [`nuts_sample`](@ref). Used by tier-3
triangulation tests to surface posterior-summary disagreements without
requiring an external statistician.

### Tolerances

- `tol_mean` — absolute difference threshold on the posterior mean,
  in units of `max(inla_sd, nuts_sd)` (i.e. \"more than `tol_mean` SDs
  apart\").
- `tol_sd` — relative difference threshold on the posterior SD,
  measured as `|nuts_sd - inla_sd| / inla_sd`.

A row is `flagged = true` if either the mean diff exceeds `tol_mean`
SDs *or* the SD relative diff exceeds `tol_sd`. Tests typically assert
that no rows are flagged.

### Example

```julia
inla_fit = inla(model, y)
chain = nuts_sample(model, y, 1000; init_from_inla = inla_fit)
diff = compare_posteriors(inla_fit, chain; model = model)
@test all(!row.flagged for row in diff)
```
"""
function compare_posteriors(inla_fit::INLAResult, chain::Chains;
        model::LatentGaussianModel,
        tol_mean::Real=0.10,
        tol_sd::Real=0.20)
    inla_rows = hyperparameters(model, inla_fit)
    names = Symbol.([r.name for r in inla_rows])

    chain_names = Symbol.(string.(chain.name_map.parameters))
    nuts_means = Dict{Symbol, Float64}()
    nuts_sds = Dict{Symbol, Float64}()
    for nm in chain_names
        col = vec(Array(chain[nm]))
        nuts_means[nm] = mean(col)
        nuts_sds[nm] = std(col)
    end

    rows = NamedTuple{
        (:name, :inla_mean, :nuts_mean, :mean_abs_diff,
            :inla_sd, :nuts_sd, :sd_rel_diff, :flagged),
        Tuple{String, Float64, Float64, Float64,
            Float64, Float64, Float64, Bool}}[]
    for (k, name) in enumerate(names)
        haskey(nuts_means, name) ||
            throw(ArgumentError("hyperparameter \"$name\" missing from chain " *
                                "(have $(collect(keys(nuts_means))))"))
        im = inla_rows[k].mean
        is = inla_rows[k].sd
        nm = nuts_means[name]
        ns = nuts_sds[name]
        scale = max(is, ns, eps())
        mdiff = abs(im - nm)
        sd_rel = is > 0 ? abs(ns - is) / is : abs(ns - is)
        flagged = (mdiff / scale > tol_mean) || (sd_rel > tol_sd)
        push!(rows,
            (name=String(name),
                inla_mean=im, nuts_mean=nm,
                mean_abs_diff=mdiff,
                inla_sd=is, nuts_sd=ns,
                sd_rel_diff=sd_rel,
                flagged=flagged))
    end
    return rows
end
