# Two-argument log-sum-exp used by every zero-inflated likelihood
# (forming `log(π + (1 - π) f(y, μ))` via `logsumexp(log π, log(1 - π) +
# log f)`). Inlined and branchless on the maxima to avoid an extra
# branch inside the Newton hot path.
@inline function logsumexp2(a::Real, b::Real)
    m = max(a, b)
    return m + log(exp(a - m) + exp(b - m))
end
