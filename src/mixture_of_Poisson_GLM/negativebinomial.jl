"""
differentiate_loglikelihood_wrt_overdispersion_mean!(H, g, α, Δt, μ, y)
negbinlikelihood(α, Δt, μ, y)
negbinloglikelihood(α, Δt, μ, y)
probabilitysuccess(α, Δt, μ)
"""

"""
    differentiate_loglikelihood_wrt_overdispersion_mean!(H, g, α, Δt, μ, y)

Hessian and gradient of the log-likelihood of a negative binomial model given an observation

MODIFIED ARGUMENT
-`g`: 2-element gradient vector. The first parameter is the overdispersion `α`, and the second parameter is the mean `μ`
-`H`: 2-by-2 hessian matrix. 

UNMODIFIED ARGUMENT
-`α`: overdispersion parameter
-`Δt`: time step duration
-`μ`: mean
-`y`: observed count response

EXAMPLE
```julia-repl
julia> using FHMDDM, ForwardDiff, Distributions
julia> α, Δt, μ, y = 1.3, 0.01, 1.0, 2
julia> x0 = [α, μ]
julia> f(x) = FHMDDM.negbinloglikelihood(x[1], Δt, x[2], y)
julia> gauto = ForwardDiff.gradient(f,x0)
julia> Hauto = ForwardDiff.hessian(f,x0)
julia> ghand = zeros(2)
julia> Hhand = zeros(2,2)
julia> FHMDDM.differentiate_loglikelihood_wrt_overdispersion_mean!(ghand, Hhand, α, Δt, μ, y)
julia> println("The maximum absolute difference between the automatically and hand-computed gradient is ϵ=", maximum(abs.(gauto.-ghand)))
julia> println("The maximum absolute difference between the automatically and hand-computed hessian is ϵ=", maximum(abs.(Hauto.-Hhand)))
```
"""
function differentiate_loglikelihood_wrt_overdispersion_mean!(g::Vector{<:AbstractFloat}, H::Matrix{<:AbstractFloat}, α::AbstractFloat, Δt::AbstractFloat, μ::AbstractFloat, y::Integer)
    r = 1/α
    r² = r^2
    p = probabilitysuccess(α, Δt, μ)
    η = (y-μ*Δt)*p
    ∂ℓ_∂α = (digamma(r) - digamma(y+r) - log(p) + α*η)*r²
    ∂ℓ_∂μ = η/μ
    g[1] = ∂ℓ_∂α
    g[2] = ∂ℓ_∂μ
    μ² = μ^2
    ω = α*μ²*Δt^2
    p² = p^2
    ∂²ℓ_∂α² = ((y+ω)*p² + (trigamma(y+r) - trigamma(r))*r²)*r² - ∂ℓ_∂α*2r
    ∂²ℓ_∂α∂μ = Δt*(μ*Δt-y)*p²
    ∂²ℓ_∂μ² = (ω - y*(1+2α*μ*Δt))/μ²*p²
    H[1,1] = ∂²ℓ_∂α²
    H[2,1] = H[1,2] = ∂²ℓ_∂α∂μ
    H[2,2] = ∂²ℓ_∂μ²
    return nothing
end

"""
    differentiate_loglikelihood_wrt_overdispersion_mean!(g, α, Δt, μ, y)

Gradient of the log-likelihood of a negative binomial model given an observation

Documented in `differentiate_loglikelihood_wrt_overdispersion_mean!(H, g, α, Δt, μ, y)`
"""
function differentiate_loglikelihood_wrt_overdispersion_mean!(g::Vector{<:AbstractFloat}, α::AbstractFloat, Δt::AbstractFloat, μ::AbstractFloat, y::Integer)
    r = 1/α
    p = probabilitysuccess(α, Δt, μ)
    η = (y-μ*Δt)*p
    ∂ℓ_∂α = (digamma(r) - digamma(y+r) - log(p) + α*η)*r^2
    ∂ℓ_∂μ = η/μ
    g[1] = ∂ℓ_∂α
    g[2] = ∂ℓ_∂μ
    return nothing
end

"""
    negbinlikelihood(α, Δt, μ, y)

Likelihood of the negative binomial model given a count response

ARGUMENT
-`α`: the overdispersion parameter
-`Δt`: duration of the time step in seconds
-`μ`: mean
-`y`: spike count response

RETURN
-a scalar indicating the likelihood
"""
function negbinlikelihood(α::Real, Δt::AbstractFloat, μ::Real, y::Integer)
    r = 1/α
    p = probabilitysuccess(α,Δt,μ)
    pdf(NegativeBinomial(r,p),y)
end

"""
    negbinloglikelihood(α, Δt, μ, y)

Log-likelihood of the negative binomial model given a count response

ARGUMENT
-`a`: the overdispersion parameter in real space
-`Δt`: duration of the time step in seconds
-`μ`: mean
-`y`: spike count response

RETURN
-a scalar indicating the likelihood
"""
function negbinloglikelihood(α::Real, Δt::Real, μ::Real, y::Integer)
    r = 1/α
    p = probabilitysuccess(α,Δt,μ)
    logpdf(NegativeBinomial(r,p),y)
end

"""
    probabilitysuccess(α,Δt,μ)

Probability of success in a Bernoulli trial in a negative binomial model

ARGUMENT
-`α`: overdispersion parameter
-`Δt`: duration of a time step
-`μ`: mean parameter

RETURN
-a scalar ∈ [0,1]
"""
probabilitysuccess(α::Real, Δt::Real, μ::Real) = 1/(1+α*μ*Δt)