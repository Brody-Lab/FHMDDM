"""
	fit_basic_glms!(model)

Fit a generalized linear model without association to any latent variable to each neuron's spike train

The accumulator-related modulation is set to be 0, and the coupling probability is set to be 1

MODIFIED ARGUMENT
-`model`: a structure containing the data, parameters, and hyperparameters of a drift-diffusion coupled generalized linear model
"""
function fit_basic_glms!(model::Model)
	for trialset in model.trialsets
		for mpGLM in trialset.mpGLMs
			@assert mpGLM.θ.c_u == 1.0
			mpGLM.θ.c[1] = Inf
			mpGLM.θ.v[1] = 0
			mpGLM.θ.β[1] = 0
			maximizeloglikelihood!(mpGLM)
		end
	end
end

"""
	maximizeloglikelihood!(mpGLM)

Learn the weights of the latent variable-independent inputs
"""
function maximizeloglikelihood!(mpGLM::MixturePoissonGLM)
	mpGLM.θ.𝐮 .= maximizeloglikelihood(mpGLM.Δt, mpGLM.𝐗[:,1:end-1], mpGLM.𝐲)
end

"""
    maximizeloglikelihood(Δt, 𝐗, 𝐲)

Learn the projection weights that maximize the log-likelihood of poisson observations

ARGUMENT
-`Δt`: time step duration, in seconds
-`𝐗`: design matrix. Rows correspond to samples, and columns to regressors

RETURN
-optimal weights
"""
function maximizeloglikelihood(Δt::AbstractFloat, 𝐗::Matrix{<:AbstractFloat}, 𝐲::Vector{<:Integer})
    derivatives = PoissonGLMDerivatives(Δt=Δt, 𝐗=𝐗, 𝐲=𝐲)
    f(𝐰) = negativeloglikelihood!(derivatives, 𝐰)
    ∇f!(∇, 𝐰) = ∇negativeloglikelihood!(∇, derivatives, 𝐰)
    ∇∇f!(∇∇, 𝐰) = ∇∇negativeloglikelihood!(∇∇, derivatives, 𝐰)
    results = Optim.optimize(f, ∇f!, ∇∇f!, rand(size(𝐗,2)), NewtonTrustRegion())
	Optim.minimizer(results)
end

"""
	negativeloglikelihood!(derivatives,𝐰)

Negative log-likelihood of the model given projection weights `𝐰`

The object `derivatives` is modified if `𝐰` differs from `derivatives.𝐰`
"""
function negativeloglikelihood!(derivatives::PoissonGLMDerivatives, 𝐰::Vector{<:Real})
    if (𝐰 != derivatives.𝐰) || isnan(derivatives.ℓ[1])
        derivatives.𝐰 .= 𝐰
        ∇∇loglikelihood!(derivatives)
    end
    -derivatives.ℓ[1]
end

"""
	∇negativeloglikelihood!(∇, derivatives, 𝐰)

Compute in `∇` the gradient of the negative of the log-likelihood with respect to `𝐰`
"""
function ∇negativeloglikelihood!(∇::Vector{<:AbstractFloat}, derivatives::PoissonGLMDerivatives, 𝐰::Vector{<:Real})
    if (𝐰 != derivatives.𝐰) || isnan(derivatives.ℓ[1])
        derivatives.𝐰 .= 𝐰
        ∇∇loglikelihood!(derivatives)
    end
    for i in eachindex(∇)
        ∇[i] = -derivatives.∇ℓ[i]
    end
end

"""
	∇∇negativeloglikelihood!(∇∇, derivatives, 𝐰)

Compute in `∇∇` the hessian of the negative of the log-likelihood with respect to `𝐰`
"""
function ∇∇negativeloglikelihood!(∇∇::Matrix{<:AbstractFloat}, derivatives::PoissonGLMDerivatives, 𝐰::Vector{<:Real})
    if (𝐰 != derivatives.𝐰) || isnan(derivatives.ℓ[1])
        derivatives.𝐰 .= 𝐰
        ∇∇loglikelihood!(derivatives)
    end
    for i in eachindex(∇∇)
        ∇∇[i] = -derivatives.∇∇ℓ[i]
    end
end

"""
	∇∇loglikelihood!(derivatives)

Compute the log-likelihood of a Poisson GLM, its gradient, and its hessian

EXAMPLE
```julia-repl
julia> using FHMDDM, ForwardDiff, Random
julia> Random.seed!(1234)
julia> 𝐰₀ = rand(10);
julia> derivatives = FHMDDM.PoissonGLMDerivatives(Δt=0.01, 𝐗=rand(100,10), 𝐲 = floor.(Int, rand(100).*3), 𝐰 = 𝐰₀)
julia> FHMDDM.∇∇loglikelihood!(derivatives)
julia> f(x) = FHMDDM.loglikelihood(derivatives, x)
julia> fauto = f(𝐰₀)
julia> gauto = ForwardDiff.gradient(f, 𝐰₀)
julia> hauto = ForwardDiff.hessian(f, 𝐰₀)
julia> println("   |Δℓ|: ", abs(fauto-derivatives.ℓ[1]))
julia> println("   max(|Δgradient|): ", maximum(abs.(gauto.-derivatives.∇ℓ)))
julia> println("   max(|Δhessian|): ", maximum(abs.(hauto.-derivatives.∇∇ℓ)))
```
"""
function ∇∇loglikelihood!(derivatives::PoissonGLMDerivatives)
	@unpack Δt, ℓ, ∇ℓ, ∇∇ℓ, 𝐰, 𝐗, 𝐲 = derivatives
    𝐋 = 𝐗*𝐰
    d²𝐥_d𝐋², d𝐥_d𝐋, 𝐥 = similar(𝐋), similar(𝐋), similar(𝐋)
    for t in eachindex(𝐋)
        d²𝐥_d𝐋²[t], d𝐥_d𝐋[t], 𝐥[t] = differentiate_twice_loglikelihood_wrt_linearpredictor(Δt, 𝐋[t], 𝐲[t])
    end
    ℓ .= sum(𝐥)
    ∇ℓ .= 𝐗'*d𝐥_d𝐋
    ∇∇ℓ .= 𝐗'*(d²𝐥_d𝐋².*𝐗)
	return nothing
end

"""
	loglikelihood(derivatives, 𝐰)

ForwardDiff-compatible computation of Poisson log-likelihood
"""
function loglikelihood(derivatives::PoissonGLMDerivatives, 𝐰::Vector{<:Real})
	𝐋 = derivatives.𝐗*𝐰
	ℓ = 0
	for (L,y) in zip(𝐋,derivatives.𝐲)
		ℓ += poissonloglikelihood(derivatives.Δt, L, y)
	end
	return ℓ
end
