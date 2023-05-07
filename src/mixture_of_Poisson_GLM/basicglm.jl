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
			maximizelikelihood!(mpGLM)
			basicglm = PoissonGLM(Δt=mpGLM.Δt, 𝐗=mpGLM.𝐗[:,1:end-1], 𝐲=mpGLM.𝐲)
			maximizelikelihood!(basicglm)
			mpGLM.θ.𝐮 .= basicglm.𝐰
		end
	end
end

"""
    maximizelikelihood!(glm::PoissonGLM)

Learn the maximum likelihood parameters of a Poisson generalized linear model
"""
function maximizelikelihood!(glm::PoissonGLM; iterations::Integer=20)
    f(𝐰) = negativeloglikelihood!(glm, 𝐰)
    ∇f!(∇, 𝐰) = ∇negativeloglikelihood!(∇, glm, 𝐰)
    ∇∇f!(∇∇, 𝐰) = ∇∇negativeloglikelihood!(∇∇, glm, 𝐰)
    results = Optim.optimize(f, ∇f!, ∇∇f!, rand(size(glm.𝐗,2)), NewtonTrustRegion(), Optim.Options(iterations=20))
	glm.𝐰 .= Optim.minimizer(results)
	return nothing
end

"""
	negativeloglikelihood!(glm,𝐰)

Negative log-likelihood of the model given projection weights `𝐰`

The object `glm` is modified if `𝐰` differs from `glm.𝐰`
"""
function negativeloglikelihood!(glm::PoissonGLM, 𝐰::Vector{<:Real})
    update!(glm, 𝐰)
    -glm.ℓ[1]
end

"""
	∇negativeloglikelihood!(∇, glm, 𝐰)

Compute in `∇` the gradient of the negative of the log-likelihood with respect to `𝐰`
"""
function ∇negativeloglikelihood!(∇::Vector{<:AbstractFloat}, glm::PoissonGLM, 𝐰::Vector{<:Real})
    update!(glm, 𝐰)
    for i in eachindex(∇)
        ∇[i] = -glm.∇ℓ[i]
    end
	return nothing
end

"""
	∇∇negativeloglikelihood!(∇∇, glm, 𝐰)

Compute in `∇∇` the hessian of the negative of the log-likelihood with respect to `𝐰`
"""
function ∇∇negativeloglikelihood!(∇∇::Matrix{<:AbstractFloat}, glm::PoissonGLM, 𝐰::Vector{<:Real})
    update!(glm, 𝐰)
    for i in eachindex(∇∇)
        ∇∇[i] = -glm.∇∇ℓ[i]
    end
	return nothing
end

"""
	update!(glm::PoissonGLM, 𝐰)
"""
function update!(glm::PoissonGLM, 𝐰::Vector{<:Real})
	if (𝐰 != glm.𝐰) || isnan(glm.ℓ[1])
		glm.𝐰 .= 𝐰
		∇∇loglikelihood!(glm)
	end
end

"""
	∇∇loglikelihood!(glm)

Compute the log-likelihood of a Poisson GLM, its gradient, and its hessian

EXAMPLE
```julia-repl
julia> using FHMDDM, ForwardDiff, Random
julia> Random.seed!(1234)
julia> 𝐰₀ = rand(10);
julia> glm = FHMDDM.PoissonGLM(Δt=0.01, 𝐗=rand(100,10), 𝐲 = floor.(Int, rand(100).*3), 𝐰 = 𝐰₀)
julia> FHMDDM.∇∇loglikelihood!(glm)
julia> f(x) = FHMDDM.loglikelihood(glm, x)
julia> fauto = f(𝐰₀)
julia> gauto = ForwardDiff.gradient(f, 𝐰₀)
julia> hauto = ForwardDiff.hessian(f, 𝐰₀)
julia> println("   |Δℓ|: ", abs(fauto-glm.ℓ[1]))
julia> println("   max(|Δgradient|): ", maximum(abs.(gauto.-glm.∇ℓ)))
julia> println("   max(|Δhessian|): ", maximum(abs.(hauto.-glm.∇∇ℓ)))
```
"""
function ∇∇loglikelihood!(glm::PoissonGLM)
	@unpack Δt, ℓ, ∇ℓ, ∇∇ℓ, 𝐰, 𝐗, 𝐲 = glm
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
	loglikelihood(glm::PoissonGLM)
"""
loglikelihood(glm::PoissonGLM) = loglikelihood(glm, glm.𝐰)

"""
	loglikelihood(glm, 𝐰)

ForwardDiff-compatible computation of Poisson log-likelihood
"""
function loglikelihood(glm::PoissonGLM, 𝐰::Vector{<:Real})
	𝐋 = glm.𝐗*𝐰
	ℓ = 0
	for (L,y) in zip(𝐋,glm.𝐲)
		ℓ += log(poissonlikelihood(glm.Δt, L, y))
	end
	return ℓ
end

"""
	loglikelihood(Δt, kfold, 𝐗, 𝐲)

Out-of-sample log-likelihood of a basic Poisson GLM

ARGUMENT
-`glm`: an object containing the quantities of a basic GLM
-`kfold`: number of cross-validation folds
"""
function loglikelihood(glm::PoissonGLM, kfold::Integer)
	@unpack Δt, 𝐗, 𝐲 = glm
    testindices, trainindices = cvpartition(kfold, size(𝐗,1))
	ℓ = 0
	for k = 1:kfold
		trainglm = PoissonGLM(Δt=Δt, 𝐗=𝐗[trainindices[k],:], 𝐲=𝐲[trainindices[k]])
		maximizelikelihood!(trainglm)
		testglm = PoissonGLM(Δt=Δt, 𝐰=trainglm.𝐰, 𝐗=𝐗[testindices[k],:], 𝐲=𝐲[testindices[k]])
		ℓ += loglikelihood(testglm)
	end
	return ℓ
end
