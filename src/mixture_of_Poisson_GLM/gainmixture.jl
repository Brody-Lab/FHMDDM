"""
	loglikelihood(glm::GainMixtureGLM, kfold)

Out-of-sample log-likelihood of a mixture of gain GLM

ARGUMENT
-`glm`: an object containing the data and parameters of a mixture of gain GLM
-`kfold`: number of cross-validation folds
"""
function loglikelihood(glm::GainMixtureGLM, kfold::Integer)
    testindices, trainindices = cvpartition(kfold, size(glm.𝐗,1))
    ℓ = 0
    for k = 1:kfold
		trainingglm = GainMixtureGLM(Δt=glm.Δt, 𝐗=glm.𝐗[trainindices[k],:], 𝐲=glm.𝐲[trainindices[k]])
		maximizelikelihood!(trainingglm)
		testglm = GainMixtureGLM(Δt=glm.Δt, 𝐗=glm.𝐗[testindices[k],:], 𝐲=glm.𝐲[testindices[k]])
		for parameter in (:𝐠, :π, :𝐮)
			getfield(testglm, parameter) .= getfield(trainingglm, parameter)
		end
		ℓ += loglikelihood(testglm)
    end
	return ℓ
end

"""
    maximizelikelihood!(glm::GainMixtureGLM)

Fit a mixture of gain Poisson generalized linear model (GLM)

ARGUMENT
-`glm`: an object containing the data and parameters of a mixture of gain GLM

OPTIONAL ARGUMENT
-`iterations`: maximum number of iterations in the expectation maximization algorithm, and within each M-step, maximum number of iterations within the Newton with trust region algorithm
-`nstarts`: number of repeated optimizations from different initial values

EXAMPLE
```julia-repl
julia> using FHMDDM, Random, Distributions
julia> Random.seed!(1234);
julia> Δt, π₀, T = 0.01, 0.5, 10000;
julia> 𝐠₀ = [50.0, 0.0];
julia> 𝐮₀ = [1.0, 1.0]
julia> 𝐆 = fill(1.0, T);
julia> t = collect(1:T)./T*2π
julia> 𝐔 = 10 .*hcat(sin.(t), cos.(t))
julia> 𝐗 = hcat(𝐆, 𝐔)
julia> 𝐋 = collect(𝐗*vcat(g,𝐮₀) for g in 𝐠₀);
julia> 𝛌 = collect(rand() < π₀ ? FHMDDM.inverselink(𝐋[1][t]) : FHMDDM.inverselink(𝐋[2][t]) for t=1:T)
julia> 𝐲 = collect(rand(Poisson(λ*Δt)) for λ in 𝛌)
julia> glm = FHMDDM.GainMixtureGLM(Δt=Δt, 𝐗=𝐗, 𝐲=𝐲)
julia> FHMDDM.maximizelikelihood!(glm)
julia> ℓ = FHMDDM.loglikelihood(glm)
julia> glm.𝐠, glm.𝐮, glm.π
julia> basicglm = FHMDDM.PoissonGLM(Δt=Δt, 𝐗=𝐗, 𝐲=𝐲)
julia> FHMDDM.maximizelikelihood!(basicglm)
julia> ℓbasic = FHMDDM.loglikelihood(basicglm)

julia> testℓmixture = FHMDDM.loglikelihood(glm,5)
julia> testℓbasic = FHMDDM.loglikelihood(basicglm,5)

```
"""
function maximizelikelihood!(glm::GainMixtureGLM; iterations::Integer=20, nstarts::Integer=10, πthreshold::Real=1e-3)
	basicglm = PoissonGLM(Δt=glm.Δt, 𝐗=glm.𝐗, 𝐲=glm.𝐲)
	maximizelikelihood!(basicglm)
	best𝐠 = fill(NaN,glm.n𝐠)
	best𝐮 = fill(NaN,glm.n𝐮)
	bestℓ = -Inf
	bestπ = NaN
	for s = 1:nstarts
		glm.𝐠 .= basicglm.𝐰[1].*(3 .*rand(2) .- 1)
		glm.𝐮 .= basicglm.𝐰[2:end]
	    glm.π[1] = rand()
	    for i = 1:iterations
	        posteriors!(glm; πthreshold=πthreshold)
	        ∑𝛄 = map(sum, glm.𝛄)
	        glm.π[1] = ∑𝛄[1]/sum(∑𝛄)
			if (glm.π[1] <= πthreshold) || (glm.π[1] >= 1-πthreshold)
				break
			end
	        maximizeECLL!(glm; iterations=iterations)
	    end
		ℓ = loglikelihood(glm)
		if ℓ > bestℓ
			best𝐠 .= glm.𝐠
			best𝐮 .= glm.𝐮
			bestπ = glm.π[1]
			bestℓ = ℓ
		end
	end
	glm.𝐠 .= best𝐠
	glm.𝐮 .= best𝐮
	glm.π[1] = bestπ
	ℓbasic = loglikelihood(basicglm)
	if bestℓ < ℓbasic
		@debug "The in-sample log-likelihood of the mixture-of-gain GLM is lower that of a GLM without any mixture" bestℓ ℓbasic
	end
	return nothing
end

"""
	loglikelihood(glm::GainMixtureGLM)

Log-likelihood of the parameters of a gain mixture GLM
"""
function loglikelihood(glm::GainMixtureGLM)
	@unpack Δt, 𝐠, π, 𝐮, 𝐗, 𝐲 = glm
	𝐋 = collect(𝐗*vcat(g,𝐮) for g in 𝐠)
	ℓ = 0
	for i in eachindex(𝐲)
		p = glm.π[1]*poissonlikelihood(Δt, 𝐋[1][i], 𝐲[i]) + (1-glm.π[1])*poissonlikelihood(Δt, 𝐋[2][i], 𝐲[i])
		ℓ += log(p)
	end
	return ℓ
end

"""
	posteriors!(glm::GainMixtureGLM)

Posterior probability of the gain state
"""
function posteriors!(glm::GainMixtureGLM; πthreshold::Real=1e-3)
	@unpack Δt, 𝐠, 𝛄, 𝐗, π, 𝐮, 𝐲 = glm
	𝐋 = collect(𝐗*vcat(g,𝐮) for g in 𝐠)
	if π[1] < πthreshold
		𝛄[1] .= πthreshold
		𝛄[2] .= 1.0-πthreshold
	elseif π[1] > 1.0-πthreshold
		𝛄[1] .= 1.0-πthreshold
		𝛄[2] .= πthreshold
	else
	    for t = eachindex(𝐲)
	        py_c1 = poissonlikelihood(Δt,𝐋[1][t],𝐲[t])
	        py_c2 = poissonlikelihood(Δt,𝐋[2][t],𝐲[t])
	        pyc1 = π[1]*py_c1
	        pyc2 = (1-π[1])*py_c2
	        py = pyc1+pyc2
	        𝛄[1][t] = pyc1/py
	        𝛄[2][t] = pyc2/py
	    end
	end
    return nothing
end

"""
	maximizeECLL!(glm)

Update weights by maximizing the expectation of the conditional log-likelihood

MODIFIED ARGUMENT
-`glm`: a structure containing the glm of the mixture of gain generalized linear model
"""
function maximizeECLL!(glm::GainMixtureGLM; iterations::Integer=20)
	@unpack Δt, 𝐆, 𝐠, 𝛄, n𝐠, 𝐔, 𝐮, 𝐲 = glm
    f(𝐰) = negativeECLL!(glm, 𝐰)
    ∇f!(∇, 𝐰) = ∇negativeECLL!(∇, glm, 𝐰)
    ∇∇f!(∇∇, 𝐰) = ∇∇negativeECLL!(∇∇, glm, 𝐰)
    results = Optim.optimize(f, ∇f!, ∇∇f!, vcat(𝐠,𝐮), NewtonTrustRegion(), Optim.Options(iterations=iterations))
	𝐰mle = Optim.minimizer(results)
	𝐠 .= 𝐰mle[1:n𝐠]
	𝐮 .= 𝐰mle[n𝐠+1:end]
	return nothing
end

"""
	negativeECLL!(glm,𝐰)

Negative of the expectaton of the conditional log-likelihood
"""
function negativeECLL!(glm::GainMixtureGLM, 𝐰::Vector{<:Real})
	update!(glm, 𝐰)
	return -glm.Q[1]
end

"""
	∇negativeECLL!(∇, glm,𝐰)

Gradient of the negative of the expectaton of the conditional log-likelihood
"""
function ∇negativeECLL!(∇::Vector{<:Real}, glm::GainMixtureGLM, 𝐰::Vector{<:Real})
	update!(glm, 𝐰)
	for i in eachindex(∇)
		∇[i] = -glm.∇Q[i]
	end
	return nothing
end

"""
	∇∇negativeECLL!(∇, glm,𝐰)

Hessian of the negative of the expectaton of the conditional log-likelihood
"""
function ∇∇negativeECLL!(∇∇::Matrix{<:Real}, glm::GainMixtureGLM, 𝐰::Vector{<:Real})
	update!(glm, 𝐰)
	for i in eachindex(∇∇)
		∇∇[i] = -glm.∇∇Q[i]
	end
	return nothing
end

"""
	update!(glm, 𝐰)

update the parameters and glm of the gain mixture GLM
"""
function update!(glm::GainMixtureGLM, 𝐰::Vector{<:Real})
	if (𝐰 != vcat(glm.𝐠, glm.𝐮)) || isnan(glm.Q[1])
		glm.𝐠 .= 𝐰[1:glm.n𝐠]
		glm.𝐮 .= 𝐰[glm.n𝐠+1:end]
		computederivatives!(glm)
	end
end

"""
	computederivatives!(glm)

Compute the expectation of the conditional log-likelihood and its glm

MODIFIED ARGUMENT
-`glm`: a structure containing the glm of the mixture of gain generalized linear model
"""
function computederivatives!(glm::GainMixtureGLM)
	@unpack Δt, 𝐆, 𝐠, 𝛄, ∇Q, ∇∇Q, n𝐠, ntimesteps, n𝐮, Q, 𝐗, 𝐔, 𝐮, 𝐲, γₖdℓ_dLₖ, γₖd²ℓ_dLₖ² = glm
	Q[1] = 0.0
	∇Q .= 0.0
	∇∇Q .= 0.0
	@inbounds for k = 1:n𝐠
		𝐰ₖ = vcat(𝐠[k], 𝐮)
		𝐋ₖ = 𝐗*𝐰ₖ
		𝐥ₖ = 𝐋ₖ
		for t=1:ntimesteps
			λ = inverselink(𝐋ₖ[t])
			d²ℓ_dLₖ², dℓ_dLₖ = differentiate_twice_loglikelihood_wrt_linearpredictor(Δt, 𝐋ₖ[t], λ, 𝐲[t])
			𝐥ₖ[t] = poissonloglikelihood(λ*Δt, 𝐲[t])
			γₖdℓ_dLₖ[k][t] = 𝛄[k][t]*dℓ_dLₖ
			γₖd²ℓ_dLₖ²[k][t] = 𝛄[k][t]*d²ℓ_dLₖ²
		end
		Q[1] += dot(𝛄[k], 𝐥ₖ)
	end
	𝐔ᵀ = transpose(𝐔)
	for k = 1:n𝐠
		∇Q[k] += dot(γₖdℓ_dLₖ[k], 𝐆)
		x = γₖd²ℓ_dLₖ²[k].*𝐆
		∇∇Q[k,k] += dot(𝐆, x)
		∇Q[n𝐠+1:end] .+= 𝐔ᵀ*γₖdℓ_dLₖ[k]
		∇∇Q[k,n𝐠+1:end] .+= 𝐔ᵀ*x
		∇∇Q[n𝐠+1:end,n𝐠+1:end] .+= 𝐔ᵀ*(γₖd²ℓ_dLₖ²[k].*𝐔)
	end
	for i = 1:size(∇∇Q,1)
		for j = i+1:size(∇∇Q,2)
			∇∇Q[j,i] = ∇∇Q[i,j]
		end
	end
	return nothing
end

"""
	negativeECLL(glm)

ForwardDiff-compatible computation of the negative expectation of the conditional log-likelihood

ARGUMENT
-`glm`: a structure containing the glm of the mixture of gain generalized linear model

EXAMPLE
```julia-repl
julia> using FHMDDM, Random, Distributions, LogExpFunctions, ForwardDiff, LinearAlgebra
julia> Random.seed!(1234);
julia> Δt, π, T, n𝐮 = 0.01, 0.75, 1000, 5;
julia> 𝐠 = [100.0, -100.0];
julia> 𝐮 = rand(n𝐮);
julia> 𝐆 = rand(T);
julia> 𝐔 = svd(rand(T,n𝐮)).U.*10
julia> 𝐗 = hcat(𝐆, 𝐔)
julia> 𝐋 = collect(𝐗*vcat(g,𝐮) for g in 𝐠);
julia> 𝛌 = collect(rand() < π ? softplus(𝐋[1][t]) : softplus(𝐋[2][t]) for t=1:T);
julia> 𝐲 = collect(rand(Poisson(λ*Δt)) for λ in 𝛌)
julia> glm = FHMDDM.GainMixtureGLM(Δt=Δt, 𝐠=rand(length(𝐠)), π=rand(1), 𝐮=rand(length(𝐮)), 𝐗=𝐗, 𝐲=𝐲);
julia> 𝐰₀ = vcat(glm.𝐠, glm.𝐮);
julia> FHMDDM.posteriors!(glm)
julia> FHMDDM.computederivatives!(glm)
julia> f(𝐰) = FHMDDM.negativeECLL(glm, 𝐰);
julia> fauto = f(𝐰₀);
julia> gauto = ForwardDiff.gradient(f, 𝐰₀);
julia> hauto = ForwardDiff.hessian(f, 𝐰₀);
julia> absΔQ = abs(fauto - glm.Q[1])
julia> maxabsΔ∇Q = maximum(abs.(gauto .- glm.∇Q))
julia> maxabsΔ∇∇Q = maximum(abs.(hauto .- glm.∇∇Q))
julia> println("   |ΔQ|: ", absΔQ)
julia> println("   max(|Δgradient|): ", maxabsΔ∇Q)
julia> println("   max(|Δhessian|): ", maxabsΔ∇∇Q)
```
"""
function negativeECLL(glm::GainMixtureGLM, 𝐰::Vector{<:Real})
	@unpack Δt, 𝛄, n𝐠, ntimesteps, 𝐗, 𝐲 = glm
	𝐠 = 𝐰[1:n𝐠]
	𝐮 = 𝐰[n𝐠+1:end]
	Q = 0
	@inbounds for k = 1:n𝐠
		𝐰ₖ = vcat(𝐠[k], 𝐮)
		𝐋ₖ = 𝐗*𝐰ₖ
		𝐥ₖ = 𝐋ₖ
		for t=1:ntimesteps
			𝐥ₖ[t] = poissonloglikelihood(Δt, 𝐋ₖ[t], 𝐲[t])
		end
		Q -= dot(𝛄[k], 𝐥ₖ)
	end
	return -Q
end
