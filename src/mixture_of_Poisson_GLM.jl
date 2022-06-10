"""
	GLMθ(options, 𝐗, 𝐕)

Randomly initiate the parameters for a mixture of Poisson generalized linear model

ARGUMENT
-`options`: settings of the model
-`𝐗`: constant input, time-varying inputs from spike history, time-varying inputs from trial events
-`𝐕`: constant and time-varying inputs from the accumulator

OUTPUT
-an instance of `GLMθ`
"""
function GLMθ(options::Options, 𝐗::Matrix{<:AbstractFloat}, 𝐕::Matrix{<:AbstractFloat})
	@unpack K, gain_state_dependent, tuning_state_dependent = options
	n𝐯 =size(𝐕,2)
	n𝐮 = size(𝐗,2)-size(𝐕,2)-1
	𝐮 = 1.0 .- 2.0.*rand(n𝐮)
	if K == 1
		𝐠 = [ones(1)]
		𝐯 = [ones(n𝐯)]
	else
		if gain_state_dependent
			𝐠 = collect(i*ones(1) for i = -1:2/(K-1):1)
		else
			𝐠 = [ones(1)]
		end
		if tuning_state_dependent
			𝐯 = collect(i*ones(n𝐯) for i = -1:2/(K-1):1)
		else
			𝐯 = [ones(n𝐯)]
		end
	end
	GLMθ(𝐠 = 𝐠, 𝐮 = 𝐮, 𝐯 = 𝐯)
end

"""
	GLMθ(glmθ, elementtype)

Create an uninitialized instance of GLMθ with the given element type.

This is for using ForwardDiff

ARGUMENT
-`glmθ`: an instance of GLMθ
-`elementtype`: type of the element in each field of GLMθ

RETURN
-an instance of GLMθ
"""
function GLMθ(glmθ::GLMθ, elementtype)
	GLMθ(𝐠 = collect(zeros(elementtype, length(𝐠)) for 𝐠 in glmθ.𝐠),
		𝐮 = zeros(elementtype, length(glmθ.𝐮)),
		𝐯 = collect(zeros(elementtype, length(𝐯)) for 𝐯 in glmθ.𝐯))
end

"""
    likelihood(mpGLM, j, k)

Conditional likelihood of the spike train, given the index of the state of the accumulator `j` and the state of the coupling `k`, and also by the prior likelihood of the regression weights

UNMODIFIED ARGUMENT
-`mpGLM`: the mixture of Poisson generalized linear model for one neuron
-`j`: state of accumulator variable
-`k`: state of the coupling variable

RETURN
-`𝐩`: a vector by which the conditional likelihood of the spike train and the prior likelihood of the regression weights are multiplied against
"""
function likelihood(mpGLM::MixturePoissonGLM, j::Integer, k::Integer)
    @unpack Δt, 𝐲 = mpGLM
    𝐋 = linearpredictor(mpGLM, j, k)
    𝐩 = 𝐋 # reuse memory
    @inbounds for i=1:length(𝐩)
        𝐩[i] = poissonlikelihood(Δt, 𝐋[i], 𝐲[i])
    end
    return 𝐩
end

"""
    likelihood!(𝐩, mpGLM, j, k)

In-place multiplication of `𝐩` by the conditional likelihood of the spike train, given the index of the state of the accumulator `j` and the state of the coupling `k`, and also by the prior likelihood of the regression weights

MODIFIED ARGUMENT
-`𝐩`: a vector by which the conditional likelihood of the spike train and the prior likelihood of the regression weights are multiplied against

UNMODIFIED ARGUMENT
-`mpGLM`: the mixture of Poisson generalized linear model for one neuron
-`j`: state of accumulator variable
-`k`: state of the coupling variable

RETURN
-`nothing`
"""
function likelihood!(𝐩::Vector{<:Real}, mpGLM::MixturePoissonGLM, j::Integer, k::Integer)
    @unpack Δt, 𝐲 = mpGLM
    𝐋 = linearpredictor(mpGLM, j, k)
    @inbounds for i=1:length(𝐩)
		𝐩[i] *= poissonlikelihood(Δt, 𝐋[i], 𝐲[i])
    end
    return nothing
end

"""
	Poissonlikelihood(λΔt, L, y)

Probability of a Poisson observation

ARGUMENT
-`λΔt`: the expected value
-`y`: the observation
-`y!`: the factorial of the observation

OUTPUT
-the likelihood
"""
function poissonlikelihood(Δt::Real, L::Real, y::Integer)
	λΔt = softplus(L)*Δt
	poissonlikelihood(λΔt, y)
end

"""
	poissonlikelihood(λΔt, y)

Likelihood of observation `y` given intensity `λΔt`
"""
function poissonlikelihood(λΔt::Real, y::Integer)
	if y==0
		exp(-λΔt)
	elseif y==1
		λΔt*exp(-λΔt)
	else
		λΔt^y * exp(-λΔt) / factorial(y)
	end
end

"""
    linearpredictor(mpGLM, j, k)

Linear combination of the weights in the j-th accumulator state and k-th coupling state

ARGUMENT
-`mpGLM`: the mixture of Poisson generalized linear model of one neuron
-`j`: state of the accumulator variable
-`k`: state of the coupling variable

RETURN
-`𝐋`: a vector whose element 𝐋[t] corresponds to the t-th time bin in the trialset
"""
function linearpredictor(mpGLM::MixturePoissonGLM, j::Integer, k::Integer)
    @unpack 𝐗, d𝛏_dB = mpGLM
    @unpack 𝐠, 𝐮, 𝐯 = mpGLM.θ
	𝐠ₖ = 𝐠[min(length(𝐠), k)]
	𝐯ₖ = 𝐯[min(length(𝐯), k)]
	𝐗*vcat(𝐠ₖ, 𝐮, 𝐯ₖ.*d𝛏_dB[j])
end

"""
    poissonloglikelihood

Differentiate the log-likelihood of a Poisson GLM with respect to the linear predictor

The Poisson GLM is assumed to have a a softplus nonlinearity

ARGUMENT
-`Δt`: duration of time step
-`L`: linear predictor
-`y`: observation

RETURN
-log-likelihood
"""
function poissonloglikelihood(Δt::AbstractFloat, L::Real, y::Integer)
    λΔt = softplus(L)*Δt
	if y == 0
		-λΔt
	elseif y == 1
		log(λΔt) - λΔt
	else
		y*log(λΔt) - λΔt
	end
end

"""
    differentiate_loglikelihood_twice_wrt_linearpredictor

First derivative of the log-likelihood of a Poisson GLM with respect to the linear predictor

The Poisson GLM is assumed to have a a softplus nonlinearity

ARGUMENT
-`Δt`: duration of time step
-`L`: linear predictor
-`y`: observation

RETURN
-first derivative of the log-likelihood with respect to the linear predictor
"""
function differentiate_loglikelihood_wrt_linearpredictor(Δt::AbstractFloat, L::Real, y::Integer)
    f₁ = logistic(L)
	if y > 0
        if L > -100.0
			f₀ = softplus(L)
            f₁*(y/f₀ - Δt)
        else
            y - f₁*Δt # the limit of logistic(x)/softplus(x) as x goes to -∞ is 1
        end
    else
        -f₁*Δt
    end
end

"""
    differentiate_loglikelihood_twice_wrt_linearpredictor

Second derivative of the log-likelihood of a Poisson GLM with respect to the linear predictor

The Poisson GLM is assumed to have a a softplus nonlinearity

ARGUMENT
-`Δt`: duration of time step
-`L`: linear predictor
-`y`: observation

RETURN
-second derivative of the log-likelihood with respect to the linear predictor
-first derivative of the log-likelihood with respect to the linear predictor
-log-likelihood
"""
function differentiate_loglikelihood_twice_wrt_linearpredictor(Δt::AbstractFloat, L::Real, y::Integer)
	f₀ = softplus(L)
    f₁ = logistic(L)
    f₂ = f₁*(1.0-f₁)
	λΔt = f₀*Δt
	if y == 0
		ℓ = -λΔt
	elseif y == 1
		ℓ = log(λΔt) - λΔt
	else
		ℓ = y*log(λΔt) - λΔt
	end
	if y > 0
        if L > -100.0
            dℓ_dL= f₁*(y/f₀ - Δt)
        else
            dℓ_dL = y - f₁*Δt # the limit of logistic(x)/softplus(x) as x goes to -∞ is 1
        end
    else
        dℓ_dL = -f₁*Δt
    end
    if y > 0 && L > -50.0
        d²ℓ_dL² = y*(f₀*f₂ - f₁^2)/f₀^2 - f₂*Δt # the limit of the second term is 0 as xw goes to -∞
    else
        d²ℓ_dL² = -f₂*Δt
    end
	return d²ℓ_dL², dℓ_dL, ℓ
end

"""
	maximize_expectation_of_loglikelihood!(mpGLM, γ)

Learn the filters of a Poisson mixture GLM by maximizing the expectation of the log-likelihood

MODIFIED ARGUMENT
-`mpGLM`: a structure containing the data and parameters of the mixture of Poisson GLM of one neuron

UNMODIFIED ARGUMENT
-`γ`: posterior probability of the latent variables. Element `γ[j][τ]` corresponds to the posterior probability of the j-th accumulator state  in the τ-th time step

EXAMPLE
```julia-repl
julia> using FHMDDM
julia> model = Model("/mnt/cup/labs/brody/tzluo/analysis_data/analysis_2022_05_05_test/data.mat");
julia> maximize_choice_posterior!(model)
julia> γ = choiceposteriors(model)[1]
julia> mpGLM = model.trialsets[1].mpGLMs[2]
julia> FHMDDM.maximize_expectation_of_loglikelihood!(mpGLM, γ)
```
"""
function maximize_expectation_of_loglikelihood!(mpGLM::MixturePoissonGLM, γ::Matrix{<:Vector{<:Real}}; show_trace::Bool=false, iterations::Integer=20)
	x₀ = concatenateparameters(mpGLM.θ)
	nparameters = length(x₀)
	Q = fill(NaN,1)
	∇Q = fill(NaN, nparameters)
	∇∇Q = fill(NaN, nparameters, nparameters)
	f(x) = -expectation_of_loglikelihood!(mpGLM,Q,∇Q,∇∇Q,γ,x)
	∇f!(∇, x) = negexpectation_of_∇loglikelihood!(∇,mpGLM,Q,∇Q,∇∇Q,γ,x)
	∇∇f!(∇∇, x) = negexpectation_of_∇∇loglikelihood!(∇∇,mpGLM,Q,∇Q,∇∇Q,γ,x)
    results = Optim.optimize(f, ∇f!, ∇∇f!, x₀, NewtonTrustRegion(), Optim.Options(show_trace=show_trace, iterations=iterations))
	sortparameters!(mpGLM.θ, Optim.minimizer(results))
	return nothing
end

"""
	expectation_of_loglikelihood!(mpGLM,Q,∇Q,∇∇Q,γ,x)

Expectation of the log-likelihood under the posterior probability of the latent variables

MODIFIED ARGUMENT
-`mpGLM`: a structure containing the data and parameters of the mixture of Poisson GLM of one neuron
-`Q`: an one-element vector that quantifies the expectation
-`∇Q`: gradient of the expectation with respect to the filters in the k-th state
-`∇∇Q`: Hessian of the expectation with respect to the filters in the k-th state

UNMODIFIED ARGUMENT
-`γ`: posterior probability of the latent variables. Element `γ[j][τ]` corresponds to the posterior probability of the j-th accumulator state  in the τ-th time step
-`x`: filters
"""
function expectation_of_loglikelihood!(mpGLM::MixturePoissonGLM, Q::Vector{<:Real}, ∇Q::Vector{<:Real}, ∇∇Q::Matrix{<:Real}, γ::Matrix{<:Vector{<:Real}}, x::Vector{<:Real})
	if (x != concatenateparameters(mpGLM.θ)[1]) || isnan(Q[1])
		sortparameters!(mpGLM.θ, x)
		expectation_of_∇∇loglikelihood!(Q,∇Q,∇∇Q,γ,mpGLM)
	end
	Q[1]
end

"""
	negexpectation_of_∇loglikelihood!(∇,mpGLM,Q,∇Q,∇∇Q,γ,x)

Gradient of the negative of the expectation of the log-likelihood under the posterior probability of the latent variables

MODIFIED ARGUMENT
-`∇`: gradient of the negative of the expectation
-`mpGLM`: a structure containing the data and parameters of the mixture of Poisson GLM of one neuron
-`Q`: an one-element vector that quantifies the expectation
-`∇Q`: gradient of the expectation with respect to the filters in the k-th state
-`∇∇Q`: Hessian of the expectation with respect to the filters in the k-th state

UNMODIFIED ARGUMENT
-`γ`: posterior probability of the latent variables. Element `γ[j][τ]` corresponds to the posterior probability of the j-th accumulator state  in the τ-th time step
-`x`: filters
"""
function negexpectation_of_∇loglikelihood!(∇::Vector{<:Real}, mpGLM::MixturePoissonGLM, Q::Vector{<:Real}, ∇Q::Vector{<:Real}, ∇∇Q::Matrix{<:Real}, γ::Matrix{<:Vector{<:Real}}, x::Vector{<:Real})
	if (x != concatenateparameters(mpGLM.θ)[1]) || isnan(Q[1])
		sortparameters!(mpGLM.θ, x)
		expectation_of_∇∇loglikelihood!(Q,∇Q,∇∇Q,γ,mpGLM)
	end
	for i in eachindex(∇)
		∇[i] = -∇Q[i]
	end
	return nothing
end

"""
	negexpectation_of_∇∇loglikelihood!(∇∇,mpGLM,Q,∇Q,∇∇Q,γ,x)

Hessian of the negative of the expectation of the log-likelihood under the posterior probability of the latent variables

MODIFIED ARGUMENT
-`∇∇`: Hessian of the negative of the expectation
-`mpGLM`: a structure containing the data and parameters of the mixture of Poisson GLM of one neuron
-`Q`: an one-element vector that quantifies the expectation
-`∇Q`: gradient of the expectation with respect to the filters in the k-th state
-`∇∇Q`: Hessian of the expectation with respect to the filters in the k-th state

UNMODIFIED ARGUMENT
-`γ`: posterior probability of the latent variables. Element `γ[j][τ]` corresponds to the posterior probability of the j-th accumulator state  in the τ-th time step
-`x`: filters
"""
function negexpectation_of_∇∇loglikelihood!(∇∇::Matrix{<:Real}, mpGLM::MixturePoissonGLM, Q::Vector{<:Real}, ∇Q::Vector{<:Real}, ∇∇Q::Matrix{<:Real}, γ::Matrix{<:Vector{<:Real}}, x::Vector{<:Real})
	if (x != concatenateparameters(mpGLM.θ)[1]) || isnan(Q[1])
		sortparameters!(mpGLM.θ, x)
		expectation_of_∇∇loglikelihood!(Q,∇Q,∇∇Q,γ,mpGLM)
	end
	nparameters = length(x)
	for i =1:nparameters
		for j=i:nparameters
			∇∇[i,j] = ∇∇[j,i] = -∇∇Q[i,j]
		end
	end
	return nothing
end

"""
	expectation_of_∇∇loglikelihood!(Q,∇Q,∇∇Q,γ,k,mpGLM)

Compute the expectation of the log-likelihood and its gradient and Hessian

ARGUMENT
-`Q`: expectation of the log-likelihood under the posterior probability of the latent variables. Only the component in the coupling state `k` is included
-`∇Q`: first-order derivatives of the expectation
-`∇∇Q`: second-order derivatives of the expectation

UNMODIFIED ARGUMENT
-`γ`: posterior probabilities of the latent variables
-`k`: index of the coupling state
-`mpGLM`: a structure containing the data and parameters of the mixture of Poisson GLM of one neuron

EXAMPLE
```julia-repl
julia> using FHMDDM, ForwardDiff, Random
julia> model = Model("/mnt/cup/labs/brody/tzluo/analysis_data/analysis_2022_05_05_test/data.mat");
julia> mpGLM = model.trialsets[1].mpGLMs[1]
julia> γ = FHMDDM.randomposterior(mpGLM; rng=MersenneTwister(1234))
julia> x₀ = concatenateparameters(mpGLM.θ)
julia> nparameters = length(x₀)
julia> fhand, ghand, hhand = fill(NaN,1), fill(NaN,nparameters), fill(NaN,nparameters,nparameters)
julia> FHMDDM.expectation_of_∇∇loglikelihood!(fhand, ghand, hhand, γ, mpGLM)
julia> f(x) = FHMDDM.expectation_of_loglikelihood(γ, mpGLM, x)
julia> fauto = f(x₀)
julia> gauto = ForwardDiff.gradient(f, x₀)
julia> hauto = ForwardDiff.hessian(f, x₀)
julia> abs(fauto - fhand[1])
julia> maximum(abs.(gauto .- ghand))
julia> maximum(abs.(hauto .- hhand))
```
"""
function expectation_of_∇∇loglikelihood!(Q::Vector{<:Real},
										∇Q::Vector{<:Real},
										∇∇Q::Matrix{<:Real},
										γ::Matrix{<:Vector{<:Real}},
										mpGLM::MixturePoissonGLM)
    @unpack Δt, 𝐕, 𝐗, d𝛏_dB, 𝐲 = mpGLM
	@unpack 𝐠, 𝐮, 𝐯 = mpGLM.θ
	d𝛏_dB² = d𝛏_dB.^2
	Ξ,K = size(γ)
	T = length(𝐲)
	∑ᵢ_dQᵢₖ_dLᵢₖ = collect(zeros(T) for k=1:K)
	∑ᵢ_dQᵢₖ_dLᵢₖ⨀dξᵢ_dB = collect(zeros(T) for k=1:K)
	∑ᵢ_d²Qᵢₖ_dLᵢₖ² = collect(zeros(T) for k=1:K)
	∑ᵢ_d²Qᵢₖ_dLᵢₖ²⨀dξᵢ_dB = collect(zeros(T) for k=1:K)
	∑ᵢ_d²Qᵢₖ_dLᵢₖ²⨀dξᵢ_dB² = collect(zeros(T) for k=1:K)
	Q[1] = 0.0
	∇Q .= 0.0
	∇∇Q .= 0.0
	@inbounds for i = 1:Ξ
		for k = 1:K
			𝐋 = FHMDDM.linearpredictor(mpGLM,i,k)
			for t=1:T
				d²ℓ_dL², dℓ_dL, ℓ = FHMDDM.differentiate_loglikelihood_twice_wrt_linearpredictor(Δt, 𝐋[t], 𝐲[t])
				Q[1] += γ[i,k][t]*ℓ
				dQᵢₖ_dLᵢₖ = γ[i,k][t] * dℓ_dL
				∑ᵢ_dQᵢₖ_dLᵢₖ[k][t] += dQᵢₖ_dLᵢₖ
				∑ᵢ_dQᵢₖ_dLᵢₖ⨀dξᵢ_dB[k][t] += dQᵢₖ_dLᵢₖ*d𝛏_dB[i]
				d²Qᵢₖ_dLᵢₖ² = γ[i,k][t] * d²ℓ_dL²
				∑ᵢ_d²Qᵢₖ_dLᵢₖ²[k][t] += d²Qᵢₖ_dLᵢₖ²
				∑ᵢ_d²Qᵢₖ_dLᵢₖ²⨀dξᵢ_dB[k][t] += d²Qᵢₖ_dLᵢₖ²*d𝛏_dB[i]
				∑ᵢ_d²Qᵢₖ_dLᵢₖ²⨀dξᵢ_dB²[k][t] += d²Qᵢₖ_dLᵢₖ²*d𝛏_dB²[i]
			end
		end
	end
	K𝐠 = length(𝐠)
	K𝐯 = length(𝐯)
	n𝐠 = length(𝐠[1])
	n𝐮 = length(𝐮)
	n𝐯 = length(𝐯[1])
	indices𝐠 = collect(((k-1)*n𝐠+1 : k*n𝐠) for k = 1:K𝐠)
	indices𝐮 = indices𝐠[end][end] .+ (1:n𝐮)
	indices𝐯 = collect(indices𝐮[end] .+ ((k-1)*n𝐯+1 : k*n𝐯) for k = 1:K𝐯)
	𝐔 = @view 𝐗[:, n𝐠+1:n𝐠+n𝐮]
	𝐔ᵀ, 𝐕ᵀ = transpose(𝐔), transpose(𝐕)
	∑ᵢₖ_dQᵢₖ_dLᵢₖ = sum(∑ᵢ_dQᵢₖ_dLᵢₖ)
	∑ᵢₖ_d²Qᵢₖ_dLᵢₖ² = sum(∑ᵢ_d²Qᵢₖ_dLᵢₖ²)
	∇Q[indices𝐮] .= 𝐔ᵀ*∑ᵢₖ_dQᵢₖ_dLᵢₖ
	∇∇Q[indices𝐮, indices𝐮] .= 𝐔ᵀ*(∑ᵢₖ_d²Qᵢₖ_dLᵢₖ².*𝐔)
	if K𝐠 == K
		@inbounds for k = 1:K
			∇Q[indices𝐠[k]] .= sum(∑ᵢ_dQᵢₖ_dLᵢₖ[k])
			∇∇Q[indices𝐠[k], indices𝐠[k]] .= sum(∑ᵢ_d²Qᵢₖ_dLᵢₖ²[k])
			∇∇Q[indices𝐠[k], indices𝐮] .= transpose(∑ᵢ_d²Qᵢₖ_dLᵢₖ²[k])*𝐔
		end
	else
		∇Q[indices𝐠[1]] .= sum(∑ᵢₖ_dQᵢₖ_dLᵢₖ)
		∇∇Q[indices𝐠[1], indices𝐠[1]] .= sum(∑ᵢₖ_d²Qᵢₖ_dLᵢₖ²)
		∇∇Q[indices𝐠[1], indices𝐮] .= transpose(∑ᵢₖ_d²Qᵢₖ_dLᵢₖ²)*𝐔
	end
	if K𝐯 == K
		@inbounds for k = 1:K
			∇Q[indices𝐯[k]] .= 𝐕ᵀ*∑ᵢ_dQᵢₖ_dLᵢₖ⨀dξᵢ_dB[k]
			∇∇Q[indices𝐯[k], indices𝐯[k]] .= 𝐕ᵀ*(∑ᵢ_d²Qᵢₖ_dLᵢₖ²⨀dξᵢ_dB²[k].*𝐕)
			∇∇Q[indices𝐮, indices𝐯[k]] .= 𝐔ᵀ*(∑ᵢ_d²Qᵢₖ_dLᵢₖ²⨀dξᵢ_dB[k].*𝐕)
		end
	else
		∇Q[indices𝐯[1]] .= 𝐕ᵀ*sum(∑ᵢ_dQᵢₖ_dLᵢₖ⨀dξᵢ_dB)
		∇∇Q[indices𝐯[1], indices𝐯[1]] .= 𝐕ᵀ*(sum(∑ᵢ_d²Qᵢₖ_dLᵢₖ²⨀dξᵢ_dB²).*𝐕)
		∇∇Q[indices𝐮, indices𝐯[1]] .= 𝐔ᵀ*(sum(∑ᵢ_d²Qᵢₖ_dLᵢₖ²⨀dξᵢ_dB).*𝐕)
	end
	@inbounds for k = 1:K
		∇∇Q[indices𝐠[min(length(𝐠), k)], indices𝐯[min(length(𝐯), k)]] .= transpose(∑ᵢ_d²Qᵢₖ_dLᵢₖ²⨀dξᵢ_dB[k])*𝐕
	end
	for i = 1:size(∇∇Q,1)
		for j = i+1:size(∇∇Q,2)
			∇∇Q[j,i] = ∇∇Q[i,j]
		end
	end
	return nothing
end

"""
    expectation_of_loglikelihood(γ, mpGLM, x)

ForwardDiff-compatible computation of the expectation of the log-likelihood of the mixture of Poisson generalized model of one neuron

Ignores the log(y!) term, which does not depend on the parameters

ARGUMENT
-`γ`: posterior probability of the latent variable
-`mpGLM`: the GLM of one neuron
-`x`: weights of the linear filters of the GLM concatenated as a vector of floating-point numbers

RETURN
-expectation of the log-likelihood of the spike train of one neuron
"""
function expectation_of_loglikelihood(γ::Matrix{<:Vector{<:AbstractFloat}},
									   mpGLM::MixturePoissonGLM,
									   x::Vector{<:Real})
	mpGLM = MixturePoissonGLM(x, mpGLM)
    @unpack Δt, 𝐲 = mpGLM
    T = length(𝐲)
    Ξ,K = size(γ)
    Q = 0.0
    @inbounds for i = 1:Ξ
	    for k = 1:K
			𝐋 = linearpredictor(mpGLM,i,k)
            for t = 1:T
				Q += γ[i,k][t]*poissonloglikelihood(Δt, 𝐋[t], 𝐲[t])
            end
        end
    end
    return Q
end
