"""
	GLMθ(options, 𝐮indices_hist, 𝐮indices_move, 𝐮indices_time, 𝐕)

Randomly initiate the parameters for a mixture of Poisson generalized linear model

ARGUMENT
-`options`: settings of the model
-`𝐮indices_hist`: indices in 𝐮 corresponding to the temporal basis functions of the postspike filter
-`𝐮indices_move`: indices in 𝐮 corresponding to the temporal basis functions of the premovement filter
-`𝐮indices_time`: indices in 𝐮 corresponding to the temporal basis functions of the time-in-trial filter
-`𝐕`: constant and time-varying inputs from the accumulator

OUTPUT
-an instance of `GLMθ`
"""
function GLMθ(options::Options, 𝐮indices_hist::UnitRange{<:Integer}, 𝐮indices_move::UnitRange{<:Integer}, 𝐮indices_time::UnitRange{<:Integer}, 𝐕::Matrix{<:AbstractFloat})
	n𝐮 = 𝐮indices_move[end]
	n𝐯 =size(𝐕,2)
	K𝐠 = options.gain_state_dependent ? options.K : 1
	K𝐯 = options.tuning_state_dependent ? options.K : 1
	θ = GLMθ(b = fill(NaN,1),
			b_scalefactor = options.b_scalefactor,
			fit_b = options.fit_b,
			𝐠 = fill(NaN, K𝐠),
			𝐮 = fill(NaN, n𝐮),
			𝐮indices_hist=𝐮indices_hist,
			𝐮indices_move=𝐮indices_move,
			𝐮indices_time=𝐮indices_time,
			𝐯 = collect(fill(NaN,n𝐯) for k=1:K𝐯))
	randomizeparameters!(θ)
	return θ
end

"""
	randomizeparameters!(θ)

Randomly initialize parameters of a mixture of Poisson GLM
"""
function randomizeparameters!(θ::GLMθ)
	θ.b[1] = 0.0
	for i in eachindex(θ.𝐮)
		θ.𝐮[i] = 1.0 .- 2rand()
	end
	θ.𝐠[1] = 0.0
	for k = 2:length(θ.𝐠)
		θ.𝐠[k] = 1.0 .- 2rand()
	end
	if length(θ.𝐯) > 1
		K = length(θ.𝐯)
		𝐯₀ = -1:2/(K-1):1
		for k = 1:K
			θ.𝐯[k] .= 𝐯₀[k]
		end
	else
		θ.𝐯[1] .= 1.0
	end
end

"""
	GLMθ(glmθ, elementtype)

Create an uninitialized instance of `GLMθ` with the given element type.

This is for using ForwardDiff

ARGUMENT
-`glmθ`: an instance of GLMθ
-`elementtype`: type of the element in each field of GLMθ

RETURN
-an instance of GLMθ
"""
function GLMθ(glmθ::GLMθ, elementtype)
	GLMθ(b = zeros(elementtype, length(glmθ.b)),
		b_scalefactor = glmθ.b_scalefactor,
		fit_b = glmθ.fit_b,
		𝐠 = zeros(elementtype, length(glmθ.𝐠)),
		𝐮 = zeros(elementtype, length(glmθ.𝐮)),
		𝐯 = collect(zeros(elementtype, length(𝐯)) for 𝐯 in glmθ.𝐯),
		𝐮indices_hist = glmθ.𝐮indices_hist,
		𝐮indices_time = glmθ.𝐮indices_time,
		𝐮indices_move = glmθ.𝐮indices_move)
end

"""
	FHMDDM.copy(glmθ)

Make a copy of a structure containing the parameters of a mixture of Poisson GLM
"""
function FHMDDM.copy(glmθ::GLMθ)
	GLMθ(b = copy(glmθ.b),
		b_scalefactor = glmθ.b_scalefactor,
		fit_b = glmθ.fit_b,
		𝐠 = copy(glmθ.𝐠),
		𝐮 = copy(glmθ.𝐮),
		𝐯 = collect(copy(𝐯ₖ) for 𝐯ₖ in glmθ.𝐯),
		𝐮indices_hist = copy(glmθ.𝐮indices_hist),
		𝐮indices_time = copy(glmθ.𝐮indices_time),
		𝐮indices_move = copy(glmθ.𝐮indices_move))
end

"""
	initialize(glmθ)

Create an uninitialized instance of `GLMθ`
"""
initialize(glmθ::GLMθ) = GLMθ(glmθ, eltype(glmθ.𝐮))

"""
	MixturePoissonGLM(concatenatedθ, glmθindex, mpGLM)

Create a structure for a mixture of Poisson GLM with updated parameters

ARGUMENT
-`concatenatedθ`: a vector of new parameter values
-`glmθindex`: index of each parameter in the vector of values
-`mpGLM`: a structure containing information on the mixture of Poisson GLM for one neuron

OUTPUT
-a new structure for the mixture of Poisson GLM of a neuron with new parameter values
"""
function MixturePoissonGLM(concatenatedθ::Vector{T},
						   mpGLM::MixturePoissonGLM;
						   offset::Integer=0,
						   omitb::Bool=false) where {T<:Real}
	mpGLM = MixturePoissonGLM(Δt=mpGLM.Δt,
							d𝛏_dB=mpGLM.d𝛏_dB,
							Φₐ=mpGLM.Φₐ,
                        	Φₕ=mpGLM.Φₕ,
							Φₘ=mpGLM.Φₘ,
							Φₜ=mpGLM.Φₜ,
							θ=GLMθ(mpGLM.θ, concatenatedθ; offset=offset, omitb=omitb),
							𝐕=mpGLM.𝐕,
							𝐗=mpGLM.𝐗,
							𝐲=mpGLM.𝐲)
	return mpGLM
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
    @unpack b, b_scalefactor, 𝐠, 𝐮, 𝐯 = mpGLM.θ
	gₖ = 𝐠[min(length(𝐠), k)]
	𝐯ₖ = 𝐯[min(length(𝐯), k)]
	transformedξ = transformaccumulator(b[1]*b_scalefactor, d𝛏_dB[j])
	𝐗*vcat(gₖ, 𝐮, 𝐯ₖ.*transformedξ)
end

"""
	poissonloglikelihood(λΔt, y)

Log-likelihood of an observation under a Poisson GLM

ARGUMENT
-`λΔt`: Poisson intensity per second
-`y`: observation

RETURN
-log-likelihood
"""
function poissonloglikelihood(λΔt::Real, y::Integer)
	if y == 0
		-λΔt
	elseif y == 1
		log(λΔt) - λΔt
	else
		y*log(λΔt) - λΔt
	end
end

"""
    poissonloglikelihood

Log-likelihood of an observation under a Poisson GLM with a softplus nonlinearity

ARGUMENT
-`Δt`: duration of time step
-`L`: linear predictor
-`y`: observation

RETURN
-log-likelihood
"""
poissonloglikelihood(Δt::AbstractFloat, L::Real, y::Integer) = poissonloglikelihood(softplus(L)*Δt, y)

"""
    differentiate_loglikelihood_wrt_linearpredictor

Differentiate the log-likelihood of a Poisson GLM with respect to the linear predictor

The Poisson GLM is assumed to have a a softplus nonlinearity

ARGUMENT
-`Δt`: duration of time step
-`L`: linear predictor at one time step
-`λ`: Poisson rate
-`y`: observation at that time step

RETURN
-the first derivative with respect to the linear predictor

EXAMPLE
```julia-repl
julia> using FHMDDM, ForwardDiff, LogExpFunctions
julia> Δt = 0.01
julia> y = 2
julia> f(x) = let λΔt = softplus(x[1])*Δt; y*log(λΔt)-λΔt+log(factorial(y)); end
julia> x = rand(1)
julia> d1auto = ForwardDiff.gradient(f, x)
julia> d1hand = FHMDDM.differentiate_loglikelihood_wrt_linearpredictor(Δt, x[1], softplus(x[1]), y)
julia> abs(d1hand - d1auto[1])
```
"""
function differentiate_loglikelihood_wrt_linearpredictor(Δt::AbstractFloat, L::Real, λ::Real, y::Integer)
	dλ_dL = logistic(L)
    if y > 0
        if L > -100.0
            dℓ_dL = dλ_dL*(y/λ - Δt)
        else
            dℓ_dL = y - dλ_dL*Δt  # the limit of `dλ_dL/λ` as x goes to -∞ is 1
        end
    else
        dℓ_dL = -dλ_dL*Δt
    end
end

"""
    differentiate_loglikelihood_wrt_linearpredictor(Δt, L, y)

First derivative of the log-likelihood of a Poisson GLM with respect to the linear predictor
"""
differentiate_loglikelihood_wrt_linearpredictor(Δt::AbstractFloat, L::Real, y::Integer) = differentiate_loglikelihood_wrt_linearpredictor(Δt, L, softplus(L), y)

"""
    differentiate_twice_loglikelihood_wrt_linearpredictor

Second derivative of the log-likelihood of a Poisson GLM with respect to the linear predictor

The Poisson GLM is assumed to have a a softplus nonlinearity

ARGUMENT
-`Δt`: duration of time step
-`L`: linear predictor at one time step
-`λ`: Poisson rate
-`y`: observation at that time step

RETURN
-the first derivative with respect to the linear predictor
-the second derivative with respect to the linear predictor

EXAMPLE
```julia-repl
julia> using FHMDDM, ForwardDiff, LogExpFunctions
julia> Δt = 0.01
julia> y = 3
julia> f(x) = FHMDDM.poissonloglikelihood(Δt, x, y)
julia> g(x) = ForwardDiff.derivative(f, x)
julia> h(x) = ForwardDiff.derivative(g, x)
julia> x₀ = 1-2rand()
julia> d1auto = g(x₀)
julia> d2auto = h(x₀)
julia> d2hand, d1hand = FHMDDM.differentiate_twice_loglikelihood_wrt_linearpredictor(Δt, x₀, softplus(x₀), y)
julia> abs(d1hand - d1auto[1])
julia> abs(d2hand - d2auto[1])
```
"""
function differentiate_twice_loglikelihood_wrt_linearpredictor(Δt::AbstractFloat, L::Real, λ::Real, y::Integer)
	dλ_dL = logistic(L)
	d²λ_dLdL = dλ_dL*(1-dλ_dL)
    if y > 0
        if L > -100.0
            dℓ_dL = dλ_dL*(y/λ - Δt)
        else
            dℓ_dL = y - dλ_dL*Δt  # the limit of `dλ_dL/λ` as x goes to -∞ is 1
        end
		if L > -50.0
			d²ℓ_dLdL = y*(λ*d²λ_dLdL - dλ_dL^2)/λ^2 - d²λ_dLdL*Δt # the limit of first second term is 0 as L goes to -∞
		else
			d²ℓ_dLdL = -d²λ_dLdL*Δt
		end
    else
        dℓ_dL = -dλ_dL*Δt
		d²ℓ_dLdL = -d²λ_dLdL*Δt
    end
	return d²ℓ_dLdL, dℓ_dL
end

"""
	differentiate_twice_loglikelihood_wrt_linearpredictor(Δt, L, y)

Second derivative of the log-likelihood of a Poisson GLM with respect to the linear predictor

ARGUMENT
-`Δt`: duration of time step
-`L`: linear predictor at one time step
-`y`: observation at that time step

RETURN
-the second derivative with respect to the linear predictor
-the first derivative with respect to the linear predictor
-the log-likelihood
"""
function differentiate_twice_loglikelihood_wrt_linearpredictor(Δt::AbstractFloat, L::Real, y::Integer)
	λ = softplus(L)
	λΔt = λ*Δt
	ℓ = poissonloglikelihood(λΔt, y)
	d²ℓ_dL², dℓ_dL = differentiate_twice_loglikelihood_wrt_linearpredictor(Δt, L, λ, y)
	return d²ℓ_dL², dℓ_dL, ℓ
end

"""
	expectation_∇loglikelihood!(∇Q, γ, mpGLM)

Expectation under the posterior probability of the gradient of the log-likelihood.

This function is used for computing the gradient of the log-likelihood of the entire model

MODIFIED ARGUMENT
-`∇`: The gradient

UNMODIFIED ARGUMENT
-`γ`: Joint posterior probability of the accumulator and coupling variable. γ[i,k][t] corresponds to the i-th accumulator state and the k-th coupling state in the t-th time bin in the trialset.
-`mpGLM`: structure containing information for the mixture of Poisson GLM for one neuron

EXAMPLE
```julia-rep
julia> using FHMDDM, Random
julia> model = Model("/mnt/cup/labs/brody/tzluo/analysis_data/analysis_2022_05_05_test/data.mat"; randomize=true);
julia> mpGLM = model.trialsets[1].mpGLMs[1]
julia> γ = FHMDDM.randomposterior(mpGLM; rng=MersenneTwister(1234))
julia> ∇Q = FHMDDM.GLMθ(mpGLM.θ, eltype(mpGLM.θ.𝐮))
julia> FHMDDM.expectation_∇loglikelihood!(∇Q, γ, mpGLM)
julia> ghand = FHMDDM.concatenateparameters(∇Q)[1]
julia> using ForwardDiff
julia> concatenatedθ = FHMDDM.concatenateparameters(mpGLM.θ)[1]
julia> f(x) = FHMDDM.expectation_loglikelihood(x, γ, mpGLM)
julia> gauto = ForwardDiff.gradient(f, concatenatedθ)
julia> maximum(abs.(gauto .- ghand))
```
"""
function expectation_∇loglikelihood!(∇Q::GLMθ, γ::Matrix{<:Vector{<:Real}}, mpGLM::MixturePoissonGLM)
	@unpack Δt, 𝐕, 𝐗, 𝐲 = mpGLM
	@unpack 𝐯 = mpGLM.θ
	𝛚 = transformaccumulator(mpGLM)
	d𝛚_db = dtransformaccumulator(mpGLM)
	Ξ, K = size(γ)
	T = length(𝐲)
	∑ᵢ_dQᵢₖ_dLᵢₖ = collect(zeros(T) for k=1:K)
	∑ᵢ_dQᵢₖ_dLᵢₖ⨀ωᵢ = collect(zeros(T) for k=1:K)
	if ∇Q.fit_b
		∑ᵢ_dQᵢₖ_dLᵢₖ⨀dωᵢ_db = collect(zeros(T) for k=1:K)
	end
	@inbounds for i = 1:Ξ
		for k = 1:K
			𝐋 = linearpredictor(mpGLM,i,k)
			for t=1:T
				dQᵢₖ_dLᵢₖ = γ[i,k][t] * differentiate_loglikelihood_wrt_linearpredictor(Δt, 𝐋[t], 𝐲[t])
				∑ᵢ_dQᵢₖ_dLᵢₖ[k][t] += dQᵢₖ_dLᵢₖ
				∑ᵢ_dQᵢₖ_dLᵢₖ⨀ωᵢ[k][t] += dQᵢₖ_dLᵢₖ*𝛚[i]
				if ∇Q.fit_b
					∑ᵢ_dQᵢₖ_dLᵢₖ⨀dωᵢ_db[k][t] += dQᵢₖ_dLᵢₖ*d𝛚_db[i]
				end
			end
		end
	end
	𝐔 = @view 𝐗[:, 2:1+length(∇Q.𝐮)]
	∑ᵢₖ_dQᵢₖ_dLᵢₖ = sum(∑ᵢ_dQᵢₖ_dLᵢₖ)
    ∇Q.𝐮 .= 𝐔' * ∑ᵢₖ_dQᵢₖ_dLᵢₖ
	@inbounds for k = 2:length(∇Q.𝐠)
		∇Q.𝐠[k] = sum(∑ᵢ_dQᵢₖ_dLᵢₖ[k])
	end
	if length(∇Q.𝐯) == K
		@inbounds for k = 1:K
			mul!(∇Q.𝐯[k], 𝐕', ∑ᵢ_dQᵢₖ_dLᵢₖ⨀ωᵢ[k])
		end
	else
		mul!(∇Q.𝐯[1], 𝐕', sum(∑ᵢ_dQᵢₖ_dLᵢₖ⨀ωᵢ))
	end
	if ∇Q.fit_b
		if length(∇Q.𝐯) == K
			∇Q.b[1] = 0.0
			@inbounds for k = 1:K
				∇Q.b[1] += dot(∑ᵢ_dQᵢₖ_dLᵢₖ⨀dωᵢ_db[k], 𝐕, 𝐯[k])
			end
		else
			∇Q.b[1] = dot(sum(∑ᵢ_dQᵢₖ_dLᵢₖ⨀dωᵢ_db), 𝐕, 𝐯[k])
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
									   x::Vector{<:Real};
										omitb::Bool=false)
	mpGLM = MixturePoissonGLM(x, mpGLM; omitb=omitb)
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

"""
	∇negativeloglikelihood!(∇nℓ, ∇ℓglms, offset)

Concatenate the first-order partial derivatives of the model's log-likelihood w.r.t. to the parameters in each neuron's GLM

MODIFIED ARGUMENT
-`∇nℓ`: a vector representing the gradient of the model's log-likelihood

UNMODIFIED ARGUMENT
-`∇ℓglm`: a nested vector of the partial derivatives of the model's log-likelihood w.r.t. to the  parameter of each neuron's mixture of Poisson GLM. Element `∇ℓglms[i][n]` corresponds to the n-th neuron in the i-th trialset
-`offset`: the number of elements at beginning of `∇nℓ` that are unrelated to the GLM's
"""
function ∇negativeloglikelihood!(∇nℓ::Vector{<:Real}, ∇ℓglm::Vector{<:Vector{<:GLMθ}}, offset::Integer)
	counter = offset
	for ∇ℓglm in ∇ℓglm
		for ∇ℓglm in ∇ℓglm
			if ∇ℓglm.fit_b
				counter+=1
				∇nℓ[counter] = -∇ℓglm.b[1]
			end
			for k = 2:length(∇ℓglm.𝐠)
				counter+=1
				∇nℓ[counter] = -∇ℓglm.𝐠[k]
			end
			for u in ∇ℓglm.𝐮
				counter+=1
				∇nℓ[counter] = -u
			end
			for 𝐯ₖ in ∇ℓglm.𝐯
				for v in 𝐯ₖ
					counter+=1
					∇nℓ[counter] = -v
				end
			end
		end
	end
	return nothing
end
