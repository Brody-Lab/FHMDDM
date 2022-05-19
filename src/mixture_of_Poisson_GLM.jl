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
	if y==0
		1/exp(λΔt)
	elseif y==1
		λΔt/exp(λΔt)
	else
		λΔt^y / exp(λΔt) / factorial(y)
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
    @unpack 𝐮, 𝐯 = mpGLM.θ
	𝐗*vcat(𝐮, 𝐯[k].*d𝛏_dB[j])
end

"""
    expectation_loglikelihood(γ, mpGLM, x)

ForwardDiff-compatible computation of the expectation of the log-likelihood of the mixture of Poisson generalized model of one neuron

Ignores the log(y!) term, which does not depend on the parameters

ARGUMENT
-`γ`: posterior probability of the latent variable
-`mpGLM`: the GLM of one neuron
-`x`: weights of the linear filters of the GLM concatenated as a vector of floating-point numbers

RETURN
-expectation of the log-likelihood of the spike train of one neuron
"""
function expectation_loglikelihood(concatenatedθ::Vector{<:Real},
								   γ::Matrix{<:Vector{<:AbstractFloat}},
								   mpGLM::MixturePoissonGLM)
	mpGLM = MixturePoissonGLM(concatenatedθ, mpGLM)
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
	expectation_∇loglikelihood!(∇Q, indexθ, γ, mpGLM)

Expectation under the posterior probability of the gradient of the log-likelihood

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
function expectation_∇loglikelihood!(∇Q::GLMθ,
	                                γ::Matrix{<:Vector{<:Real}},
	                                mpGLM::MixturePoissonGLM)
	@unpack Δt, d𝛏_dB, 𝐕, 𝐗, 𝐲 = mpGLM
	Ξ, K = size(γ)
	T = length(𝐲)
	∑ᵢ_dQᵢₖ_dLᵢₖ = collect(zeros(T) for k=1:K)
	∑ᵢ_dQᵢₖ_dLᵢₖ⨀dξᵢ_dB = collect(zeros(T) for k=1:K)
	@inbounds for i = 1:Ξ
		for k = 1:K
			𝐋 = linearpredictor(mpGLM,i,k)
			for t=1:T
				dQᵢₖ_dLᵢₖ = γ[i,k][t] * differentiate_loglikelihood_wrt_linearpredictor(Δt, 𝐋[t], 𝐲[t])
				∑ᵢ_dQᵢₖ_dLᵢₖ[k][t] += dQᵢₖ_dLᵢₖ
				∑ᵢ_dQᵢₖ_dLᵢₖ⨀dξᵢ_dB[k][t] += dQᵢₖ_dLᵢₖ*d𝛏_dB[i]
			end
		end
	end
	q = size(𝐗,2)-size(𝐕,2)
	𝐔ᵀ = transpose(@view 𝐗[:,1:q])
	𝐕ᵀ = transpose(𝐕)
	∇Q.𝐮 .= 𝐔ᵀ*sum(∑ᵢ_dQᵢₖ_dLᵢₖ)
	@inbounds for k = 1:K
		∇Q.𝐯[k] .= 𝐕ᵀ*∑ᵢ_dQᵢₖ_dLᵢₖ⨀dξᵢ_dB[k]
	end
	return nothing
end

"""
	learn_state_independent_filters!(model)

Learn the filters of the state-independent inputs of each neuron's GLM

EXAMPLE
```julia-repl
julia> using FHMDDM, Random
julia> model = Model("/mnt/cup/labs/brody/tzluo/analysis_data/analysis_2022_05_05_test/data.mat"; randomize=true);
julia> FHMDDM.learn_state_independent_filters!(model)
```
"""
function learn_state_independent_filters!(model::Model)
	q = length(model.trialsets[1].mpGLMs[1].θ.𝐮)
	Opt = PoissonGLMOptimization(𝐮 = fill(NaN, q))
	for trialset in model.trialsets
		for mpGLM in trialset.mpGLMs
			learn_state_independent_filters!(mpGLM, Opt)
		end
	end
	return nothing
end

"""
    learn_state_independent_filters!(mpGLM, Opt)

Learn the filters of the state-independent inputs

ARGUMENT
-`mpGLM`: the Poisson mixture GLM of one neuron

OPTIONAL ARGUMENT
-`show_trace`: whether to show information about each step of the optimization

RETURN
-weights concatenated into a single vector

EXAMPLE
```julia-repl
julia> using FHMDDM, Random
julia> model = Model("/mnt/cup/labs/brody/tzluo/analysis_data/analysis_2022_05_05_test/data.mat"; randomize=true);
julia> mpGLM = model.trialsets[1].mpGLMs[1]
julia> Opt = FHMDDM.PoissonGLMOptimization(𝐮 = fill(NaN, length(mpGLM.θ.𝐮)))
julia> FHMDDM.estimatefilters!(mpGLM, Opt)
```
"""
function learn_state_independent_filters!(mpGLM::MixturePoissonGLM,
										Opt::PoissonGLMOptimization,
										iterations::Integer=20,
										show_trace::Bool=false)
    f(𝐮) = -loglikelihood!(mpGLM,Opt,𝐮)
	g!(∇, 𝐮) = ∇negloglikelihood!(∇,mpGLM,Opt,𝐮)
	h!(∇∇, 𝐮) = ∇∇negloglikelihood!(∇∇,mpGLM,Opt,𝐮)
    results = Optim.optimize(f, g!, h!, copy(mpGLM.θ.𝐮), NewtonTrustRegion(), Optim.Options(show_trace=show_trace, iterations=iterations))
	mpGLM.θ.𝐮 .= Optim.minimizer(results)
	return nothing
end

"""
	loglikelihood!(mpGLM, Opt, concatenatedθ)

Compute the log-likelihood of Poisson GLM

ARGUMENT
-`mpGLM`: a structure containing the parameters, input, and observations a mixture of Poisson GLM
-`Opt`: a structure for maximizing the expectation of the log-likelihood of a mixture of Poisson GLM
-`concatenatedθ`: parameters of a GLM concatenated into a vector

RETURN
-log-likelihood of a Poisson GLM
"""
function loglikelihood!(mpGLM::MixturePoissonGLM,
						Opt::PoissonGLMOptimization,
						𝐮::Vector{<:AbstractFloat})
	update!(mpGLM, Opt, 𝐮)
	return Opt.ℓ[1]
end

"""
	∇negloglikelihood!(g, mpGLM, Opt, γ, concatenatedθ)

Compute the gradient of the negative of the log-likelihood of a Poisson GLM

MODIFIED ARGUMENT
-`g`: gradient

ARGUMENT
-`mpGLM`: a structure containing the parameters, input, and observations a mixture of Poisson GLM
-`Opt`: a structure for maximizing the expectation of the log-likelihood of a mixture of Poisson GLM
-`concatenatedθ`: parameters of a GLM concatenated into a vector
"""
function ∇negloglikelihood!(g::Vector{<:AbstractFloat},
							mpGLM::MixturePoissonGLM,
							Opt::PoissonGLMOptimization,
							𝐮::Vector{<:AbstractFloat})
	update!(mpGLM, Opt, 𝐮)
	for i in eachindex(g)
		g[i] = -Opt.∇ℓ[i]
	end
	return nothing
end

"""
	∇∇negloglikelihood!(h, mpGLM, Opt, γ, concatenatedθ)

Compute the hessian of the negative of the log-likelihood of a Poisson GLM

MODIFIED ARGUMENT
-`h`: hessian

ARGUMENT
-`mpGLM`: a structure containing the parameters, input, and observations a mixture of Poisson GLM
-`Opt`: a structure for maximizing the expectation of the log-likelihood of a mixture of Poisson GLM
-`concatenatedθ`: parameters of a GLM concatenated into a vector
"""
function ∇∇negloglikelihood!(h::Matrix{<:AbstractFloat},
							mpGLM::MixturePoissonGLM,
							Opt::PoissonGLMOptimization,
							𝐮::Vector{<:AbstractFloat})
	update!(mpGLM, Opt, 𝐮)
	for i in eachindex(h)
		h[i] = -Opt.∇∇ℓ[i]
	end
	return nothing
end

"""
	update!(mpGLM, Opt, concatenatedθ)

Update quantities for computing the log-likelihood of a Poisson GLM and its gradient and hessian

MODIFIED ARGUMENT
-`mpGLM`: a structure containing the parameters, input, and observations a mixture of Poisson GLM
-`Opt`: a structure for maximizing the expectation of the log-likelihood of a mixture of Poisson GLM

ARGUMENT
-`concatenatedθ`: parameters of a GLM concatenated into a vector

EXAMPLE
```julia-rep
julia> using FHMDDM, Random
julia> model = Model("/mnt/cup/labs/brody/tzluo/analysis_data/analysis_2022_05_05_test/data.mat"; randomize=true);
julia> mpGLM = model.trialsets[1].mpGLMs[1]
julia> Opt = FHMDDM.PoissonGLMOptimization(𝐮 = fill(NaN, length(mpGLM.θ.𝐮)))
julia> rand𝐮 = copy(mpGLM.θ.𝐮)
julia> FHMDDM.update!(mpGLM, Opt, rand𝐮)
julia> using ForwardDiff
julia> f(𝐮) = FHMDDM.loglikelihood(mpGLM, 𝐮)
julia> fauto = f(rand𝐮)
julia> gauto = ForwardDiff.gradient(f, rand𝐮)
julia> hauto = ForwardDiff.hessian(f, rand𝐮)
julia> abs(fauto - Opt.ℓ[1])
julia> maximum(abs.(gauto .- Opt.∇ℓ))
julia> maximum(abs.(hauto .- Opt.∇∇ℓ))
```
"""
function update!(mpGLM::MixturePoissonGLM,
				Opt::PoissonGLMOptimization,
				𝐮::Vector{<:AbstractFloat})
	if 𝐮 != Opt.𝐮
		Opt.𝐮 .= 𝐮
		mpGLM.θ.𝐮 .= 𝐮
		∇∇loglikelihood!(Opt, mpGLM)
	end
end

"""
    ∇∇loglikelihood!(Opt, mpGLM)

Compute the log-likelihood of a Poisson mixture GLM and its first and second derivatives

MODIFIED ARGUMENT
-`Opt`: a structure for maximizing the expectation of the log-likelihood of a mixture of Poisson GLM

UNMODIFIED ARGUMENT
-`mpGLM`: the Poisson mixture GLM of one neuron
```
"""
function ∇∇loglikelihood!(Opt::PoissonGLMOptimization, mpGLM::MixturePoissonGLM)
	@unpack ℓ, ∇ℓ, ∇∇ℓ = Opt
    @unpack Δt, 𝐗, 𝐲 = mpGLM
	@unpack 𝐮 = mpGLM.θ
	𝐔 = @view 𝐗[:,1:length(𝐮)]
	𝐔ᵀ = transpose(𝐔)
	𝐋 = 𝐔*𝐮
	T = length(𝐲)
	d²ℓ_dL², dℓ_dL = zeros(T), zeros(T)
	ℓ[1] = 0.0
	for t = 1:T
		d²ℓ_dL²[t], dℓ_dL[t], ℓₜ = differentiate_loglikelihood_twice_wrt_linearpredictor(Δt, 𝐋[t], 𝐲[t])
		ℓ[1] += ℓₜ
	end
	∇ℓ .= 𝐔ᵀ*dℓ_dL
	∇∇ℓ .= 𝐔ᵀ*(d²ℓ_dL².*𝐔)
	return nothing
end

"""
	loglikelihood(mpGLM, concatenatedθ)

Compute the log-likelihood of Poisson GLM

ARGUMENT
-`mpGLM`: a structure containing the parameters, input, and observations a mixture of Poisson GLM
-`𝐮`: filters of the state-independent inputs of the GLM

RETURN
-log-likelihood of a Poisson GLM
"""
function loglikelihood(mpGLM::MixturePoissonGLM, 𝐮::Vector{type}) where {type<:Real}
	@unpack Δt, 𝐗, 𝐲 = mpGLM
	𝐔 = @view 𝐗[:,1:length(𝐮)]
	𝐔ᵀ = transpose(𝐔)
	𝐋 = 𝐔*𝐮
	T = length(𝐲)
	d²ℓ_dL², dℓ_dL = zeros(type, T), zeros(type, T)
	ℓ = 0.0
	for t = 1:T
		ℓ += poissonloglikelihood(Δt, 𝐋[t], 𝐲[t])
	end
	ℓ
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
	GLMθ(K, 𝐇, 𝐔, 𝐕)

Randomly initiate the parameters for a mixture of Poisson generalized linear model

ARGUMENT
-`K`: number of coupling states
-`𝐗`: constant input, time-varying inputs from spike history, time-varying inputs from trial events
-`𝐕`: constant and time-varying inputs from the accumulator

OUTPUT
-an instance of `GLMθ`
"""
function GLMθ(K::Integer,
			𝐗::Matrix{<:AbstractFloat},
			𝐕::Matrix{<:AbstractFloat})
	n𝐯 =size(𝐕,2)
	n𝐮 = size(𝐗,2)-size(𝐕,2)
	if K == 1
		𝐯 = [ones(n𝐯)]
	else
		𝐯 = collect(i*ones(n𝐯) for i = -1:2/(K-1):1)
	end
	θ = GLMθ(𝐮 = 1.0 .- 2.0.*rand(n𝐮),
			 𝐯 = 𝐯)
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
	GLMθ(𝐮 = zeros(elementtype, length(glmθ.𝐮)),
		 𝐯 = collect(zeros(elementtype, length(𝐯)) for 𝐯 in glmθ.𝐯))
end

"""
    transformaccumulator

Nonlinearly transform the normalized values of the accumulator

ARGUMENT
-`ξ`: value of the accumulator: expected to be between -1 and 1
-`b`: parameter specifying the transformation

RETURN
-transformed value of the accumulator
"""
function transformaccumulator(b::Real, ξ::Real)
    if b == 0.0
        ξ
    else
        if ξ < 0
            if b > 709.0 # 709 is close to which exp returns Inf for a 64-bit floating point number
                ξ == -1.0 ? -1.0 : 0.0
            else
                -expm1(-b*ξ)/expm1(b)
                # (exp(-b*ξ)-1.0)/(1.0-exp(b))
            end
        elseif ξ > 0
            if b > 709.0
                ξ == 1.0 ? 1.0 : 0.0
            else
                expm1(b*ξ)/expm1(b)
                # (1.0-exp(b*ξ))/(1.0-exp(b))
            end
        else
            0.0
        end
    end
end

"""
    dtransformaccumulator

Derivative of the nonlinear transformation of the normalized values of the accumulator with respect to b

ARGUMENT
-`ξ`: value of the accumulator: expected to be between -1 and 1
-`b`: parameter specifying the transformation

RETURN
-transformed value of the accumulator
"""
function dtransformaccumulator(b::Real, ξ::Real)
    if ξ == -1.0 || ξ == 0.0 || ξ == 1.0 || b > 709.0 # 709 is close to which exp returns Inf for a 64-bit floating point number
        0.0
    elseif abs(b) < 1e-6
        ξ < 0 ? (-ξ^2-ξ)/2 : (ξ^2-ξ)/2
    elseif ξ < 0
        eᵇ = exp(b)
        eᵇm1 = expm1(b)
        e⁻ᵇˣ = exp(-b*ξ)
        e⁻ᵇˣm1 = expm1(-b*ξ)
        if b < 1
            (ξ*e⁻ᵇˣ*eᵇm1 + e⁻ᵇˣm1*eᵇ)/eᵇm1^2
        else
            ξ*e⁻ᵇˣ/eᵇm1 + e⁻ᵇˣm1/(eᵇ-2+exp(-b))
        end
    elseif ξ > 0
        eᵇ = exp(b)
        eᵇm1 = expm1(b)
        eᵇˣ = exp(b*ξ)
        eᵇˣm1 = expm1(b*ξ)
        if b < 1
            ξ*eᵇˣ/eᵇm1 - eᵇˣm1*eᵇ/eᵇm1^2
        else
            ξ*eᵇˣ/eᵇm1 - eᵇˣm1/(eᵇ-2+exp(-b))
        end
    end
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
			𝐋 = linearpredictor(mpGLM,i,k)
			for t=1:T
				d²ℓ_dL², dℓ_dL, ℓ = differentiate_loglikelihood_twice_wrt_linearpredictor(Δt, 𝐋[t], 𝐲[t])
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
	n𝐮 = length(mpGLM.θ.𝐮)
	n𝐯ₖ = length(mpGLM.θ.𝐯[1])
	indices𝐮 = 1:n𝐮
	𝐔 = @view 𝐗[:, indices𝐮]
	𝐔ᵀ, 𝐕ᵀ = transpose(𝐔), transpose(𝐕)
	∑ᵢₖ_dQᵢₖ_dLᵢₖ = sum(∑ᵢ_dQᵢₖ_dLᵢₖ)
	∑ᵢₖ_d²Qᵢₖ_dLᵢₖ² = sum(∑ᵢ_d²Qᵢₖ_dLᵢₖ²)
	∇Q[indices𝐮] = 𝐔ᵀ*∑ᵢₖ_dQᵢₖ_dLᵢₖ
	∇∇Q[indices𝐮, indices𝐮] = 𝐔ᵀ*(∑ᵢₖ_d²Qᵢₖ_dLᵢₖ².*𝐔)
	@inbounds for k=1:K
		indices𝐯ₖ = n𝐮+(k-1)*n𝐯ₖ+1:n𝐮+k*n𝐯ₖ
		∇Q[indices𝐯ₖ] = 𝐕ᵀ*∑ᵢ_dQᵢₖ_dLᵢₖ⨀dξᵢ_dB[k]
		∇∇Q[indices𝐮, indices𝐯ₖ] = 𝐔ᵀ*(∑ᵢ_d²Qᵢₖ_dLᵢₖ²⨀dξᵢ_dB[k].*𝐕)
		∇∇Q[indices𝐯ₖ, indices𝐯ₖ] = 𝐕ᵀ*(∑ᵢ_d²Qᵢₖ_dLᵢₖ²⨀dξᵢ_dB²[k].*𝐕)
		for i in indices𝐮
			for j in indices𝐯ₖ
				∇∇Q[j,i] = ∇∇Q[i,j]
			end
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
