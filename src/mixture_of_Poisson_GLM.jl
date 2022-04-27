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
	Poissonlikelihood(λΔt, y, y!)

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
-`𝛌`: a vector whose element 𝛌[t] corresponds to the t-th time bin in the trialset
"""
function linearpredictor(mpGLM::MixturePoissonGLM, j::Integer, k::Integer)
    @unpack 𝐇, 𝐔, 𝐕, d𝛏_dB = mpGLM
    @unpack 𝐡, 𝐮, 𝐯, 𝐰 = mpGLM.θ
	𝐇*𝐡 .+ 𝐔*𝐮[k] .+ 𝐕*(𝐯[k].*d𝛏_dB[j]) .+ 𝐰[k]
end

"""
    estimatefilters(γ, Opt, mpGLM)

Estimate the filters of the Poisson mixture GLM of one neuron

ARGUMENT
-`γ`: posterior probabilities of the latent
-`mpGLM`: the Poisson mixture GLM of one neuron

OPTIONAL ARGUMENT
-`show_trace`: whether to show information about each step of the optimization
-`fit_a`: whether to fit the asymmetric scaling factor
-`fit_b`: whether to fit the nonlinearity factor

RETURN
-weights concatenated into a single vector

EXAMPLE
```julia-repl
julia> using FHMDDM, Random
julia> model = Model("/mnt/cup/labs/brody/tzluo/analysis_data/analysis_2022_04_27_test/data.mat"; randomize=true);
julia> mpGLM = model.trialsets[1].mpGLMs[1]
julia> concatenatedθ, indexθ = FHMDDM.concatenateparameters(mpGLM.θ)
julia> Opt = FHMDDM.MixturePoissonGLM_Optimization(concatenatedθ=fill(NaN, length(concatenatedθ)), indexθ=indexθ)
julia> γ = FHMDDM.randomposterior(mpGLM; rng=MersenneTwister(1234))
julia> FHMDDM.estimatefilters!(mpGLM, Opt, γ)
```
"""
function estimatefilters!(mpGLM::MixturePoissonGLM,
						Opt::MixturePoissonGLM_Optimization,
						γ::Matrix{<:Vector{<:AbstractFloat}};
						iterations::Integer=20,
						show_trace::Bool=true)
    x₀ = concatenateparameters(mpGLM.θ)[1]
    f(x) = expectation_negloglikelihood!(mpGLM,Opt,γ,x)
	g!(∇, x) = expectation_∇negloglikelihood!(∇,mpGLM,Opt,γ,x)
	h!(∇∇, x) = expectation_∇∇negloglikelihood!(∇∇,mpGLM,Opt,γ,x)
    results = Optim.optimize(f, g!, h!, x₀, NewtonTrustRegion(), Optim.Options(show_trace=show_trace, iterations=iterations))
    show_trace && println("The model converged: ", Optim.converged(results))
	sortparameters!(mpGLM.θ, Optim.minimizer(results))
	return nothing
end

"""
	expectation_negloglikelihood!(mpGLM, Opt, γ, concatenatedθ)

Compute the negative of the expectation of the log-likelihood of a mixture of Poisson GLM

ARGUMENT
-`mpGLM`: a structure containing the parameters, input, and observations a mixture of Poisson GLM
-`γ`: posterior probabilities of the latent variables
-`Opt`: a structure for maximizing the expectation of the log-likelihood of a mixture of Poisson GLM
-`concatenatedθ`: parameters of a GLM concatenated into a vector

RETURN
-negative of the expectation of the log-likelihood of a mixture of Poisson GLM
"""
function expectation_negloglikelihood!(mpGLM::MixturePoissonGLM,
									Opt::MixturePoissonGLM_Optimization,
									γ::Matrix{<:Vector{<:AbstractFloat}},
									concatenatedθ::Vector{<:AbstractFloat})
	update!(mpGLM, Opt, γ, concatenatedθ)
	return -Opt.Q[1]
end

"""
	expectation_∇negloglikelihood!(g, mpGLM, Opt, γ, concatenatedθ)

Compute the gradient of the negative of the expectation of the log-likelihood of a mixture of Poisson GLM

MODIFIED ARGUMENT
-`g`: gradient

ARGUMENT
-`mpGLM`: a structure containing the parameters, input, and observations a mixture of Poisson GLM
-`γ`: posterior probabilities of the latent variables
-`Opt`: a structure for maximizing the expectation of the log-likelihood of a mixture of Poisson GLM
-`concatenatedθ`: parameters of a GLM concatenated into a vector
"""
function expectation_∇negloglikelihood!(g::Vector{<:AbstractFloat},
									mpGLM::MixturePoissonGLM,
									Opt::MixturePoissonGLM_Optimization,
									γ::Matrix{<:Vector{<:AbstractFloat}},
									concatenatedθ::Vector{<:AbstractFloat})
	update!(mpGLM, Opt, γ, concatenatedθ)
	for i in eachindex(g)
		g[i] = -Opt.∇Q[i]
	end
	return nothing
end

"""
	expectation_∇∇negloglikelihood!(h, mpGLM, Opt, γ, concatenatedθ)

Compute the hessian of the negative of the expectation of the log-likelihood of a mixture of Poisson GLM

MODIFIED ARGUMENT
-`h`: hessian

ARGUMENT
-`mpGLM`: a structure containing the parameters, input, and observations a mixture of Poisson GLM
-`γ`: posterior probabilities of the latent variables
-`Opt`: a structure for maximizing the expectation of the log-likelihood of a mixture of Poisson GLM
-`concatenatedθ`: parameters of a GLM concatenated into a vector
"""
function expectation_∇∇negloglikelihood!(h::Matrix{<:AbstractFloat},
									mpGLM::MixturePoissonGLM,
									Opt::MixturePoissonGLM_Optimization,
									γ::Matrix{<:Vector{<:AbstractFloat}},
									concatenatedθ::Vector{<:AbstractFloat})
	update!(mpGLM, Opt, γ, concatenatedθ)
	for i in eachindex(h)
		h[i] = -Opt.∇∇Q[i]
	end
	return nothing
end

"""
	update!(mpGLM, Opt, γ, concatenatedθ)

Update the expectation of the log-likelihood of a mixture of Poisson GLM and its gradient and hessian

MODIFIED ARGUMENT
-`mpGLM`: a structure containing the parameters, input, and observations a mixture of Poisson GLM
-`Opt`: a structure for maximizing the expectation of the log-likelihood of a mixture of Poisson GLM

ARGUMENT
-`γ`: posterior probabilities of the latent variables
-`concatenatedθ`: parameters of a GLM concatenated into a vector
"""
function update!(mpGLM::MixturePoissonGLM,
				Opt::MixturePoissonGLM_Optimization,
				γ::Matrix{<:Vector{<:AbstractFloat}},
				concatenatedθ::Vector{<:AbstractFloat})
	if concatenatedθ != Opt.concatenatedθ
		sortparameters!(mpGLM.θ, concatenatedθ)
		Opt.concatenatedθ .= concatenatedθ
		expectation_∇∇loglikelihood!(Opt, γ, mpGLM)
	end
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
								   mpGLM::MixturePoissonGLM,
								   γ::Matrix{<:Vector{<:AbstractFloat}})
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
	expectation_∇loglikelihood!(∇, γ, mpGLM)

Expectation under the posterior probability of the gradient of the log-likelihood

MODIFIED ARGUMENT
-`∇`: The gradient

UNMODIFIED ARGUMENT
-`γ`: Joint posterior probability of the accumulator and coupling variable. γ[i,k][t] corresponds to the i-th accumulator state and the k-th coupling state in the t-th time bin in the trialset.
-`mpGLM`: structure containing information for the mixture of Poisson GLM for one neuron

EXAMPLE
```julia-rep
julia> using FHMDDM, Random
julia> model = Model("/mnt/cup/labs/brody/tzluo/analysis_data/analysis_2022_04_27_test/data.mat"; randomize=true);
julia> mpGLM = model.trialsets[1].mpGLMs[1]
julia> concatenatedθ, indexθ = FHMDDM.concatenateparameters(mpGLM.θ)
julia> γ = FHMDDM.randomposterior(mpGLM; rng=MersenneTwister(1234))
julia> ghand = similar(concatenatedθ)
julia> FHMDDM.expectation_∇loglikelihood!(ghand, indexθ, γ, mpGLM)
julia> using ForwardDiff
julia> f(x) = FHMDDM.expectation_loglikelihood(x, mpGLM, γ)
julia> gauto = ForwardDiff.gradient(f, concatenatedθ)
julia> maximum(abs.(gauto .- ghand))
```
"""
function expectation_∇loglikelihood!(∇Q::Vector{<:Real},
									indexθ::GLMθ,
	                                γ::Matrix{<:Vector{<:Real}},
	                                mpGLM::MixturePoissonGLM)
	@unpack Δt, 𝐇, 𝐔, 𝐕, d𝛏_dB, θ, 𝐲 = mpGLM
	Ξ = size(γ,1)
	K = length(mpGLM.θ.𝐯)
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
	∑ᵢₖ_dQᵢₖ_dLᵢₖ = sum(∑ᵢ_dQᵢₖ_dLᵢₖ)
	𝐇ᵀ, 𝐔ᵀ, 𝐕ᵀ = transpose(𝐇), transpose(𝐔), transpose(𝐕)
	∇Q[indexθ.𝐡] = 𝐇ᵀ*∑ᵢₖ_dQᵢₖ_dLᵢₖ
	@inbounds for k = 1:K
		∇Q[indexθ.𝐮[k]] = 𝐔ᵀ*∑ᵢ_dQᵢₖ_dLᵢₖ[k]
		∇Q[indexθ.𝐯[k]] = 𝐕ᵀ*∑ᵢ_dQᵢₖ_dLᵢₖ⨀dξᵢ_dB[k]
		∇Q[indexθ.𝐰[k]] = sum(∑ᵢ_dQᵢₖ_dLᵢₖ[k])
	end
	return nothing
end

"""
    expectation_∇∇loglikelihood(Opt, γ, mpGLM)

Compute the log-likelihood of a Poisson mixture GLM and its first and second derivatives

MODIFIED ARGUMENT
-`∇∇Q`: Hessian matrix

UNMODIFIED ARGUMENT
-`γ`: posterior probabilities of the latents
-`mpGLM`: the Poisson mixture GLM of one neuron
-`x`: filters of the Poisson mixture GLM

RETURN
-nothing

EXAMPLE
```julia-rep
julia> using FHMDDM, Random
julia> model = Model("/mnt/cup/labs/brody/tzluo/analysis_data/analysis_2022_04_27_test/data.mat"; randomize=true);
julia> mpGLM = model.trialsets[1].mpGLMs[1]
julia> concatenatedθ, indexθ = FHMDDM.concatenateparameters(mpGLM.θ)
julia> Opt = FHMDDM.MixturePoissonGLM_Optimization(concatenatedθ=fill(NaN, length(concatenatedθ)), indexθ=indexθ)
julia> γ = FHMDDM.randomposterior(mpGLM; rng=MersenneTwister(1234))
julia> FHMDDM.expectation_∇∇loglikelihood!(Opt, γ, mpGLM)
julia> using ForwardDiff
julia> f(x) = FHMDDM.expectation_loglikelihood(x, mpGLM, γ)
julia> fauto = f(concatenatedθ)
julia> gauto = ForwardDiff.gradient(f, concatenatedθ)
julia> hauto = ForwardDiff.hessian(f, concatenatedθ)
julia> abs(fauto - Opt.Q[1])
julia> maximum(abs.(gauto .- Opt.∇Q))
julia> maximum(abs.(hauto .- Opt.∇∇Q))
```
"""
function expectation_∇∇loglikelihood!(Opt::MixturePoissonGLM_Optimization,
									γ::Matrix{<:Vector{<:AbstractFloat}},
									mpGLM::MixturePoissonGLM)
	@unpack indexθ, Q, ∇Q, ∇∇Q = Opt
    @unpack Δt, 𝐇, 𝐔, 𝐕, d𝛏_dB, θ, 𝐲 = mpGLM
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
	𝐇ᵀ, 𝐔ᵀ, 𝐕ᵀ = transpose(𝐇), transpose(𝐔), transpose(𝐕)
	∑ᵢₖ_dQᵢₖ_dLᵢₖ = sum(∑ᵢ_dQᵢₖ_dLᵢₖ)
	∇Q[indexθ.𝐡] = 𝐇ᵀ*∑ᵢₖ_dQᵢₖ_dLᵢₖ
	@inbounds for k = 1:K
		∇Q[indexθ.𝐮[k]] = 𝐔ᵀ*∑ᵢ_dQᵢₖ_dLᵢₖ[k]
		∇Q[indexθ.𝐯[k]] = 𝐕ᵀ*∑ᵢ_dQᵢₖ_dLᵢₖ⨀dξᵢ_dB[k]
		∇Q[indexθ.𝐰[k]] = sum(∑ᵢ_dQᵢₖ_dLᵢₖ[k])
	end
	∑ᵢₖ_d²Qᵢₖ_dLᵢₖ² = sum(∑ᵢ_d²Qᵢₖ_dLᵢₖ²)
	∇∇Q[indexθ.𝐡, indexθ.𝐡] = 𝐇ᵀ*(∑ᵢₖ_d²Qᵢₖ_dLᵢₖ².*𝐇)
	@inbounds for k=1:K
		∇∇Q[indexθ.𝐡, indexθ.𝐮[k]] = 𝐇ᵀ*(∑ᵢ_d²Qᵢₖ_dLᵢₖ²[k].*𝐔)
		∇∇Q[indexθ.𝐡, indexθ.𝐯[k]] = 𝐇ᵀ*(∑ᵢ_d²Qᵢₖ_dLᵢₖ²⨀dξᵢ_dB[k].*𝐕)
		∇∇Q[indexθ.𝐡, indexθ.𝐰[k]] = 𝐇ᵀ*∑ᵢ_d²Qᵢₖ_dLᵢₖ²[k]
		∇∇Q[indexθ.𝐮[k], indexθ.𝐮[k]] = 𝐔ᵀ*(∑ᵢ_d²Qᵢₖ_dLᵢₖ²[k].*𝐔)
		∇∇Q[indexθ.𝐮[k], indexθ.𝐯[k]] = 𝐔ᵀ*(∑ᵢ_d²Qᵢₖ_dLᵢₖ²⨀dξᵢ_dB[k].*𝐕)
		∇∇Q[indexθ.𝐮[k], indexθ.𝐰[k]] = 𝐔ᵀ*(∑ᵢ_d²Qᵢₖ_dLᵢₖ²[k])
		∇∇Q[indexθ.𝐯[k], indexθ.𝐯[k]] = 𝐕ᵀ*(∑ᵢ_d²Qᵢₖ_dLᵢₖ²⨀dξᵢ_dB²[k].*𝐕)
		∇∇Q[indexθ.𝐯[k], indexθ.𝐰[k]] = 𝐕ᵀ*(∑ᵢ_d²Qᵢₖ_dLᵢₖ²⨀dξᵢ_dB[k])
		∇∇Q[indexθ.𝐰[k], indexθ.𝐰[k]] = sum(∑ᵢ_d²Qᵢₖ_dLᵢₖ²[k])
	end
	@inbounds for q=1:size(∇∇Q,1) # update the lower triangle
		for r=q+1:size(∇∇Q,2)
			∇∇Q[r,q] = ∇∇Q[q,r]
		end
	end
	return nothing
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
    λ = softplus(L)
	if y == 0
		-λ*Δt
	elseif y == 1
		log(λ) - λ*Δt
	else
		y*log(λ) - λ*Δt
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
	if y == 0
		ℓ = -f₀*Δt
	elseif y == 1
		ℓ = log(f₀) - f₀*Δt
	else
		ℓ = y*log(f₀) - f₀*Δt
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
-`𝐇`: time-varying inputs from spike history
-`𝐔`: time-varying inputs from trial events
-`𝐕`: time-varying inputs from the accumulator

OUTPUT
-an instance of `GLMθ`
"""
function GLMθ(K::Integer,
			𝐇::Matrix{<:AbstractFloat},
			𝐔::Matrix{<:AbstractFloat},
			𝐕::Matrix{<:AbstractFloat})
	n𝐡 = size(𝐇,2)
	n𝐮 = size(𝐔,2)
	n𝐯 = size(𝐕,2)
	θ = GLMθ(𝐡 = 1.0 .- 2.0.*rand(n𝐡),
			 𝐰 = 1.0 .- 2.0.*rand(K),
			 𝐮 = collect(1.0 .- 2.0.*rand(n𝐮) for k=1:K),
			 𝐯 = collect(1.0 .- 2.0.*rand(n𝐯) for k=1:K))
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
	GLMθ(𝐡 = zeros(elementtype, length(glmθ.𝐡)),
		 𝐮 = collect(zeros(elementtype, length(𝐮)) for 𝐮 in glmθ.𝐮),
		 𝐯 = collect(zeros(elementtype, length(𝐯)) for 𝐯 in glmθ.𝐯),
		 𝐰 = zeros(elementtype, length(glmθ.𝐰)))
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
