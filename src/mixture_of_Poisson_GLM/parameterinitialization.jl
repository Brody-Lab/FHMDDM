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
		θ.𝐮[i] = 0.0 #1.0 .- 2rand()
	end
	θ.𝐠[1] = 0.0
	for k = 2:length(θ.𝐠)
		θ.𝐠[k] = 1.0 .- 2rand()
	end
	if length(θ.𝐯) > 1
		K = length(θ.𝐯)
		𝐯₀ = -0.01:0.02/(K-1):0.01
		for k = 1:K
			θ.𝐯[k] .= 𝐯₀[k]
		end
	else
		θ.𝐯[1] .= 0.0
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
	initialize_GLM_parameters!(model)

Initialize the GLM parameters

MODIFIED ARGUMENT
-`model`: an instance of the factorial hidden Markov drift-diffusion model
"""
function initialize_GLM_parameters!(model::Model; show_trace::Bool=false)
	memory = FHMDDM.Memoryforgradient(model)
	choiceposteriors!(memory, model)
	for i in eachindex(model.trialsets)
	    for mpGLM in model.trialsets[i].mpGLMs
	        maximize_expectation_of_loglikelihood!(mpGLM, memory.γ[i]; show_trace=show_trace)
	    end
	end
	if model.options.gain_state_dependent
		for i in eachindex(model.trialsets)
		    for mpGLM in model.trialsets[i].mpGLMs
		        for k = 2:length(mpGLM.θ.𝐠)
					mpGLM.θ.𝐠[k] = 1-2rand()
				end
		    end
		end
	end
	if model.options.tuning_state_dependent
		for i in eachindex(model.trialsets)
			for mpGLM in model.trialsets[i].mpGLMs
				vmean = mean(mpGLM.θ.𝐯)
				mpGLM.θ.𝐯[1] .= 3.0.*vmean
				mpGLM.θ.𝐯[2] .= -vmean
			end
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
	x₀ = concatenateparameters(mpGLM.θ; omitb=true)
	nparameters = length(x₀)
	Q = fill(NaN,1)
	∇Q = fill(NaN, nparameters)
	∇∇Q = fill(NaN, nparameters, nparameters)
	f(x) = -expectation_of_loglikelihood!(mpGLM,Q,∇Q,∇∇Q,γ,x)
	∇f!(∇, x) = negexpectation_of_∇loglikelihood!(∇,mpGLM,Q,∇Q,∇∇Q,γ,x)
	∇∇f!(∇∇, x) = negexpectation_of_∇∇loglikelihood!(∇∇,mpGLM,Q,∇Q,∇∇Q,γ,x)
    results = Optim.optimize(f, ∇f!, ∇∇f!, x₀, NewtonTrustRegion(), Optim.Options(show_trace=show_trace, iterations=iterations))
	sortparameters!(mpGLM.θ, Optim.minimizer(results); omitb=true)
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
	x₀ = concatenateparameters(mpGLM.θ; omitb=true)
	if (x != x₀) || isnan(Q[1])
		sortparameters!(mpGLM.θ, x; omitb=true)
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
	x₀ = concatenateparameters(mpGLM.θ; omitb=true)
	if (x != x₀) || isnan(Q[1])
		sortparameters!(mpGLM.θ, x; omitb=true)
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
	x₀ = concatenateparameters(mpGLM.θ; omitb=true)
	if (x != x₀) || isnan(Q[1])
		sortparameters!(mpGLM.θ, x; omitb=true)
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
    @unpack Δt, 𝐕, 𝐗, 𝐲 = mpGLM
	@unpack 𝐠, 𝐮, 𝐯 = mpGLM.θ
	𝛚 = transformaccumulator(mpGLM)
	𝛚² = 𝛚.^2
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
				d²ℓ_dL², dℓ_dL, ℓ = differentiate_twice_loglikelihood_wrt_linearpredictor(Δt, 𝐋[t], 𝐲[t])
				Q[1] += γ[i,k][t]*ℓ
				dQᵢₖ_dLᵢₖ = γ[i,k][t] * dℓ_dL
				∑ᵢ_dQᵢₖ_dLᵢₖ[k][t] += dQᵢₖ_dLᵢₖ
				∑ᵢ_dQᵢₖ_dLᵢₖ⨀dξᵢ_dB[k][t] += dQᵢₖ_dLᵢₖ*𝛚[i]
				d²Qᵢₖ_dLᵢₖ² = γ[i,k][t] * d²ℓ_dL²
				∑ᵢ_d²Qᵢₖ_dLᵢₖ²[k][t] += d²Qᵢₖ_dLᵢₖ²
				∑ᵢ_d²Qᵢₖ_dLᵢₖ²⨀dξᵢ_dB[k][t] += d²Qᵢₖ_dLᵢₖ²*𝛚[i]
				∑ᵢ_d²Qᵢₖ_dLᵢₖ²⨀dξᵢ_dB²[k][t] += d²Qᵢₖ_dLᵢₖ²*𝛚²[i]
			end
		end
	end
	K𝐠 = length(𝐠)
	K𝐯 = length(𝐯)
	n𝐮 = length(𝐮)
	n𝐯 = length(𝐯[1])
	if K𝐠 > 1
		indices𝐠 = 1:K𝐠-1
		indices𝐮 = indices𝐠[end] .+ (1:n𝐮)
	else
		indices𝐮 = 1:n𝐮
	end
	indices𝐯 = collect(indices𝐮[end] .+ ((k-1)*n𝐯+1 : k*n𝐯) for k = 1:K𝐯)
	𝐔 = @view 𝐗[:, 2:1+n𝐮]
	𝐔ᵀ, 𝐕ᵀ = transpose(𝐔), transpose(𝐕)
	∑ᵢₖ_dQᵢₖ_dLᵢₖ = sum(∑ᵢ_dQᵢₖ_dLᵢₖ)
	∑ᵢₖ_d²Qᵢₖ_dLᵢₖ² = sum(∑ᵢ_d²Qᵢₖ_dLᵢₖ²)
	∇Q[indices𝐮] .= 𝐔ᵀ*∑ᵢₖ_dQᵢₖ_dLᵢₖ
	∇∇Q[indices𝐮, indices𝐮] .= 𝐔ᵀ*(∑ᵢₖ_d²Qᵢₖ_dLᵢₖ².*𝐔)
	if K𝐠 > 1
		@inbounds for k = 2:K
			∇Q[indices𝐠[k-1]] = sum(∑ᵢ_dQᵢₖ_dLᵢₖ[k])
			∇∇Q[indices𝐠[k-1], indices𝐠[k-1]] = sum(∑ᵢ_d²Qᵢₖ_dLᵢₖ²[k])
			∇∇Q[indices𝐠[k-1], indices𝐮] = transpose(∑ᵢ_d²Qᵢₖ_dLᵢₖ²[k])*𝐔
		end
		@inbounds for k = 2:K𝐯
			∇∇Q[indices𝐠[k-1], indices𝐯[k]] = transpose(∑ᵢ_d²Qᵢₖ_dLᵢₖ²⨀dξᵢ_dB[k])*𝐕
		end
	end
	if K𝐯 > 1
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
	for i = 1:size(∇∇Q,1)
		for j = i+1:size(∇∇Q,2)
			∇∇Q[j,i] = ∇∇Q[i,j]
		end
	end
	return nothing
end
