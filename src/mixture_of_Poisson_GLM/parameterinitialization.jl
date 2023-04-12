"""
	GLMθ(indices𝐮, options, n𝐯)

Randomly initiate the parameters for a mixture of Poisson generalized linear model

ARGUMENT
-`indices𝐮`: indices of the encoding weights of the temporal basis vectors of each filter that is independent of the accumulator
-`n𝐯`: number of temporal basis vectors specifying the time-varying weight of the accumulator
-`options`: settings of the model

OUTPUT
-an instance of `GLMθ`
"""
function GLMθ(indices𝐮::Indices𝐮, n𝐯::Integer, options::Options)
	n𝐮 = maximum(vcat((getfield(indices𝐮, field) for field in fieldnames(Indices𝐮))...))
	θ = GLMθ(b_scalefactor = options.tbf_b_scalefactor,
			fit_b = options.fit_b,
			fit_𝛃 = options.fit_𝛃,
			fit_overdispersion = options.fit_overdispersion,
			𝐮 = fill(NaN, n𝐮),
			indices𝐮=indices𝐮,
			𝐯 = collect(fill(NaN,n𝐯) for k=1:options.K))
	randomizeparameters!(θ, options)
	return θ
end

"""
	randomizeparameters!(θ, options)

Randomly initialize parameters of a mixture of Poisson GLM

MODIFIED ARGUMENT
-`θ`: structure containing parameters of a mixture of Poisson GLM

UNMODIFIED ARGUMENT
-`options`: hyperparameters of the model
"""
function randomizeparameters!(θ::GLMθ, options::Options)
	θ.a[1] = θ.fit_overdispersion ? rand() : -Inf
	θ.b[1] = 0.0
	for i in eachindex(θ.𝐮)
		θ.𝐮[i] = 1.0 .- 2rand()
	end
	for fieldname in fieldnames(typeof(θ.indices𝐮))
		indices = getfield(θ.indices𝐮, fieldname)
		scalefactor = getfield(options, Symbol("tbf_"*String(fieldname)*"_scalefactor"))*options.sf_tbf[1]
		θ.𝐮[indices] ./= scalefactor
	end
	scalefactor = options.tbf_accumulator_scalefactor*options.sf_tbf[1]
	K = length(θ.𝐯)
	if K > 1
		𝐯₀ = (-1.0:2.0/(K-1):1.0)./scalefactor
		for k = 1:K
			θ.𝐯[k] .= 𝐯₀[k]
		end
	else
		θ.𝐯[1] .= (1.0 .- 2rand(length(θ.𝐯[1])))./scalefactor
	end
	for k = 1:K
		θ.𝛃[k] .= θ.fit_𝛃 ? -θ.𝐯[k] : 0.0
	end
end

"""
	initialize_GLM_parameters!(model)

Initialize the GLM parameters using expectation-maximization.

In the E-step, the posterior probability of the accumulator is computed by conditioning on only the behavioral choices. In the M-step, only the GLM parameters are updated. The E- and M-steps assume the coupling variable have only one state. After performing these two steps, if there are multiple coupling states, and the gain is state-dependent, it is randomly initialized. If there are multiple coupling states, and the encoding of the accumulated evidence is state-dependent, then the weight in the first state is set to be three times of the initialized value, and the weight in the second state is set to be the negative of the initialized value.

MODIFIED ARGUMENT
-`model`: structure containing the data, parameters, and hyperparameters

OPTIONAL ARGUMENT
-`show_trace`: whether the details of the M-step should be shown
"""
function initialize_GLM_parameters!(model::Model; iterations::Integer=5, show_trace::Bool=false)
	memory = FHMDDM.Memoryforgradient(model)
	P = choiceposteriors!(memory, model)
	for (trialset, γᵢ) in zip(model.trialsets, memory.γ)
	    for mpGLM in trialset.mpGLMs
	        maximize_expectation_of_loglikelihood!(mpGLM, γᵢ; show_trace=show_trace)
	    end
	end
	if model.options.K > 1
		for i in eachindex(model.trialsets)
			for mpGLM in model.trialsets[i].mpGLMs
				vmean = mean(mpGLM.θ.𝐯)
				mpGLM.θ.𝐯[1] .= mpGLM.θ.𝛃[1] .= 3.0.*vmean
				mpGLM.θ.𝐯[2] .= mpGLM.θ.𝛃[2] .= -vmean
			end
		end
	end
	maximize_expectation_of_loglikelihood!(model;iterations=iterations, show_trace=show_trace)
end

"""
	maximize_expectation_of_loglikelihood!(model)

Learn the parameters of each neuron's Poisson mixture GLM in the model
"""
function maximize_expectation_of_loglikelihood!(model::Model; iterations::Integer=5, show_trace::Bool=false)
	memory = FHMDDM.Memoryforgradient(model)
	P = update!(memory, model)
	for j = 1:iterations
		posteriors!(memory, P, model)
		for i in eachindex(model.trialsets)
			for mpGLM in model.trialsets[i].mpGLMs
				maximize_expectation_of_loglikelihood!(mpGLM, memory.γ[i]; show_trace=show_trace)
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
"""
function maximize_expectation_of_loglikelihood!(mpGLM::MixturePoissonGLM, γ::Matrix{<:Vector{<:Real}}; show_trace::Bool=false, iterations::Integer=20)
	x₀ = concatenateparameters(mpGLM.θ; initialization=true)
	nparameters = length(x₀)
	D = GLMDerivatives(mpGLM)
	Q = fill(NaN,1)
	∇Q = fill(NaN, nparameters)
	∇∇Q = fill(NaN, nparameters, nparameters)
	f(x) = negexpectation_of_loglikelihood!(mpGLM,D,Q,∇Q,∇∇Q,γ,x)
	∇f!(∇, x) = negexpectation_of_∇loglikelihood!(∇,mpGLM,D,Q,∇Q,∇∇Q,γ,x)
	∇∇f!(∇∇, x) = negexpectation_of_∇∇loglikelihood!(∇∇,mpGLM,D,Q,∇Q,∇∇Q,γ,x)
    results = Optim.optimize(f, ∇f!, ∇∇f!, x₀, NewtonTrustRegion(), Optim.Options(show_trace=show_trace, iterations=iterations))
	sortparameters!(mpGLM.θ, Optim.minimizer(results); initialization=true)
	return nothing
end

"""
	negexpectation_of_∇∇loglikelihood!(∇∇,mpGLM,D,Q,∇Q,∇∇Q,γ,x)

Hessian of the negative of the expectation of the log-likelihood under the posterior probability of the latent variables

MODIFIED ARGUMENT
-`∇∇`: Hessian of the negative of the expectation
-`mpGLM`: a structure containing the data and parameters of the mixture of Poisson GLM of one neuron
-`Q`: an one-element vector that quantifies the expectation
-`D`: an object for in-place computation of first and second derivatives of the log-likelihood
-`∇Q`: gradient of the expectation with respect to the filters in the k-th state
-`∇∇Q`: Hessian of the expectation with respect to the filters in the k-th state

UNMODIFIED ARGUMENT
-`γ`: posterior probability of the latent variables. Element `γ[j][τ]` corresponds to the posterior probability of the j-th accumulator state  in the τ-th time step
-`x`: filters
"""
function negexpectation_of_∇∇loglikelihood!(∇∇::Matrix{<:Real}, mpGLM::MixturePoissonGLM, D::GLMDerivatives, Q::Vector{<:Real}, ∇Q::Vector{<:Real}, ∇∇Q::Matrix{<:Real}, γ::Matrix{<:Vector{<:Real}}, x::Vector{<:Real})
	x₀ = concatenateparameters(mpGLM.θ; initialization=true)
	if (x != x₀) || isnan(Q[1])
		sortparameters!(mpGLM.θ, x; initialization=true)
		expectation_of_∇∇loglikelihood!(D,Q,∇Q,∇∇Q,γ,mpGLM)
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
	negexpectation_of_loglikelihood!(mpGLM,D,Q,∇Q,∇∇Q,γ,x)

Negative expectation of the log-likelihood under the posterior probability of the latent variables

MODIFIED ARGUMENT
-`∇`: gradient of the negative of the expectation

For other modified and unmodified arguments see documentation for `negexpectation_of_∇∇loglikelihood!(∇∇,mpGLM,D,Q,∇Q,∇∇Q,γ,x)`

RETURN
-a scalar that is the negative of the expectation of the log-likelihood under the posterior probability distribution
"""
function negexpectation_of_loglikelihood!(mpGLM::MixturePoissonGLM, D::GLMDerivatives, Q::Vector{<:Real}, ∇Q::Vector{<:Real}, ∇∇Q::Matrix{<:Real}, γ::Matrix{<:Vector{<:Real}}, x::Vector{<:Real})
	x₀ = concatenateparameters(mpGLM.θ; initialization=true)
	if (x != x₀) || isnan(Q[1])
		sortparameters!(mpGLM.θ, x; initialization=true)
		expectation_of_∇∇loglikelihood!(D,Q,∇Q,∇∇Q,γ,mpGLM)
	end
	-Q[1]
end

"""
	negexpectation_of_∇loglikelihood!(∇,mpGLM,D,Q,∇Q,∇∇Q,γ,x)

Gradient of the negative of the expectation of the log-likelihood under the posterior probability of the latent variables

ARGUMENT
For other modified and unmodified arguments see documentation for `negexpectation_of_∇∇loglikelihood!(∇∇,mpGLM,D,Q,∇Q,∇∇Q,γ,x)`
"""
function negexpectation_of_∇loglikelihood!(∇::Vector{<:Real}, mpGLM::MixturePoissonGLM, D::GLMDerivatives, Q::Vector{<:Real}, ∇Q::Vector{<:Real}, ∇∇Q::Matrix{<:Real}, γ::Matrix{<:Vector{<:Real}}, x::Vector{<:Real})
	x₀ = concatenateparameters(mpGLM.θ; initialization=true)
	if (x != x₀) || isnan(Q[1])
		sortparameters!(mpGLM.θ, x; initialization=true)
		expectation_of_∇∇loglikelihood!(D,Q,∇Q,∇∇Q,γ,mpGLM)
	end
	for i in eachindex(∇)
		∇[i] = -∇Q[i]
	end
	return nothing
end

"""
	expectation_of_∇∇loglikelihood!(D, Q,∇Q,∇∇Q,γ,k,mpGLM)

Compute the expectation of the log-likelihood and its gradient and Hessian

ARGUMENT
-`D`: an object for in-place computation of first and second derivatives of the log-likelihood
-`Q`: expectation of the log-likelihood under the posterior probability of the latent variables. Only the component in the coupling state `k` is included
-`∇Q`: first-order derivatives of the expectation
-`∇∇Q`: second-order derivatives of the expectation

UNMODIFIED ARGUMENT
-`γ`: posterior probabilities of the latent variables
-`k`: index of the coupling state
-`mpGLM`: a structure containing the data and parameters of the mixture of Poisson GLM of one neuron
"""
function expectation_of_∇∇loglikelihood!(D::GLMDerivatives, Q::Vector{<:type}, ∇Q::Vector{<:type}, ∇∇Q::Matrix{<:type}, γ::Matrix{<:Vector{<:type}}, mpGLM::MixturePoissonGLM) where {type<:AbstractFloat}
    @unpack Δt, 𝐕, 𝐗, 𝐲, d𝛏_dB = mpGLM
	@unpack a, 𝐮, 𝐯, 𝛃, fit_𝛃, fit_overdispersion = mpGLM.θ
	d𝛏_dB² = d𝛏_dB.^2
	Ξ, K = size(γ)
	T = length(𝐲)
	Q[1] = 0.0
	∇Q .= 0.0
	∇∇Q .= 0.0
	∑ᵢₖ_dQᵢₖ_dLᵢₖ = zeros(type,T)
	∑ᵢₖ_d²Qᵢₖ_dLᵢₖ² = zeros(type,T)
	∑ᵢₖ_d²Qᵢₖ_dadLᵢₖ = zeros(type,T)
	∑_post_dQᵢₖ_dLᵢₖ⨀dξᵢ_dB = collect(zeros(type,T) for k=1:K)
	∑_post_d²Qᵢₖ_dLᵢₖ²⨀dξᵢ_dB = collect(zeros(type,T) for k=1:K)
	∑_post_d²Qᵢₖ_dLᵢₖ²⨀dξᵢ_dB² = collect(zeros(type,T) for k=1:K)
	∑_pre_dQᵢₖ_dLᵢₖ⨀dξᵢ_dB = collect(zeros(type,T) for k=1:K)
	∑_pre_d²Qᵢₖ_dLᵢₖ²⨀dξᵢ_dB = collect(zeros(type,T) for k=1:K)
	∑_pre_d²Qᵢₖ_dLᵢₖ²⨀dξᵢ_dB² = collect(zeros(type,T) for k=1:K)
	differentiate_twice_overdispersion!(D, a[1])
	if fit_overdispersion
		∑_d²Q_da² = 0.0
		∑_dQ_da = 0.0
		∑_post_d²Qᵢₖ_dadLᵢₖ⨀dξᵢ_dB = collect(zeros(type,T) for k=1:K)
		∑_pre_d²Qᵢₖ_dadLᵢₖ⨀dξᵢ_dB = collect(zeros(type,T) for k=1:K)
	end
	@inbounds for i = 1:Ξ
		for k = 1:K
			𝐋 = linearpredictor(mpGLM,i,k)
			for t=1:T
				differentiate_twice_loglikelihood!(D,𝐋[t],mpGLM.𝐲[t])
				if fit_overdispersion
					∑_dQ_da += γ[i,k][t]*D.dℓ_da[1]
					∑_d²Q_da² += γ[i,k][t]*D.d²ℓ_da²[1]
					d²Qᵢₖ_dadLᵢₖ = γ[i,k][t]*D.d²ℓ_dadL[1]
					∑ᵢₖ_d²Qᵢₖ_dadLᵢₖ[t] += d²Qᵢₖ_dadLᵢₖ
					if (i==1) || (i==Ξ)
						∑_post_d²Qᵢₖ_dadLᵢₖ⨀dξᵢ_dB[k][t] += d²Qᵢₖ_dadLᵢₖ*d𝛏_dB[i]
					else
						∑_pre_d²Qᵢₖ_dadLᵢₖ⨀dξᵢ_dB[k][t] += d²Qᵢₖ_dadLᵢₖ*d𝛏_dB[i]
					end
				end
				Q[1] += γ[i,k][t]*D.ℓ[1]
				dQᵢₖ_dLᵢₖ = γ[i,k][t] * D.dℓ_dL[1]
				d²Qᵢₖ_dLᵢₖ² = γ[i,k][t] * D.d²ℓ_dL²[1]
				∑ᵢₖ_dQᵢₖ_dLᵢₖ[t] += dQᵢₖ_dLᵢₖ
				∑ᵢₖ_d²Qᵢₖ_dLᵢₖ²[t] += d²Qᵢₖ_dLᵢₖ²
				if (i==1) || (i==Ξ)
					∑_post_dQᵢₖ_dLᵢₖ⨀dξᵢ_dB[k][t] += dQᵢₖ_dLᵢₖ*d𝛏_dB[i]
					∑_post_d²Qᵢₖ_dLᵢₖ²⨀dξᵢ_dB[k][t] += d²Qᵢₖ_dLᵢₖ²*d𝛏_dB[i]
					∑_post_d²Qᵢₖ_dLᵢₖ²⨀dξᵢ_dB²[k][t] += d²Qᵢₖ_dLᵢₖ²*d𝛏_dB²[i]
				else
					∑_pre_dQᵢₖ_dLᵢₖ⨀dξᵢ_dB[k][t] += dQᵢₖ_dLᵢₖ*d𝛏_dB[i]
					∑_pre_d²Qᵢₖ_dLᵢₖ²⨀dξᵢ_dB[k][t] += d²Qᵢₖ_dLᵢₖ²*d𝛏_dB[i]
					∑_pre_d²Qᵢₖ_dLᵢₖ²⨀dξᵢ_dB²[k][t] += d²Qᵢₖ_dLᵢₖ²*d𝛏_dB²[i]
				end
			end
		end
	end
	n𝐮 = length(𝐮)
	n𝐯 = length(𝐯[1])
	indices𝐮 = 1:n𝐮
	indices𝐯 = collect(indices𝐮[end] .+ ((k-1)*n𝐯+1 : k*n𝐯) for k = 1:K)
	indices𝛃 = collect(indices𝐯[end][end] .+ ((k-1)*n𝐯+1 : k*n𝐯) for k = 1:K)
	indexa = 1 + (fit_𝛃 ? indices𝛃[end][end] : indices𝐯[end][end])
	𝐔 = @view 𝐗[:, 1:n𝐮]
	𝐔ᵀ, 𝐕ᵀ = transpose(𝐔), transpose(𝐕)
	∇Q[indices𝐮] .= 𝐔ᵀ*∑ᵢₖ_dQᵢₖ_dLᵢₖ
	∇∇Q[indices𝐮, indices𝐮] .= 𝐔ᵀ*(∑ᵢₖ_d²Qᵢₖ_dLᵢₖ².*𝐔)
	if fit_𝛃
		@inbounds for k = 1:K
			∇Q[indices𝐯[k]] .= 𝐕ᵀ*∑_pre_dQᵢₖ_dLᵢₖ⨀dξᵢ_dB[k]
			∇Q[indices𝛃[k]] .= 𝐕ᵀ*∑_post_dQᵢₖ_dLᵢₖ⨀dξᵢ_dB[k]
			∇∇Q[indices𝐮, indices𝐯[k]] .= 𝐔ᵀ*(∑_pre_d²Qᵢₖ_dLᵢₖ²⨀dξᵢ_dB[k].*𝐕)
			∇∇Q[indices𝐮, indices𝛃[k]] .= 𝐔ᵀ*(∑_post_d²Qᵢₖ_dLᵢₖ²⨀dξᵢ_dB[k].*𝐕)
			∇∇Q[indices𝐯[k], indices𝐯[k]] .= 𝐕ᵀ*(∑_pre_d²Qᵢₖ_dLᵢₖ²⨀dξᵢ_dB²[k].*𝐕)
			∇∇Q[indices𝛃[k], indices𝛃[k]] .= 𝐕ᵀ*(∑_post_d²Qᵢₖ_dLᵢₖ²⨀dξᵢ_dB²[k].*𝐕)
		end
	else
		@inbounds for k = 1:K
			∑ᵢ_dQᵢₖ_dLᵢₖ⨀dξᵢ_dB = ∑_pre_dQᵢₖ_dLᵢₖ⨀dξᵢ_dB[k] + ∑_post_dQᵢₖ_dLᵢₖ⨀dξᵢ_dB[k]
			∑ᵢ_d²Qᵢₖ_dLᵢₖ²⨀dξᵢ_dB = ∑_pre_d²Qᵢₖ_dLᵢₖ²⨀dξᵢ_dB[k] + ∑_post_d²Qᵢₖ_dLᵢₖ²⨀dξᵢ_dB[k]
			∑ᵢ_d²Qᵢₖ_dLᵢₖ²⨀dξᵢ_dB² = ∑_pre_d²Qᵢₖ_dLᵢₖ²⨀dξᵢ_dB²[k] + ∑_post_d²Qᵢₖ_dLᵢₖ²⨀dξᵢ_dB²[k]
			∇Q[indices𝐯[k]] .= 𝐕ᵀ*∑ᵢ_dQᵢₖ_dLᵢₖ⨀dξᵢ_dB
			∇∇Q[indices𝐮, indices𝐯[k]] .= 𝐔ᵀ*(∑ᵢ_d²Qᵢₖ_dLᵢₖ²⨀dξᵢ_dB.*𝐕)
			∇∇Q[indices𝐯[k], indices𝐯[k]] .= 𝐕ᵀ*(∑ᵢ_d²Qᵢₖ_dLᵢₖ²⨀dξᵢ_dB².*𝐕)
		end
	end
	if fit_overdispersion
		∇Q[indexa] = ∑_dQ_da
		∇∇Q[indices𝐮, indexa] = 𝐔ᵀ*∑ᵢₖ_d²Qᵢₖ_dadLᵢₖ
		if fit_𝛃
			@inbounds for k = 1:K
				∇∇Q[indices𝐯[k], indexa] .= 𝐕ᵀ*∑_pre_d²Qᵢₖ_dadLᵢₖ⨀dξᵢ_dB[k]
				∇∇Q[indices𝛃[k], indexa] .= 𝐕ᵀ*∑_post_d²Qᵢₖ_dadLᵢₖ⨀dξᵢ_dB[k]
			end
		else
			@inbounds for k = 1:K
				∑ᵢ_d²Qᵢₖ_dadLᵢₖ⨀dξᵢ_dB = ∑_pre_d²Qᵢₖ_dadLᵢₖ⨀dξᵢ_dB[k] + ∑_post_d²Qᵢₖ_dadLᵢₖ⨀dξᵢ_dB[k]
				∇∇Q[indices𝐯[k], indexa] .= 𝐕ᵀ*∑ᵢ_d²Qᵢₖ_dadLᵢₖ⨀dξᵢ_dB
			end
		end
		∇∇Q[indexa, indexa] = ∑_d²Q_da²
	end
	for i = 1:size(∇∇Q,1)
		for j = i+1:size(∇∇Q,2)
			∇∇Q[j,i] = ∇∇Q[i,j]
		end
	end
	return nothing
end
