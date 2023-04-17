"""
	GLMθ(indices𝐮, options, n𝐯)

Randomly initiate the parameters for a mixture of Poisson generalized linear model

ARGUMENT
-`indices𝐮`: indices of the encoding weights of the temporal basis vectors of each filter that is independent of the accumulator
-`options`: settings of the model
-`v_scalefactor`: scalefactor of the accumulator encoding weight

OUTPUT
-an instance of `GLMθ`
"""
function GLMθ(indices𝐮::Indices𝐮, options::Options)
	n𝐮 = maximum(vcat((getfield(indices𝐮, field) for field in fieldnames(Indices𝐮))...))
	θ = GLMθ(fit_b = options.fit_b,
			 fit_β = options.fit_β,
		 	 𝐮 = fill(NaN, n𝐮),
			 indices𝐮=indices𝐮)
	randomizeparameters!(θ, options)
	return θ
end

"""
	randomizeparameters!(glmθ, options)

Randomly initialize parameters of a mixture of Poisson GLM

MODIFIED ARGUMENT
-`θ`: structure containing parameters of a mixture of Poisson GLM

UNMODIFIED ARGUMENT
-`options`: hyperparameters of the model
"""
function randomizeparameters!(glmθ::GLMθ, options::Options)
	glmθ.b[1] = 0.0
	for i in eachindex(glmθ.𝐮)
		glmθ.𝐮[i] = 1.0 - 2rand()
	end
	for fieldname in fieldnames(typeof(glmθ.indices𝐮))
		indices = getfield(glmθ.indices𝐮, fieldname)
		scalefactor = getfield(options, Symbol("tbf_"*String(fieldname)*"_scalefactor"))*options.sf_tbf[1]
		glmθ.𝐮[indices] ./= scalefactor
	end
    θ.v[1] = (1.0 - 2rand())/v_scalefactor
	if glmθ.fit_β
		glmθ.β[1] = -θ.v[1]
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
	Q = fill(NaN,1)
	∇Q = fill(NaN, nparameters)
	∇∇Q = fill(NaN, nparameters, nparameters)
	f(x) = negexpectation_of_loglikelihood!(mpGLM,Q,∇Q,∇∇Q,γ,x)
	∇f!(∇, x) = negexpectation_of_∇loglikelihood!(∇,mpGLM,Q,∇Q,∇∇Q,γ,x)
	∇∇f!(∇∇, x) = negexpectation_of_∇∇loglikelihood!(∇∇,mpGLM,Q,∇Q,∇∇Q,γ,x)
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
-`∇Q`: gradient of the expectation with respect to the filters in the k-th state
-`∇∇Q`: Hessian of the expectation with respect to the filters in the k-th state

UNMODIFIED ARGUMENT
-`γ`: posterior probability of the latent variables. Element `γ[j][τ]` corresponds to the posterior probability of the j-th accumulator state  in the τ-th time step
-`x`: filters
"""
function negexpectation_of_∇∇loglikelihood!(∇∇::Matrix{<:Real}, mpGLM::MixturePoissonGLM, Q::Vector{<:Real}, ∇Q::Vector{<:Real}, ∇∇Q::Matrix{<:Real}, γ::Matrix{<:Vector{<:Real}}, x::Vector{<:Real})
	x₀ = concatenateparameters(mpGLM.θ; initialization=true)
	if (x != x₀) || isnan(Q[1])
		sortparameters!(mpGLM.θ, x; initialization=true)
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
	negexpectation_of_loglikelihood!(mpGLM,D,Q,∇Q,∇∇Q,γ,x)

Negative expectation of the log-likelihood under the posterior probability of the latent variables

MODIFIED ARGUMENT
-`∇`: gradient of the negative of the expectation

For other modified and unmodified arguments see documentation for `negexpectation_of_∇∇loglikelihood!(∇∇,mpGLM,D,Q,∇Q,∇∇Q,γ,x)`

RETURN
-a scalar that is the negative of the expectation of the log-likelihood under the posterior probability distribution
"""
function negexpectation_of_loglikelihood!(mpGLM::MixturePoissonGLM, Q::Vector{<:Real}, ∇Q::Vector{<:Real}, ∇∇Q::Matrix{<:Real}, γ::Matrix{<:Vector{<:Real}}, x::Vector{<:Real})
	x₀ = concatenateparameters(mpGLM.θ; initialization=true)
	if (x != x₀) || isnan(Q[1])
		sortparameters!(mpGLM.θ, x; initialization=true)
		expectation_of_∇∇loglikelihood!(Q,∇Q,∇∇Q,γ,mpGLM)
	end
	-Q[1]
end

"""
	negexpectation_of_∇loglikelihood!(∇,mpGLM,D,Q,∇Q,∇∇Q,γ,x)

Gradient of the negative of the expectation of the log-likelihood under the posterior probability of the latent variables

ARGUMENT
For other modified and unmodified arguments see documentation for `negexpectation_of_∇∇loglikelihood!(∇∇,mpGLM,D,Q,∇Q,∇∇Q,γ,x)`
"""
function negexpectation_of_∇loglikelihood!(∇::Vector{<:Real}, mpGLM::MixturePoissonGLM, Q::Vector{<:Real}, ∇Q::Vector{<:Real}, ∇∇Q::Matrix{<:Real}, γ::Matrix{<:Vector{<:Real}}, x::Vector{<:Real})
	x₀ = concatenateparameters(mpGLM.θ; initialization=true)
	if (x != x₀) || isnan(Q[1])
		sortparameters!(mpGLM.θ, x; initialization=true)
		expectation_of_∇∇loglikelihood!(Q,∇Q,∇∇Q,γ,mpGLM)
	end
	for i in eachindex(∇)
		∇[i] = -∇Q[i]
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
"""
function expectation_of_∇∇loglikelihood!(Q::Vector{<:type}, ∇Q::Vector{<:type}, ∇∇Q::Matrix{<:type}, γ::Matrix{<:Vector{<:type}}, mpGLM::MixturePoissonGLM) where {type<:AbstractFloat}
    @unpack Δt, 𝐗, 𝐲, d𝛏_dB, index0, Ξ = mpGLM
	@unpack 𝐮, v, β, fit_β = mpGLM.θ
	d𝛏_dB² = d𝛏_dB.^2
	Q[1] = 0.0
	∇Q .= 0.0
	∇∇Q .= 0.0
	for parameter in fieldnames(mpGLM.θ.concatenationorder)
		getfield(∇Q, parameter) .= 0
	end
	𝐋₀ = linearpredictor(mpGLM,index0)
	𝛌₀ = inverselink.(𝐋₀)
	𝐟₀ = collect(poissonlikelihood(λ₀*Δt, y) for (λ₀,y) in zip(𝛌₀,𝐲))
	𝐃₀ = collect(differentiate_twice_loglikelihood_wrt_linearpredictor(Δt, L₀, λ₀, y) for (L₀, λ₀, y) in zip(𝐋₀,𝛌₀,𝐲))
	π₁ = couplingprobability(mpGLM)
	π₀ = 1-π₁
	n𝐮 = length(𝐮)
	@inbounds for i = 1:Ξ
		𝐋₁ = (i == index0) ? 𝐋₀ : linearpredictor(mpGLM,i)
		for t=1:length(𝐲)
			d²ℓ₀_dL₀² = 𝐃₀[t][1]
			dℓ₀_dL₀ = 𝐃₀[t][2]
			if i == index0
				d²ℓ₁_dL₁² = dℓ₀_dL₀
				dℓ₁_dL₁ = dℓ₀_dL₀
				f₁ = 𝐟₀[t]
			else
				λ₁ = inverselink(𝐋₁[t])
				d²ℓ₁_dL₁², dℓ₁_dL₁ = differentiate_twice_loglikelihood_wrt_linearpredictor(Δt, 𝐋₁[t], λ₁, 𝐲[t])
				f₁ = poissonlikelihood(λ₁*Δt, 𝐲[t])
			end
			f₁π₁ = f₁*π₁
			f₀π₀ = 𝐟₀[t]*π₀
			f = f₁π₁ + f₀π₀
			Q[1] += γ[i][t]*log(f)
			for j in eachindex(∇Q.𝐮)
				∇Q[j] += γ[i][t]*𝐗[t,j]*(f₁π₁*dℓ₁_dL₁ + f₀π₀*dℓ₀_dL₀)/f
			end
			if i != index0
				j = (fit_β && ((i==1) || (i==Ξ))) ? n𝐮+2 : n𝐮+1
				∇Q[j] += γ[i][t]*d𝛏_dB[i]*𝐗[t,end]*f₁π₁*dℓ₁_dL₁/f
			end
			for j in eachindex(∇Q.𝐮)
				for k = j:n𝐮

				end
			end


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
