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
	P = update!(memory, model, concatenateparameters(model))
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
function maximize_expectation_of_loglikelihood!(mpGLM::MixturePoissonGLM, γ::Vector{<:Vector{<:Real}}; show_trace::Bool=false, iterations::Integer=20)
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
function negexpectation_of_∇∇loglikelihood!(∇∇::Matrix{<:Real}, mpGLM::MixturePoissonGLM, Q::Vector{<:Real}, ∇Q::Vector{<:Real}, ∇∇Q::Matrix{<:Real}, γ::Vector{<:Vector{<:Real}}, x::Vector{<:Real})
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
function negexpectation_of_loglikelihood!(mpGLM::MixturePoissonGLM, Q::Vector{<:Real}, ∇Q::Vector{<:Real}, ∇∇Q::Matrix{<:Real}, γ::Vector{<:Vector{<:Real}}, x::Vector{<:Real})
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
function negexpectation_of_∇loglikelihood!(∇::Vector{<:Real}, mpGLM::MixturePoissonGLM, Q::Vector{<:Real}, ∇Q::Vector{<:Real}, ∇∇Q::Matrix{<:Real}, γ::Vector{<:Vector{<:Real}}, x::Vector{<:Real})
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
function expectation_of_∇∇loglikelihood!(Q::Vector{<:type}, ∇Q::Vector{<:type}, ∇∇Q::Matrix{<:type}, γ::Vector{<:Vector{<:type}}, mpGLM::MixturePoissonGLM) where {type<:AbstractFloat}
    @unpack Δt, 𝐗, 𝐲, index0, Ξ = mpGLM
	@unpack 𝐮, v, β, fit_β = mpGLM.θ
	𝛚 = transformaccumulator(mpGLM)
	Q[1] = 0.0
	∇Q .= 0.0
	∇∇Q .= 0.0
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
				d²ℓ₁_dL₁² = d²ℓ₀_dL₀²
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
			x = (f₁π₁*dℓ₁_dL₁ + f₀π₀*dℓ₀_dL₀)/f
			dℓ_d𝐮 = x.*𝐗[t,:]
			for j = 1:n𝐮
				∇Q[j] += γ[i][t]*dℓ_d𝐮[j]
			end
			indexwₐ = n𝐮 + ((fit_β && ((i==1) || (i==Ξ))) ? 2 : 1)
			dL₁_dwₐ = 𝛚[i]*𝐗[t,end]
			dℓ_dwₐ = dL₁_dwₐ*f₁π₁*dℓ₁_dL₁/f
			∇Q[indexwₐ] += γ[i][t]*dℓ_dwₐ
			temp1 = (f₁π₁*(dℓ₁_dL₁^2+d²ℓ₁_dL₁²) + f₀π₀*(dℓ₀_dL₀^2+d²ℓ₀_dL₀²))/f
			temp2 = f₁π₁*(dℓ₁_dL₁^2+d²ℓ₁_dL₁²)/f
			for j = 1:n𝐮
				for k = j:n𝐮
					∇∇Q[j,k] += γ[i][t]*(temp1*𝐗[t,j]*𝐗[t,k] - dℓ_d𝐮[j]*dℓ_d𝐮[k])
				end
				∇∇Q[j,indexwₐ] += γ[i][t]*(temp2*𝐗[t,j]*dL₁_dwₐ - dℓ_d𝐮[j]*dℓ_dwₐ)
			end
			∇∇Q[indexwₐ,indexwₐ] += γ[i][t]*(temp2*dL₁_dwₐ^2 - dℓ_dwₐ^2)
		end
	end
	for i = 1:size(∇∇Q,1)
		for j = i+1:size(∇∇Q,2)
			∇∇Q[j,i] = ∇∇Q[i,j]
		end
	end
	return nothing
end
