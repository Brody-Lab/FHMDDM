"""
	Set of functions compatible with ReverseDiff.jl and ForwardDiff.jl
"""

"""
    choiceloglikelihood!(model, concatenatedθ)

Compute the log-likelihood of the choices

ARGUMENT
-`model`: an instance of FHM-DDM
-`concatenatedθ`: a vector of concatenated parameter values
-`indexθ`: index of each parameter after if all parameters being fitted are concatenated

RETURN
-log-likelihood
"""
function loglikelihoodchoices(concatenatedθ::Vector{<:Real},
							  indexθ::Indexθ,
							  model::Model)
	model = sortparameters(concatenatedθ, indexθ, model)
	trialinvariant = Trialinvariant(model; purpose="loglikelihood")
	ℓ = map(model.trialsets) do trialset
			map(trialset.trials) do trial #pmap
				loglikelihood(model.θnative, trial, trialinvariant)
			end
		end
	return sum(sum(ℓ))
end

"""
    loglikelihood(concatenatedθ, indexθ, model)

Computation the log-likelihood meant for automatic differentiation

ARGUMENT
-`model`: an instance of FHM-DDM
-`p𝐘𝑑`: a nested matrix representing the conditional likelihood of the emissions.

UNMODIFIED ARGUMENT
-`concatenatedθ`: a vector of concatenated parameter values
-`indexθ`: struct indexing of each parameter in the vector of concatenated values

RETURN
-log-likelihood
"""
function loglikelihood(	concatenatedθ,
					    indexθ::Indexθ,
						model::Model)
	model = sortparameters(concatenatedθ, indexθ, model)
	@unpack options, θnative, θreal, trialsets = model
	@unpack Ξ, K = options
	trialinvariant = Trialinvariant(model; purpose="loglikelihood")
	T = eltype(concatenatedθ)
	p𝐘𝑑=map(model.trialsets) do trialset
			map(trialset.trials) do trial
				map(1:trial.ntimesteps) do t
					ones(T,Ξ,K)
				end
			end
		end
    likelihood!(p𝐘𝑑, trialsets, θnative.ψ[1]) # `p𝐘𝑑` is the conditional likelihood p(𝐘ₜ, d ∣ aₜ, zₜ)
	ℓ = map(trialsets, p𝐘𝑑) do trialset, p𝐘𝑑
			pmap(trialset.trials, p𝐘𝑑) do trial, p𝐘𝑑
				loglikelihood(p𝐘𝑑, θnative, trial, trialinvariant)
			end
		end
	return sum(sum(ℓ))
end

"""
	sortparameters(concatenatedθ, indexθ, model)

Sort a vector of concatenated parameter values

UNMODIFIED ARGUMENT
-`concatenatedθ`: a vector of concatenated parameter values
-`indexθ`: struct indexing of each parameter in the vector of concatenated values
-`model`: the model with old parameter values

RETURN
-`model`: the model with new parameter values
"""
function sortparameters(concatenatedθ,
				 		indexθ::Indexθ,
						model::Model)
	T = eltype(concatenatedθ)
	θreal = Latentθ((zeros(T,1) for field in fieldnames(Latentθ))...)
	for field in fieldnames(Latentθ) # `Latentθ` is the type of `indexθ.latentθ`
		index = getfield(indexθ.latentθ, field)[1]
		if index != 0 # an index of 0 indicates that the parameter is not being fit
			getfield(θreal, field)[1] = concatenatedθ[index]
		end
	end
	trialsets = map(model.trialsets, indexθ.𝐮, indexθ.𝐥, indexθ.𝐫) do trialset, index𝐮, index𝐥, index𝐫
					mpGLMs =map(trialset.mpGLMs, index𝐮, index𝐥, index𝐫) do mpGLM, index𝐮, index𝐥, index𝐫
						MixturePoissonGLM(	Δt=mpGLM.Δt,
											K=mpGLM.K,
											𝚽=mpGLM.𝚽,
											Φ=mpGLM.Φ,
											𝐔=mpGLM.𝐔,
											𝐗=mpGLM.𝐗,
											𝛏=mpGLM.𝛏,
											𝐲=mpGLM.𝐲,
											𝐮=isempty(index𝐮) ? mpGLM.𝐮 : concatenatedθ[index𝐮],
											𝐥=isempty(index𝐥) ? mpGLM.𝐥 : concatenatedθ[index𝐥],
											𝐫=isempty(index𝐫) ? mpGLM.𝐫 : concatenatedθ[index𝐫])
					end
					Trialset(mpGLMs=mpGLMs, trials=trialset.trials)
				end
	Model(	options = model.options,
			θnative = real2native(model.options, θreal),
			θ₀native=model.θ₀native,
			θreal = θreal,
			trialsets=trialsets)
end
