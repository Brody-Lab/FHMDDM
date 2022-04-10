"""
	∇∇loglikelihood(model)

Hessian of the log-likelihood of data under a factorial hidden Markov drift-diffusion model (FHMDDM).

The gradient and the log-likelihood are also returned

ARGUMENT
-`model`: a structure containing the data, parameters, and hyperparameters of an FHMDDM

RETURN
-`∇∇ℓ`: Hessian matrix of the log-likelihood
-`∇ℓ`: gradient of the log-likelihood
-`ℓ`: log-likelihood
"""
function ∇∇loglikelihood(model::Model)
	trialinvariant = Trialinvariant(model; purpose="Hessian")
	output =map(model.trialsets) do trialset
				glmθs = collect(trialset.mpGLMs[n].θ for n = 1:length(trialset.mpGLMs))
		 		pmap(trialset.trials) do trial
					∇∇loglikelihood(glmθs, model.θnative, trial, trialinvariant)
				end
			end
	∇∇ℓ = output[1][1][3]
	∇ℓ = output[1][1][2]
	ℓ = output[1][1][1]
	for i in eachindex(output)
		for m = 2:length(output[i])
			∇∇ℓ .+= output[i][m][1]
			∇ℓ .+= output[i][m][2]
			ℓ += output[i][m][3]
		end
	end
	return ∇∇ℓ, ∇ℓ, ℓ
end

"""
	∇∇loglikelihood(glmθs, θnative, trial, trialinvariant)

Hessian of the log-likelihood of the observations from one trial

The gradient and the log-likelihood are also returned

ARGUMENT
-`glmθs`: a vector whose each element is a structure containing the parameters of of the generalized linear model of a neuron
-`θnative`: a structure containing parameters specifying the latent variables in their native space
-`trial`: a structure containing information on the sensory stimuli, spike trains, input to each neuron's GLM, and behavioral choice
-`trialinvariant`: a structure containing quantities used in each trial

RETURN
-`∇∇ℓ`: Hessian matrix of the log-likelihood
-`∇ℓ`: gradient of the log-likelihood
-`ℓ`: log-likelihood
"""
function ∇∇loglikelihood(glmθs::Vector{<:GLMθ},
						 θnative::Latentθ,
						 trial::Trial,
						 trialinvariant::Trialinvariant)
	@unpack clicks = trial
	@unpack ∇∇Aᵃsilent, ∇Aᵃsilent, Aᵃsilent, Aᶜ, Aᶜᵀ, Δt, K, 𝛚, πᶜᵀ, Ξ, 𝛏 = trialinvariant # need second derivative of the silent transition matrix and silent prior probability (without including `previousreward`)

	P = Probabilityvector(Δt, θnative, Ξ)
	∇∇𝛑 = map(i->zeros(Ξ), CartesianIndices((4,4)))
 	∇𝛑 = map(i->zeros(Ξ), 1:4)
	∇∇priorprobability!(∇∇𝛑, ∇𝛑, P, trial.previousreward)

	# do stuff with P

	if !isempty(clicks.inputtimesteps)
		adaptedclicks = ∇∇adapt(clicks, θnative.k[1], θnative.ϕ[1])
		∇∇Aᵃinput = map(i->zeros(Ξ,Ξ), CartesianIndices((6,6)))
		∇Aᵃinput = map(i->zeros(Ξ,Ξ), 1:6)
		Aᵃinput = zeros(Ξ,Ξ)
		Aᵃinput[1,1] = Aᵃinput[Ξ, Ξ] = 1.0
	end
	for t =1:trial.ntimesteps
		if t ∈ clicks.inputtimesteps
			update_for_∇∇transition_probabilities!(P, adaptedclicks, clicks, t)
			∇∇transitionmatrix!(∇∇Aᵃinput, ∇Aᵃinput, Aᵃinput, P, t)
			∇∇Aᵃ = ∇∇Aᵃinput
			∇Aᵃ = ∇Aᵃinput
			Aᵃ = Aᵃinput
		else
			∇∇Aᵃ = ∇∇Aᵃsilent
			∇Aᵃ = ∇Aᵃsilent
			Aᵃ = Aᵃsilent
		end
		∇∇conditionallikelihood!(∇∇pY, ∇pY, pY, glmθs, t, trial.spiketrainmodels, trialinvariant)
		# then, compute ∇∇ℓ, ∇ℓ, and ℓ using ∇∇p𝐘, ∇p𝐘, p𝐘, ∇∇Aᵃ, ∇Aᵃ, Aᵃ
	end

	return ∇∇ℓ, ∇ℓ, ℓ
end
