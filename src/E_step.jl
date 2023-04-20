"""
	forward!(memory, P, θnative, trial)

Compute the forward terms and the log-likelihood

MODIFIED ARGUMENT
-`memory`: memory allocated for computing the gradient. The log-likelihood is updated.
-`P`: a structure containing allocated memory for computing the accumulator's initial and transition probabilities as well as the partial derivatives of these probabilities

UNMODIFIED ARGUMENT
-`θnative`: accumulation parameters in native space
-`trial`: an object containing the data of one trial
"""
function forward!(memory::Memoryforgradient, P::Probabilityvector, θnative::Latentθ, trial::Trial)
	@unpack clicks, index_in_trialset, τ₀, trialsetindex = trial
	@unpack Aᵃinput, Aᵃsilent, choiceLLscaling, D, f, ℓ, p𝐚₁, Ξ = memory
	γ = memory.γ[trialsetindex]
	p𝐘𝑑 = memory.p𝐘𝑑[trialsetindex][index_in_trialset]
	accumulator_prior_transitions!(Aᵃinput, P, p𝐚₁, trial)
	p𝐚 = p𝐚₁
	t = 1
	@inbounds for i=1:Ξ
		f[t][i] = p𝐘𝑑[t][i] * p𝐚[i]
	end
	D[t] = sum(f[t])
	D[t] = max(D[t], nextfloat(0.0))
	f[t] ./= D[t]
	ℓ[1] += log(D[t])
	@inbounds for t=2:trial.ntimesteps
		Aᵃ = isempty(clicks.inputindex[t]) ? Aᵃsilent : Aᵃinput[clicks.inputindex[t][1]]
		f[t] = p𝐘𝑑[t] .* (Aᵃ * f[t-1])
		D[t] = sum(f[t])
		D[t] = max(D[t], nextfloat(0.0))
		f[t] ./= D[t]
		ℓ[1] += log(D[t])
		if choiceLLscaling > 1
			p𝐚 = Aᵃ*p𝐚
		end
	end
	if choiceLLscaling > 1
		p𝑑_a = ones(Ξ)
		conditionallikelihood!(p𝑑_a, trial.choice, θnative.ψ[1])
		ℓ[1] += (choiceLLscaling-1)*log(dot(p𝑑_a, p𝐚))
	end
	return nothing
end

"""
	backward!(memory, P, trial)

Compute the posterior probability of the latent variables

MODIFIED ARGUMENT
-`γ`: posterior probability of the latent variables on each timestep in a trialset. Element `γ[i,j][τ]` corresponds to the posterior probability of the accumulator in the i-th state and the coupling variable in the j-th state in the τ-th timestep in the trialset.
-`memory`: memory allocated for computing the gradient. The log-likelihood is updated.
-`P`: a structure containing allocated memory for computing the accumulator's initial and transition probabilities as well as the partial derivatives of these probabilities

UNMODIFIED ARGUMENT
-`p𝐘𝑑`: conditional likelihood of the emissions in one trial. Element `p𝐘𝑑[t][i,j]` corresponds to t-th time step of the trial, i-th accumulator state, and j-th coupling state
-`τ₀`: number of time steps summed across all preceding trials in the trialset
-`trial`: structure containing information regarding one trial
"""
function backward!(memory::Memoryforgradient, P::Probabilityvector, trial::Trial)
	@unpack clicks, index_in_trialset, τ₀, trialsetindex = trial
	@unpack Aᵃinput, Aᵃsilent, D, Ξ = memory
	γ = memory.γ[trialsetindex]
	p𝐘𝑑 = memory.p𝐘𝑑[trialsetindex][index_in_trialset]
	f⨀b = memory.f # reuse memory
	b = ones(Ξ)
	@inbounds for t = trial.ntimesteps-1:-1:1
		if t+1 ∈ clicks.inputtimesteps
			clickindex = clicks.inputindex[t+1][1]
			Aᵃₜ₊₁ = Aᵃinput[clickindex]
		else
			Aᵃₜ₊₁ = Aᵃsilent
		end
		b = transpose(Aᵃₜ₊₁) * (b.*p𝐘𝑑[t+1]./D[t+1])
		f⨀b[t] .*= b
	end
	for t = 1:trial.ntimesteps
		τ = τ₀+t
		for i = 1:Ξ
			γ[i][τ] = f⨀b[t][i]
		end
	end
	return nothing
end

"""
    scaledlikelihood!(p𝐘𝑑, trialset, ψ)

Update the conditional likelihood of the emissions (spikes and/or behavioral choice)

MODIFIED ARGUMENT
-`p𝐘𝑑`: Conditional likelihood of the emissions (spikes and/or choice) at each time bin. For time bins of each trial other than the last, it is the product of the conditional likelihood of all spike trains. For the last time bin, it corresponds to the product of the conditional likelihood of the spike trains and the choice. Element p𝐘𝑑[i][m][t][j] corresponds to ∏ₙᴺ p(𝐲ₙ(t) | aₜ = ξⱼ) across N neural units at the t-th time bin in the m-th trial of the i-th trialset. The last element p𝐘𝑑[i][m][end][j] of each trial corresponds to p(𝑑 | aₜ = ξⱼ) ∏ₙᴺ p(𝐲ₙ(t) | aₜ = ξⱼ)
-`p𝑑_a`: a vector used for in-place computation of the conditional likelihood of the choice given 𝑎 for one trial

UNMODIFIED ARGUMENT
-`s`: scale factor of the conditional likelihood of the spike train
-`trialsets`: data used to constrain the model
-`ψ`: lapse rate

RETURN
-`nothing`
"""
function scaledlikelihood!(p𝐘𝑑::Vector{<:Vector{<:Vector{<:Vector{<:Real}}}}, trialsets::Vector{<:Trialset}, ψ::Real)
    @inbounds for (p𝐘𝑑, trialset) in zip(p𝐘𝑑, trialsets)
		scaledlikelihood!(p𝐘𝑑, trialset.mpGLMs)
		for (p𝐘𝑑, trial) in zip(p𝐘𝑑, trialset.trials)
			conditionallikelihood!(p𝐘𝑑[trial.ntimesteps], trial.choice, ψ)
		end
    end
    return nothing
end

"""
	posteriors!(memory, P, model)

Compute the posterior distribution of the latent variables

MODIFIED ARGUMENT
-`memory`: structure containing the memory for computing the gradient of the log-likelihood of the model
-`P`: a structure for computing the derivatives with respect to the drift-diffusion parameters

UNMODIFIED ARGUMENT
-`model`: structure containing the parameters, data, and hyperparameters
"""
function posteriors!(memory::Memoryforgradient, P::Probabilityvector, model::Model)
	@inbounds for trialset in model.trialsets
		for trial in trialset.trials
			forward!(memory, P, model.θnative, trial)
			backward!(memory, P, trial)
		end
	end
	return nothing
end

"""
	posteriors(model)
"""
function posteriors(model::Model)
	memory = Memoryforgradient(model)
	posteriors(memory, model)
	return memory.γ
end

"""
	posteriors!(memory, model)
"""
function posteriors(memory::Memoryforgradient, model::Model)
	P = update!(memory, model, concatenateparameters(model))
	posteriors!(memory, P, model)
	return memory.γ
end

"""
	accumulator_prior_transitions!(Aᵃinput, P, p𝐚₁, trial)

Update the prior distribution and transition matrices of the accumulator for one trial

MODIFIED ARGUMENT
-`Aᵃinput`: nested array whose element `Aᵃinput[i][j,k]` represents the transition probability from state k to state j for the i-th time step during which auditory clicks occured
-`P`: a composite used to compute the probability vector of the accumulator
-`p𝐚₁`: vector whose element `p𝐚₁[i]` corresponds to the prior probability of the accumulator in the i-th state

UNMODIFIED ARGUMENT

"""
function accumulator_prior_transitions!(Aᵃinput::Vector{<:Matrix{<:AbstractFloat}},
										P::Probabilityvector,
										p𝐚₁::Vector{<:AbstractFloat},
										trial::Trial)
	adaptedclicks = adapt(trial.clicks, P.k, P.ϕ)
	priorprobability!(P, trial.previousanswer)
	p𝐚₁ .= P.𝛑
	@inbounds for t=2:trial.ntimesteps
		if !isempty(trial.clicks.inputindex[t])
			update_for_transition_probabilities!(P, adaptedclicks, trial.clicks, t)
			transitionmatrix!(Aᵃinput[trial.clicks.inputindex[t][1]], P)
		end
	end
	return nothing
end

"""
	choiceposteriors!(memory, model)

Posterior probability of the latent variable conditioned on only the choice and not the spikes

MODIFIED ARGUMENT
-`memory`: structure containing the memory for computing the gradient of the log-likelihood of the model. In particular, the field `γ` corresponds to the single-time-step posterior probabilities of the accumulator and coupling variables at each time step conditioned on the choices (𝑑). Element `γ[i][j,k][t]` represent the posterior probability at the t-th timestep in the i-th trialset: p(aₜⱼ=1, cₜₖ=1 ∣ 𝑑)

ARGUMENT
-`model`: custom type containing the settings, data, and parameters of a factorial hidden Markov drift-diffusion model

RETURN
-`P`: a structure for computing the derivatives with respect to the drift-diffusion parameters, in case it is to be reused
"""
function choiceposteriors!(memory::Memoryforgradient, model::Model)
	for (p𝐘𝑑, trialset) in zip(memory.p𝐘𝑑, model.trialsets)
		for (p𝐘𝑑, trial) in zip(p𝐘𝑑, trialset.trials)
			conditionallikelihood!(p𝐘𝑑[trial.ntimesteps], trial.choice, model.θnative.ψ[1])
		end
	end
	P = update_for_latent_dynamics!(memory, model.options, model.θnative)
	posteriors!(memory, P, model)
	return P
end

"""
	posterior_on_spikes!(memory, model)

Posterior probability of the latent variable conditioned on only the spiking and not the choice

MODIFIED ARGUMENT
-`memory`: structure containing variables memory between computations of the model's log-likelihood and its gradient

UNMODIFIED ARGUMENT
-`model`: structure containing the data, parameters, and hyperparameters
"""
function posterior_on_spikes!(memory::Memoryforgradient, model::Model)
	for (p𝐘,trialset) in zip(memory.p𝐘𝑑, model.trialsets)
		scaledlikelihood!(p𝐘, trialset.mpGLMs)
	end
	P = update_for_latent_dynamics!(memory, model.options, model.θnative)
	posteriors!(memory, P, model)
	return nothing
end

"""
	update_for_latent_dynamics!(memory, options, θnative)

Update quantities for computing the prior and transition probabilities of the latent variables

MODIFIED ARGUMENT
-`memory`: structure containing variables memory between computations of the model's log-likelihood and its gradient

UNMODIFIED ARGUMENT
-`options`: settings of the model
-`θnative`: values of the parameters that control the latent variables, in the parameters' native space

RETURN
-`P`: an instance of `Probabilityvector`
"""
function update_for_latent_dynamics!(memory::Memoryforgradient, options::Options, θnative::Latentθ)
	P = Probabilityvector(options.Δt, options.minpa, θnative, options.Ξ)
	update_for_transition_probabilities!(P)
	transitionmatrix!(memory.Aᵃsilent, P)
	return P
end
