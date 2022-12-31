"""
    scaledlikelihood!(p𝐘𝑑, trialset, ψ)

Update the conditional likelihood of the emissions (spikes and/or behavioral choice)

MODIFIED ARGUMENT
-`p𝐘𝑑`: Conditional likelihood of the emissions (spikes and/or choice) at each time bin. For time bins of each trial other than the last, it is the product of the conditional likelihood of all spike trains. For the last time bin, it corresponds to the product of the conditional likelihood of the spike trains and the choice. Element p𝐘𝑑[i][m][t][j,k] corresponds to ∏ₙᴺ p(𝐲ₙ(t) | aₜ = ξⱼ, zₜ=k) across N neural units at the t-th time bin in the m-th trial of the i-th trialset. The last element p𝐘𝑑[i][m][end][j,k] of each trial corresponds to p(𝑑 | aₜ = ξⱼ, zₜ=k) ∏ₙᴺ p(𝐲ₙ(t) | aₜ = ξⱼ, zₜ=k)
-`p𝑑_a`: a vector used for in-place computation of the conditional likelihood of the choice given 𝑎 for one trial

UNMODIFIED ARGUMENT
-`s`: scale factor of the conditional likelihood of the spike train
-`trialsets`: data used to constrain the model
-`ψ`: lapse rate

RETURN
-`nothing`
"""
function scaledlikelihood!(p𝐘𝑑::Vector{<:Vector{<:Vector{<:Matrix{<:Real}}}},
					 p𝑑_a::Vector{<:Vector{<:Vector{<:Real}}},
					 s::Real,
                     trialsets::Vector{<:Trialset},
                     ψ::Real)
	Ξ = size(p𝐘𝑑[1][1][end],1)
	K = size(p𝐘𝑑[1][1][end],2)
    @inbounds for i in eachindex(p𝐘𝑑)
		N = length(trialsets[i].mpGLMs)
	    for j = 1:Ξ
	        for k = 1:K
				𝐩 = scaledlikelihood(trialsets[i].mpGLMs[1], j, k, s)
	            for n = 2:N
				    scaledlikelihood!(𝐩, trialsets[i].mpGLMs[n], j, k, s)
	            end
	            t = 0
	            for m in eachindex(p𝐘𝑑[i])
	                for tₘ in eachindex(p𝐘𝑑[i][m])
	                    t += 1
	                    p𝐘𝑑[i][m][tₘ][j,k] = 𝐩[t]
	                end
	            end
	        end
	    end
		for m in eachindex(p𝐘𝑑[i])
			choicelikelihood!(p𝑑_a[i][m], trialsets[i].trials[m].choice, ψ)
			p𝐘𝑑[i][m][end] .*= p𝑑_a[i][m]
		end
    end
    return nothing
end

"""
    choicelikelihood!(p𝑑, choice, ψ)

Conditional likelihood of a right choice given the state of the accumulator

MODIFIED ARGUMENT
-`p𝑑`: A vector for in-place computation

UNMODIFIED ARGUMENT
-`choice`: the observed choice, either right (`choice`=true) or left.
-`ψ`: the prior probability of a lapse state
"""
function choicelikelihood!(p𝑑::Vector{<:Real}, choice::Bool, ψ::Real)
	zeroindex = cld(size(p𝑑,1),2)
    p𝑑[zeroindex] = 0.5
    if choice
        p𝑑[1:zeroindex-1] .= ψ/2
        p𝑑[zeroindex+1:end] .= 1-ψ/2
    else
        p𝑑[1:zeroindex-1]   .= 1-ψ/2
        p𝑑[zeroindex+1:end] .= ψ/2
    end
    return nothing
end

"""
	posteriors(model)

Compute the joint posteriors of the latent variables at each time step.

ARGUMENT
-`model`: custom type containing the settings, data, and parameters of a factorial hidden Markov drift-diffusion model

RETURN
-`γ`: joint posterior probabilities of the accumulator and coupling variables at each time step conditioned on the emissions at all time steps in the trialset. Element `γ[i][j,k][t]` represent the posterior probability at the t-th timestep in the i-th trialset: p(aₜⱼ=1, cₜₖ=1 ∣ 𝐘, 𝑑)
```
"""
function posteriors(model::Model)
	memory = Memoryforgradient(model)
	P = update!(memory, model)
	posteriors!(memory, P, model)
	return memory.γ
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
	forward!(memory, P, θnative, trial)

Compute the forward terms and the log-likelihood

MODIFIED ARGUMENT
-`memory`: memory allocated for computing the gradient. The log-likelihood is updated.
-`P`: a structure containing allocated memory for computing the accumulator's initial and transition probabilities as well as the partial derivatives of these probabilities

UNMODIFIED ARGUMENT
-`i`: triaset index
-`m`: trial index
-`model`: structure containing the parameters, hyperparameters, and settings of the model
"""
function forward!(memory::Memoryforgradient, P::Probabilityvector, θnative::Latentθ, trial::Trial)
	@unpack clicks, index_in_trialset, τ₀, trialsetindex = trial
	@unpack Aᵃinput, Aᵃsilent, Aᶜᵀ, D, f, K, ℓ, p𝐚₁, πᶜ, Ξ = memory
	γ = memory.γ[trialsetindex]
	p𝐘𝑑 = memory.p𝐘𝑑[trialsetindex][index_in_trialset]
	t = 1
	priorprobability!(P, trial.previousanswer)
	p𝐚₁ .= P.𝛑
	@inbounds for j=1:Ξ
		for k = 1:K
			f[t][j,k] = p𝐘𝑑[t][j,k] * p𝐚₁[j] * πᶜ[k]
		end
	end
	D[t] = sum(f[t])
	f[t] ./= D[t]
	ℓ[1] += log(D[t])
	if length(clicks.time) > 0
		adaptedclicks = adapt(clicks, θnative.k[1], θnative.ϕ[1])
	end
	@inbounds for t=2:trial.ntimesteps
		if t ∈ clicks.inputtimesteps
			clickindex = clicks.inputindex[t][1]
			Aᵃ = Aᵃinput[clickindex]
			update_for_transition_probabilities!(P, adaptedclicks, clicks, t)
			transitionmatrix!(Aᵃ, P)
		else
			Aᵃ = Aᵃsilent
		end
		f[t] = p𝐘𝑑[t] .* (Aᵃ * f[t-1] * Aᶜᵀ)
		D[t] = sum(f[t])
		f[t] ./= D[t]
		ℓ[1] += log(D[t])
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
	@unpack Aᵃinput, Aᵃsilent, Aᶜ, D, K, Ξ = memory
	γ = memory.γ[trialsetindex]
	p𝐘𝑑 = memory.p𝐘𝑑[trialsetindex][index_in_trialset]
	f⨀b = memory.f # reuse memory
	b = ones(Ξ,K)
	@inbounds for t = trial.ntimesteps-1:-1:1
		if t+1 ∈ clicks.inputtimesteps
			clickindex = clicks.inputindex[t+1][1]
			Aᵃₜ₊₁ = Aᵃinput[clickindex]
		else
			Aᵃₜ₊₁ = Aᵃsilent
		end
		b = transpose(Aᵃₜ₊₁) * (b.*p𝐘𝑑[t+1]./D[t+1]) * Aᶜ
		f⨀b[t] .*= b
	end
	for t = 1:trial.ntimesteps
		τ = τ₀+t
		for i = 1:Ξ
			for j = 1:K
				γ[i,j][τ] = f⨀b[t][i,j]
			end
		end
	end
	return nothing
end

"""
	joint_posteriors_of_coupling(model)

Sum the joint posteriors of the coupling variable at two consecutive time steps across time

ARGUMENT
-`model`: custom type containing the settings, data, and parameters of a factorial hidden Markov drift-diffusion model

RETURN
-`∑χ`: joint posterior probabilities of the coupling variables at two consecutive time step conditioned on the emissions at all time steps in the trialset. Element `∑χ[j,k]` represent p{c(t)=j, c(t-1)=k ∣ 𝐘, 𝑑) summed across trials and trialsets

"""
function joint_posteriors_of_coupling!(memory::Memoryforgradient, model::Model, ∑χ::Matrix{<:Real}, ∑γ::Vector{<:Real})
	P = update!(memory, model)
	memory.ℓ .= 0.0
	∑χ .= 0.0
	∑γ .= 0.0
	@inbounds for s in eachindex(model.trialsets)
		for m in eachindex(model.trialsets[s].trials)
			joint_posteriors_of_coupling!(memory, P, ∑χ, ∑γ, model, s, m)
		end
	end
	return nothing
end

"""
	joint_posteriors_of_coupling!(∑χ, memory, P, model, s, m)

Sum the joint posteriors of the coupling variable at two consecutive time steps across time

Update the gradient

MODIFIED ARGUMENT
-`memory`: memory allocated for computing the gradient. The log-likelihood is updated.
-`P`: a structure containing allocated memory for computing the accumulator's initial and transition probabilities as well as the partial derivatives of these probabilities
- `∑χ`: joint posterior probabilities of the coupling variables at two consecutive time step

UNMODIFIED ARGUMENT
-`model`: structure containing the data, parameters, and hyperparameters of the model
-`s`: index of the trialset
-`m`: index of the trial
"""
function joint_posteriors_of_coupling!(memory::Memoryforgradient,
					P::Probabilityvector,
					∑χ::Matrix{<:Real},
					∑γ::Vector{<:Real},
					model::Model,
					s::Integer,
					m::Integer)
	trial = model.trialsets[s].trials[m]
	p𝐘𝑑 = memory.p𝐘𝑑[s][m]
	@unpack θnative = model
	@unpack clicks = trial
	@unpack inputtimesteps, inputindex = clicks
	@unpack Aᵃinput, Aᵃsilent, Aᶜ, Aᶜᵀ, D, f, indexθ_pa₁, indexθ_paₜaₜ₋₁, indexθ_pc₁, indexθ_pcₜcₜ₋₁, indexθ_ψ, K, ℓ, ∇ℓlatent, nθ_pa₁, nθ_paₜaₜ₋₁, nθ_pc₁, nθ_pcₜcₜ₋₁, ∇pa₁, πᶜ, ∇πᶜ, Ξ = memory
	t = 1
	priorprobability!(P, trial.previousanswer)
	@inbounds for j=1:Ξ
		for k = 1:K
			f[t][j,k] = p𝐘𝑑[t][j,k] * P.𝛑[j] * πᶜ[k]
		end
	end
	D[t] = sum(f[t])
	f[t] ./= D[t]
	ℓ[1] += log(D[t])
	if length(clicks.time) > 0
		adaptedclicks = adapt(clicks, θnative.k[1], θnative.ϕ[1])
	end
	@inbounds for t=2:trial.ntimesteps
		if t ∈ clicks.inputtimesteps
			clickindex = clicks.inputindex[t][1]
			Aᵃ = Aᵃinput[clickindex]
			update_for_transition_probabilities!(P, adaptedclicks, clicks, t)
			transitionmatrix!(Aᵃ, P)
		else
			Aᵃ = Aᵃsilent
		end
		f[t] = p𝐘𝑑[t] .* (Aᵃ * f[t-1] * Aᶜᵀ)
		D[t] = sum(f[t])
		f[t] ./= D[t]
		ℓ[1] += log(D[t])
	end
	b = ones(Ξ,K)
	f⨀b = f # reuse memory
	@inbounds for t = trial.ntimesteps:-1:1
		if t < trial.ntimesteps
			if t+1 ∈ clicks.inputtimesteps
				clickindex = clicks.inputindex[t+1][1]
				Aᵃₜ₊₁ = Aᵃinput[clickindex]
			else
				Aᵃₜ₊₁ = Aᵃsilent
			end
			b = transpose(Aᵃₜ₊₁) * (b.*p𝐘𝑑[t+1]./D[t+1]) * Aᶜ
			f⨀b[t] .*= b
		end
		if t > 1
			if t ∈ clicks.inputtimesteps
				clickindex = clicks.inputindex[t][1]
				Aᵃₜ = Aᵃinput[clickindex]
			else
				Aᵃₜ = Aᵃsilent
			end
			for j = 1:K
	 			for k = 1:K
					∑χ[j,k] += sum_product_over_accumulator_states(D[t],f[t-1],b,p𝐘𝑑[t],Aᵃₜ,Aᶜ,j,k)
				end
			end
		end
	end
	∑γ .+= dropdims(sum(f⨀b[1],dims=1),dims=1)
	offset = 0
	for i = 1:m-1
		offset += model.trialsets[s].trials[i].ntimesteps
	end
	for t = 1:trial.ntimesteps
		τ = offset+t
		for i = 1:Ξ
			for k = 1:K
				memory.γ[s][i,k][τ] = f⨀b[t][i,k]
			end
		end
	end
	return nothing
end

"""
	sum_product_over_accumulator_states(D,fₜ₋₁,bₜ,Y,A,C,icₜ,icₜ₋₁)

Multiply terms across different states of the latent variables at consecutive time step and sum

ARGUMENT
-`Y`: similar to η, element Y[i,j] corresponds to i-th state of the accumulator at time t and the j-th state of the coupling at time t
-`A`: element A[i,j] corresponds to i-th state of the accumulator at time t and the j-th state of the accumulator at time t-1
-`C`: element C[i,j] corresponds to i-th state of the coupling at time t and the j-th state of the coupling at time t-1
-`icₜ`: index of the coupling state at the current time step
-`icₜ₋₁`: index of the coupling state at the previous time step

RETURN
-`s`: sum of the product across all states of the two latent variables at two consecutive time steps
"""
function sum_product_over_accumulator_states(D::Real, fₜ₋₁::Matrix{<:Real}, bₜ::Matrix{<:Real}, Y::Matrix{<:Real}, A::Matrix{<:Real}, C::Matrix{<:Real}, icₜ::Integer, icₜ₋₁::Integer)
	s = 0.0
	Ξ = size(fₜ₋₁,1)
	@inbounds for iaₜ = 1:Ξ
		for iaₜ₋₁ = 1:Ξ
			s += fₜ₋₁[iaₜ₋₁,icₜ₋₁]*bₜ[iaₜ,icₜ]*Y[iaₜ,icₜ]*A[iaₜ,iaₜ₋₁]*C[icₜ, icₜ₋₁]
		end
	end
	return s/D
end


"""
	choiceposteriors!(memory, model)

Compute the posterior probability of the latent variables at each time step, conditioned on only the choice

MODIFIED ARGUMENT
-`memory`: structure containing the memory for computing the gradient of the log-likelihood of the model. In particular, the field `γ` corresponds to the single-time-step posterior probabilities of the accumulator and coupling variables at each time step conditioned on the choices (𝑑). Element `γ[i][j,k][t]` represent the posterior probability at the t-th timestep in the i-th trialset: p(aₜⱼ=1, cₜₖ=1 ∣ 𝑑)

ARGUMENT
-`model`: custom type containing the settings, data, and parameters of a factorial hidden Markov drift-diffusion model

RETURN
-`P`: a structure for computing the derivatives with respect to the drift-diffusion parameters, in case it is to be reused
"""
function choiceposteriors!(memory::Memoryforgradient, model::Model)
	@unpack options, θnative, trialsets = model
	@unpack K, Ξ = options
	@unpack p𝑑_a, p𝐘𝑑 = memory
	@inbounds for i in eachindex(p𝐘𝑑)
		for m in eachindex(p𝐘𝑑[i])
			choicelikelihood!(p𝑑_a[i][m], trialsets[i].trials[m].choice, θnative.ψ[1])
			for j = 1:Ξ
				for k = 1:K
					p𝐘𝑑[i][m][end][j,k] = p𝑑_a[i][m][j]
				end
			end
		end
    end
	P = update_for_latent_dynamics!(memory, options, θnative)
	posteriors!(memory, P, model)
	return P
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
	updatecoupling!(memory, θnative)
	return P
end

"""
	posteriorcoupled(model)

Posterior probability of being in the first coupling state

ARGUMENT
-`model`: instance of the factorial hidden markov drift diffusion model

OUTPUT
-`fbz`: a nested array whose element `fbz[i][m][t]` represents the posterior porobability that the neural population is coupled to the accumulator in the timestep t of trial m of trialset i.
"""
function posterior_first_state(model::Model)
	γ = posteriors(model)
	fb = sortbytrial(γ, model)
	map(fb) do fb # trialset
		map(fb) do fb # trial
			map(fb) do fb #timestep
				sum(fb[:,1])
			end
		end
	end
end

"""
	randomposterior(mpGLM)

Create random posterior probabilities of the latent variables for testing

ARGUMENT
-`mpGLM`: a mixture of Poisson GLM

OPTIONAL ARGUMENT
-`rng`: random number generator

RETURN
-`γ`: γ[j,k][t] represents p{a(t)=ξ(j), c(t)=k ∣ 𝐘}
"""
function randomposterior(mpGLM::MixturePoissonGLM; rng::AbstractRNG=MersenneTwister())
	T = length(mpGLM.𝐲)
	Ξ = length(mpGLM.d𝛏_dB)
	K = length(mpGLM.θ.𝐯)
	γ = map(index->zeros(T), CartesianIndices((Ξ,K)))
	for t=1:T
		randγₜ = rand(rng,Ξ,K)
		randγₜ ./= sum(randγₜ)
		for j = 1:Ξ
			for k = 1:K
				γ[j,k][t] = randγₜ[j,k]
			end
		end
	end
	γ
end

"""
	posteriors!(memory, i, n, model)

Posterior probability of the latent variables conditioned on the spike train of one neuron

MODIFIED ARGUMENT
-`memory`: structure containing variables memory between computations of the model's log-likelihood and its gradient

UNMODIFIED ARGUMENT
-`i`: index of the trialset containing the neuron
-`n`: index of the neuron in the trialset
-`model`: structure containing the data, parameters, and hyperparameters

RETURN
-`γ`: posterior probability of the latent variables. The element `γ[j,k][τ]` corresponds to the posterior probability of the accumulator in the j-th state, the coupling in the k-th state, for the τ-timestep in the trialset.
"""
function posteriors!(memory::Memoryforgradient, i::Integer, n::Integer, model::Model)
	likelihood!(memory.p𝐘𝑑[i], model.trialsets[i].mpGLMs[n])
	P = update_for_latent_dynamics!(memory, model.options, model.θnative)
	posteriors!(memory, P, model)
	return memory.γ[i]
end

"""
	likelihood!(p𝐲, mpGLM)

Conditional likelihood of the spiking of one neuron

MODIFIED ARGUMENT
-`𝐩`: A nested array whose element `𝐩[m][t][j,k]` corresponds to the conditional likelihood of the spiking given the coupling in the k-th state and the accumulator in the j-th state, at the t-th time step of the m-th trial

UNMODIFIED ARGUMENT
-`mpGLM`: structure containing the data and parameters of the mixture Poisson GLM of one neuron
"""
function likelihood!(𝐩::Vector{<:Vector{<:Matrix{<:Real}}}, mpGLM::MixturePoissonGLM)
	(Ξ,K) = size(𝐩[1][end])
	for j = 1:Ξ
		for k = 1:K
			p𝐲_jk = scaledlikelihood(mpGLM, j, k, 1.0)
			τ = 0
			for m in eachindex(𝐩)
				for t in eachindex(𝐩[m])
					τ += 1
					𝐩[m][t][j,k] = p𝐲_jk[τ]
				end
			end
		end
	end
	return nothing
end

"""
	posterior_on_spikes!(memory, model)

Posterior probability of the latent variables conditioned on only the spiking and not the choice

MODIFIED ARGUMENT
-`memory`: structure containing variables memory between computations of the model's log-likelihood and its gradient

UNMODIFIED ARGUMENT
-`model`: structure containing the data, parameters, and hyperparameters
"""
function posterior_on_spikes!(memory::Memoryforgradient, model::Model)
	p𝐘 = memory.p𝐘𝑑
	for i in eachindex(p𝐘)
		scaledlikelihood!(p𝐘[i], model.options.sf_y, model.trialsets[i])
	end
	P = update_for_latent_dynamics!(memory, model.options, model.θnative)
	posteriors!(memory, P, model)
	return nothing
end

"""
    scaledlikelihood!(p𝐘, s, trialset)

In-place computation the conditional likelihood of the simultaneous spike response

MODIFIED ARGUMENT
-`p𝐘`: Conditional likelihood of the spike response at each time step. Element `p𝐘[m][t][j,k] `corresponds to ∏ₙᴺ p(𝐲ₙ(t) | aₜ = ξⱼ, cₜ=k) across N neurons at the t-th time step in the m-th trial.

UNMODIFIED ARGUMENT
-`s`: scale factor of the conditional likelihood of the spike train
-`trialset`: a group of trials in which the neurons are simultaneously recorded

RETURN
-`nothing`
"""
function scaledlikelihood!(p𝐘::Vector{<:Vector{<:Matrix{<:Real}}}, s::Real, trialset::Trialset)
	(Ξ,K) = size(p𝐘[1][end])
	N = length(trialset.mpGLMs)
    for j = 1:Ξ
        for k = 1:K
			𝐩 = scaledlikelihood(trialset.mpGLMs[1], j, k, s)
            for n = 2:N
			    scaledlikelihood!(𝐩, trialset.mpGLMs[n], j, k, s)
            end
            τ = 0
            for m in eachindex(p𝐘)
                for t in eachindex(p𝐘[m])
                    τ += 1
                    p𝐘[m][t][j,k] = 𝐩[τ]
                end
            end
        end
    end
    return nothing
end
