"""
	functions

-maximizelikelihood!(model, optimizer::Optim.FirstOrderOptimizer)
-loglikelihood(model)
-loglikelihood!(model, memory, concatenatedparameters)
"""

"""
    maximizelikelihood!(model, optimizer)

Optimize the parameters of the factorial hidden Markov drift-diffusion model using a first-order optimizer in Optim

MODIFIED ARGUMENT
- a structure containing information for a factorial hidden Markov drift-diffusion model

UNMODIFIED ARGUMENT
-`optimizer`: an optimizer implemented by Optim.jl. The limited memory quasi-Newton algorithm `LBFGS()` does pretty well, and when using L-BFGS the `HagerZhang()` line search seems to do better than `BackTracking()`

OPTIONAL ARGUMENT
-`extended_trace`: save additional information
-`f_tol`: threshold for determining convergence in the objective value
-`g_tol`: threshold for determining convergence in the gradient
-`iterations`: number of inner iterations that will be run before the optimizer gives up
-`show_every`: trace output is printed every `show_every`th iteration.
-`show_trace`: should a trace of the optimizer's state be shown?
-`x_tol`: threshold for determining convergence in the input vector

RETURN
`losses`: value of the loss function (negative of the model's log-likelihood) across iterations. If `store_trace` were set to false, then these are NaN's
`gradientnorms`: 2-norm of the gradient of  of the loss function across iterations. If `store_trace` were set to false, then these are NaN's
"""
function maximizelikelihood!(model::Model,
							 optimizer::Optim.FirstOrderOptimizer;
			                 extended_trace::Bool=false,
			                 f_tol::AbstractFloat=0.0,
			                 iterations::Integer=500,
			                 show_every::Integer=10,
			                 show_trace::Bool=true,
							 store_trace::Bool=true,
			                 x_tol::AbstractFloat=0.0)
	memory = Memoryforgradient(model)
    f(concatenatedθ) = -loglikelihood!(model, memory, concatenatedθ)
    g!(∇, concatenatedθ) = ∇negativeloglikelihood!(∇, memory, model, concatenatedθ)
    Optim_options = Optim.Options(extended_trace=extended_trace,
								  f_tol=f_tol,
                                  g_tol=model.options.g_tol,
                                  iterations=iterations,
                                  show_every=show_every,
                                  show_trace=show_trace,
								  store_trace=store_trace,
                                  x_tol=x_tol)
	θ₀ = concatenateparameters(model)
	optimizationresults = Optim.optimize(f, g!, θ₀, optimizer, Optim_options)
    θₘₗ = Optim.minimizer(optimizationresults)
	sortparameters!(model, θₘₗ, memory.indexθ)
	real2native!(model.θnative, model.options, model.θreal)
	println(optimizationresults)
	losses, gradientnorms = fill(NaN, iterations+1), fill(NaN, iterations+1)
	if store_trace
		traces = Optim.trace(optimizationresults)
		for i in eachindex(traces)
			gradientnorms[i] = traces[i].g_norm
			losses[i] = traces[i].value
		end
	end
    return losses, gradientnorms
end

"""
	loglikelihood(model)

Log of the likelihood of the data given the parameters
"""
loglikelihood(model::Model) = loglikelihood!(model, Memoryforgradient(model), concatenateparameters(model))

"""
    loglikelihood!(model, memory, concatenatedθ)

Compute the log-likelihood

MODIFIED ARGUMENT
-`model`: an instance of FHM-DDM
-`memory`: a container of variables used by both the log-likelihood and gradient computation

UNMODIFIED ARGUMENT
-`concatenatedθ`: a vector of concatenated parameter values

RETURN
-log-likelihood
```
"""
function loglikelihood!(model::Model, memory::Memoryforgradient, concatenatedθ::Vector{<:Real})
	if (concatenatedθ != memory.concatenatedθ) || isnan(memory.ℓ[1])
		P = update!(memory, model, concatenatedθ)
		memory.ℓ[1] = 0.0
		log_s = log(model.options.sf_y)
		@inbounds for trialset in model.trialsets
			N = length(trialset.mpGLMs)
			for trial in trialset.trials
				T = trial.ntimesteps
				memory.ℓ[1] -= N*T*log_s
				forward!(memory, P, model.θnative, trial)
			end
		end
	end
	memory.ℓ[1]
end

"""
    loglikelihood(concatenatedθ, indexθ, model)

ForwardDiff-compatible computation of the log-likelihood

ARGUMENT
-`concatenatedθ`: a vector of concatenated parameter values
-`indexθ`: struct indexing of each parameter in the vector of concatenated values
-`model`: an instance of FHM-DDM

RETURN
-log-likelihood
"""
function loglikelihood(concatenatedθ::Vector{type}, indexθ::Indexθ, model::Model) where {type<:Real}
	model = Model(concatenatedθ, indexθ, model)
	@unpack options, θnative, θreal, trialsets = model
	@unpack Δt, minpa, sf_y, K, Ξ = options
	p𝐘𝑑=map(model.trialsets) do trialset
			map(trialset.trials) do trial
				map(1:trial.ntimesteps) do t
					ones(type,Ξ,K)
				end
			end
		end
	p𝑑_a=map(model.trialsets) do trialset
			map(trialset.trials) do trial
				ones(type,Ξ)
			end
		end
    scaledlikelihood!(p𝐘𝑑, p𝑑_a, trialsets, θnative.ψ[1])
	choiceLLscaling = scale_factor_choiceLL(model)
	Aᵃinput = ones(type,Ξ,Ξ).*minpa
	one_minus_Ξminpa = 1.0-Ξ*minpa
	Aᵃinput[1,1] += one_minus_Ξminpa
	Aᵃinput[Ξ,Ξ] += one_minus_Ξminpa
	Aᵃsilent = copy(Aᵃinput)
	expλΔt = exp(θnative.λ[1]*Δt)
	dμ_dΔc = differentiate_μ_wrt_Δc(Δt, θnative.λ[1])
	d𝛏_dB = (2 .*collect(1:Ξ) .- Ξ .- 1)./(Ξ-2)
	𝛏 = θnative.B[1].*d𝛏_dB
	transitionmatrix!(Aᵃsilent, minpa, expλΔt.*𝛏, √(Δt*θnative.σ²ₐ[1]), 𝛏)
	Aᶜ₁₁ = θnative.Aᶜ₁₁[1]
	Aᶜ₂₂ = θnative.Aᶜ₂₂[1]
	πᶜ₁ = θnative.πᶜ₁[1]
	if K == 2
		Aᶜᵀ = [Aᶜ₁₁ 1-Aᶜ₁₁; 1-Aᶜ₂₂ Aᶜ₂₂]
		πᶜᵀ = [πᶜ₁ 1-πᶜ₁]
	else
		Aᶜᵀ = ones(type,1,1)
		πᶜᵀ = ones(type,1,1)
	end
	log_s = log(sf_y)
	ℓ = zero(type)
	@inbounds for s in eachindex(trialsets)
		nneurons = length(trialsets[s].mpGLMs)
		for m in eachindex(trialsets[s].trials)
			trial = trialsets[s].trials[m]
			ℓ-=nneurons*trial.ntimesteps*log_s
			p𝐚ₜ = probabilityvector(minpa, θnative.μ₀[1]+θnative.wₕ[1]*trial.previousanswer, √θnative.σ²ᵢ[1], 𝛏)
			f = p𝐘𝑑[s][m][1] .* p𝐚ₜ .* πᶜᵀ
			D = sum(f)
			D = max(D, nextfloat(0.0))
			f./=D
			ℓ+=log(D)
			if length(trial.clicks.time) > 0
				adaptedclicks = adapt(trial.clicks, θnative.k[1], θnative.ϕ[1])
			end
			for t=2:trial.ntimesteps
				if t ∈ trial.clicks.inputtimesteps
					cL = sum(adaptedclicks.C[trial.clicks.left[t]])
					cR = sum(adaptedclicks.C[trial.clicks.right[t]])
					𝛍 = expλΔt.*𝛏 .+ (cR-cL).*dμ_dΔc
					σ = √((cR+cL)*θnative.σ²ₛ[1] + Δt*θnative.σ²ₐ[1])
					transitionmatrix!(Aᵃinput, minpa, 𝛍, σ, 𝛏)
					Aᵃ = Aᵃinput
				else
					Aᵃ = Aᵃsilent
				end
				f = p𝐘𝑑[s][m][t] .* (Aᵃ * f * Aᶜᵀ)
				D = sum(f)
				D = max(D, nextfloat(0.0))
				f./=D
				ℓ+=log(D)
				if choiceLLscaling > 1
					p𝐚ₜ = Aᵃ*p𝐚ₜ
				end
			end
			if choiceLLscaling > 1
				ℓ += (choiceLLscaling-1)*log(dot(p𝑑_a[s][m], p𝐚ₜ))
			end
		end
	end
	ℓ
end

"""
	∇negativeloglikelihood!(∇nℓ, memory, model, concatenatedθ)

Update the gradient of the negative log-likelihood of the model

MODIFIED ARGUMENT
-`∇nℓ`: the vector representing the gradient
-`memory`: memory allocated for computing the gradient. The log-likelihood is updated.
-`model`: structure containing the data, parameters, and hyperparameters of the model

ARGUMENT
-`concatenatedθ`: values of the model's parameters concatenated into a vector
"""
function ∇negativeloglikelihood!(∇nℓ::Vector{<:Real},
 								 memory::Memoryforgradient,
								 model::Model,
								 concatenatedθ::Vector{<:AbstractFloat})
	if concatenatedθ != memory.concatenatedθ
		P = update!(memory, model, concatenatedθ)
	else
		P = Probabilityvector(model.options.Δt, model.options.minpa, model.θnative, model.options.Ξ)
	end
	∇loglikelihood!(memory,model,P)
	indexall = 0
	indexfit = 0
	for field in fieldnames(Latentθ)
		indexall+=1
		if getfield(memory.indexθ.latentθ, field)[1] > 0
			indexfit +=1
			∇nℓ[indexfit] = -memory.∇ℓlatent[indexall]
		end
	end
	native2real!(∇nℓ, memory.indexθ.latentθ, model)
	∇ℓglm = vcat((vcat((concatenateparameters(∇) for ∇ in ∇s)...) for ∇s in memory.∇ℓglm)...)
	for i in eachindex(∇ℓglm)
		∇nℓ[indexfit+i] = -∇ℓglm[i]
	end
	return nothing
end

"""
	∇loglikelihood!(memory, model, P)

Compute the gradient of the log-likelihood within the fields of an object of composite type `Memoryforgradient`

MODIFIED ARGUMENT
-`memory`: memory allocated for computing the gradient. The log-likelihood is updated.
-`model`: structure containing the data, parameters, and hyperparameters of the model
-`P`: a structure containing allocated memory for computing the accumulator's initial and transition probabilities as well as the partial derivatives of these probabilities
"""
function ∇loglikelihood!(memory::Memoryforgradient, model::Model, P::Probabilityvector)
	memory.ℓ .= 0.0
	memory.∇ℓlatent .= 0.0
	@inbounds for s in eachindex(model.trialsets)
		for m in eachindex(model.trialsets[s].trials)
			∇loglikelihood!(memory, model, P, s, m)
		end
	end
	@inbounds for s in eachindex(model.trialsets)
		for n = 1:length(model.trialsets[s].mpGLMs)
			expectation_∇loglikelihood!(memory.∇ℓglm[s][n], memory.glmderivatives, memory.γ[s], model.trialsets[s].mpGLMs[n])
		end
	end
	return nothing
end

"""
	∇loglikelihood!(memory, model, P, s, m)

Update the gradient

MODIFIED ARGUMENT
-`memory`: memory allocated for computing the gradient. The log-likelihood is updated.
-`model`: structure containing the data, parameters, and hyperparameters of the model
-`P`: a structure containing allocated memory for computing the accumulator's initial and transition probabilities as well as the partial derivatives of these probabilities
-`s`: index of the trialset
-`m`: index of the trial
"""
function ∇loglikelihood!(memory::Memoryforgradient,
						 model::Model,
						 P::Probabilityvector,
						 s::Integer,
						 m::Integer)
	trial = model.trialsets[s].trials[m]
	p𝐘𝑑 = memory.p𝐘𝑑[s][m]
	p𝑑_a = memory.p𝑑_a[s][m]
	@unpack θnative = model
	@unpack clicks = trial
	@unpack inputtimesteps, inputindex = clicks
	@unpack Aᵃinput, ∇Aᵃinput, Aᵃsilent, ∇Aᵃsilent, Aᶜ, Aᶜᵀ, ∇Aᶜ, choiceLLscaling, D, f, fᶜ, indexθ_pa₁, indexθ_paₜaₜ₋₁, indexθ_pc₁, indexθ_pcₜcₜ₋₁, indexθ_ψ, K, ℓ, ∇ℓlatent, nθ_pa₁, nθ_paₜaₜ₋₁, nθ_pc₁, nθ_pcₜcₜ₋₁, ∇pa₁, πᶜ, ∇πᶜ, Ξ = memory
	if length(clicks.time) > 0
		adaptedclicks = ∇adapt(trial.clicks, θnative.k[1], θnative.ϕ[1])
	end
	ℓ[1] -= length(model.trialsets[s].mpGLMs)*trial.ntimesteps*log(model.options.sf_y)
	t = 1
	∇priorprobability!(∇pa₁, P, trial.previousanswer)
	fᶜ[1] = copy(P.𝛑)
	pa₁ = fᶜ[1]
	@inbounds for j=1:Ξ
		for k = 1:K
			f[t][j,k] = p𝐘𝑑[t][j,k] * pa₁[j] * πᶜ[k]
		end
	end
	D[t] = sum(f[t])
	D[t] = max(D[t], nextfloat(0.0))
	f[t] ./= D[t]
	ℓ[1] += log(D[t])
	@inbounds for t=2:trial.ntimesteps
		if t ∈ clicks.inputtimesteps
			clickindex = clicks.inputindex[t][1]
			Aᵃ = Aᵃinput[clickindex]
			∇Aᵃ = ∇Aᵃinput[clickindex]
			update_for_∇transition_probabilities!(P, adaptedclicks, clicks, t)
			∇transitionmatrix!(∇Aᵃ, Aᵃ, P)
		else
			Aᵃ = Aᵃsilent
		end
		f[t] = p𝐘𝑑[t] .* (Aᵃ * f[t-1] * Aᶜᵀ)
		D[t] = sum(f[t])
		D[t] = max(D[t], nextfloat(0.0))
		f[t] ./= D[t]
		ℓ[1] += log(D[t])
		if choiceLLscaling > 1
			fᶜ[t] = Aᵃ*fᶜ[t-1]
		end
	end
	b = ones(Ξ,K)
	f⨀b = f # reuse memory
	∇ℓlatent[indexθ_ψ[1]] += expectation_derivative_logp𝑑_wrt_ψ(trial.choice, f⨀b[trial.ntimesteps], θnative.ψ[1])
	if choiceLLscaling > 1
		fᶜ[trial.ntimesteps] .*= p𝑑_a
		Dᶜ = sum(fᶜ[trial.ntimesteps])
		Dᶜ = max(Dᶜ, nextfloat(0.0))
		ℓ[1] += (choiceLLscaling-1)*log(Dᶜ)
		fᶜ[trial.ntimesteps] ./= Dᶜ
		bᶜ = p𝑑_a./Dᶜ # backward term for the last time step
		γ = bᶜ.*fᶜ[trial.ntimesteps] # posterior probability for the last time step
		∇ℓlatent[indexθ_ψ[1]] += (choiceLLscaling-1)*expectation_derivative_logp𝑑_wrt_ψ(trial.choice, fᶜ[trial.ntimesteps], θnative.ψ[1])
	end
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
			if choiceLLscaling > 1
				bᶜ = transpose(Aᵃₜ₊₁) * bᶜ
			end
		end
		if t > 1
			if t ∈ clicks.inputtimesteps
				clickindex = clicks.inputindex[t][1]
				Aᵃ = Aᵃinput[clickindex]
				∇Aᵃ = ∇Aᵃinput[clickindex]
			else
				Aᵃ = Aᵃsilent
				∇Aᵃ = ∇Aᵃsilent
			end
			for i = 1:nθ_paₜaₜ₋₁
				∇ℓlatent[indexθ_paₜaₜ₋₁[i]] += sum_product_over_states(D[t], f[t-1], b, p𝐘𝑑[t], ∇Aᵃ[i], Aᶜ)
			end
			for i = 1:nθ_pcₜcₜ₋₁
				∇ℓlatent[indexθ_pcₜcₜ₋₁[i]] += sum_product_over_states(D[t], f[t-1], b, p𝐘𝑑[t], Aᵃ, ∇Aᶜ[i])
			end
			if choiceLLscaling > 1
				for i = 1:nθ_paₜaₜ₋₁
					∇ℓlatent[indexθ_paₜaₜ₋₁[i]] += (choiceLLscaling-1)*(transpose(bᶜ)*∇Aᵃ[i]*fᶜ[t-1])[1]
				end
			end
		end
	end
	t = 1
	@inbounds for i = 1:nθ_pa₁
		∇ℓlatent[indexθ_pa₁[i]] += sum_product_over_states(D[t], b, p𝐘𝑑[t], ∇pa₁[i], πᶜ)
	end
	@inbounds for i = 1:nθ_pc₁
		∇ℓlatent[indexθ_pc₁[i]] += sum_product_over_states(D[t], b, p𝐘𝑑[t], pa₁, ∇πᶜ[i])
	end
	if choiceLLscaling > 1
		@inbounds for i = 1:nθ_pa₁
			∇ℓlatent[indexθ_pa₁[i]] += (choiceLLscaling-1)*dot(bᶜ, ∇pa₁[i])
		end
	end
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
	Memoryforgradient(model)

Create variables that are memory by the computations of the log-likelihood and its gradient

ARGUMENT
-`model`: structure with information about the factorial hidden Markov drift-diffusion model

OUTPUT
-an instance of the custom type `Memoryforgradient`, which contains the memory quantities
```
"""
function Memoryforgradient(model::Model; choicemodel::Bool=false)
	@unpack options, θnative = model
	@unpack Δt, K, minpa, Ξ = options
	Aᶜ₁₁ = θnative.Aᶜ₁₁[1]
	Aᶜ₂₂ = θnative.Aᶜ₂₂[1]
	πᶜ₁ = θnative.πᶜ₁[1]
	if K == 2
		Aᶜ = [Aᶜ₁₁ 1-Aᶜ₂₂; 1-Aᶜ₁₁ Aᶜ₂₂]
		∇Aᶜ = [[1.0 0.0; -1.0 0.0], [0.0 -1.0; 0.0 1.0]]
		πᶜ = [πᶜ₁, 1-πᶜ₁]
		∇πᶜ = [[1.0, -1.0]]
	else
		Aᶜ = ones(1,1)
		∇Aᶜ = [zeros(1,1), zeros(1,1)]
		πᶜ = ones(1)
		∇πᶜ = [zeros(1)]
	end
	maxclicks = maximum_number_of_clicks(model)
	maxtimesteps = maximum_number_of_time_steps(model)
	if choicemodel
		concatenatedθ, indexθ = concatenate_choice_related_parameters(model)
		f = collect(zeros(Ξ,1) for t=1:maxtimesteps)
	else
		concatenatedθ = concatenateparameters(model)
		indexθ = indexparameters(model)
		f = collect(zeros(Ξ,K) for t=1:maxtimesteps)
	end
	indexθ_pa₁ = [3,6,11,13]
	indexθ_paₜaₜ₋₁ = [3,4,5,7,10,12]
	indexθ_pc₁ = [8]
	indexθ_pcₜcₜ₋₁ = [1,2]
	indexθ_ψ = [9]
	nθ_pa₁ = length(indexθ_pa₁)
	nθ_paₜaₜ₋₁ = length(indexθ_paₜaₜ₋₁)
	∇ℓglm = map(model.trialsets) do trialset
				map(trialset.mpGLMs) do mpGLM
					GLMθ(eltype(mpGLM.θ.𝐮), mpGLM.θ)
				end
			end
	one_minus_Ξminpa = 1.0 - Ξ*minpa
	Aᵃinput=map(1:maxclicks) do t
				A = ones(Ξ,Ξ).*minpa
				A[1,1] += one_minus_Ξminpa
				A[Ξ,Ξ] += one_minus_Ξminpa
				return A
			end
	Aᵃsilent = copy(Aᵃinput[1])
	∇Aᵃinput = collect(collect(zeros(Ξ,Ξ) for q=1:nθ_paₜaₜ₋₁) for t=1:maxclicks)
	∇Aᵃsilent = map(i->zeros(Ξ,Ξ), 1:nθ_paₜaₜ₋₁)
	p𝐘𝑑=map(model.trialsets) do trialset
			map(trialset.trials) do trial
				map(1:trial.ntimesteps) do t
					ones(Ξ,K)
				end
			end
		end
	p𝑑_a=map(model.trialsets) do trialset
			map(trialset.trials) do trial
				ones(Ξ)
			end
		end
	γ =	map(model.trialsets) do trialset
			map(CartesianIndices((Ξ,K))) do index
				zeros(trialset.ntimesteps)
			end
		end
	memory = Memoryforgradient(Aᵃinput=Aᵃinput,
								∇Aᵃinput=∇Aᵃinput,
								Aᵃsilent=Aᵃsilent,
								∇Aᵃsilent=∇Aᵃsilent,
								Aᶜ=Aᶜ,
								∇Aᶜ=∇Aᶜ,
								choiceLLscaling = scale_factor_choiceLL(model),
								concatenatedθ=similar(concatenatedθ),
								Δt=options.Δt,
								f=f,
								glmderivatives = GLMDerivatives(model.trialsets[1].mpGLMs[1]),
								indexθ=indexθ,
								indexθ_pa₁=indexθ_pa₁,
								indexθ_paₜaₜ₋₁=indexθ_paₜaₜ₋₁,
								indexθ_pc₁=indexθ_pc₁,
								indexθ_pcₜcₜ₋₁=indexθ_pcₜcₜ₋₁,
								indexθ_ψ=indexθ_ψ,
								γ=γ,
								K=K,
								maxtimesteps=maxtimesteps,
								∇ℓglm=∇ℓglm,
								πᶜ=πᶜ,
								∇πᶜ=∇πᶜ,
								p𝑑_a=p𝑑_a,
								p𝐘𝑑=p𝐘𝑑,
								Ξ=Ξ)
	return memory
end

"""
	maximum_number_of_clicks(model)

Return the maximum number of clicks across all trials.

The stereoclick is excluded from this analysis as well as all other analyses.
"""
function maximum_number_of_clicks(model::Model)
	maxclicks = 0
	@inbounds for trialset in model.trialsets
		for trial in trialset.trials
			maxclicks = max(maxclicks, length(trial.clicks.time))
		end
	end
	return maxclicks
end

"""
	maximum_number_of_time_steps(model)

Return the maximum number of time steps across all trials
"""
function maximum_number_of_time_steps(model::Model)
	maxtimesteps = 0
	@inbounds for trialset in model.trialsets
		for trial in trialset.trials
			maxtimesteps = max(maxtimesteps, trial.ntimesteps)
		end
	end
	return maxtimesteps
end

"""
	update!(memory, model)

Update the memory quantities

MODIFIED ARGUMENT
-`memory`: structure containing variables memory between computations of the model's log-likelihood and its gradient

UNMODIFIED ARGUMENT
-`model`: structure with information concerning a factorial hidden Markov drift-diffusion model

RETURN
-`P`: a composite of the type `Probabilityvector` that contains quantifies used for computing the probability vectors of the accumulator variables and its first and second derivatives
"""

update!(memory::Memoryforgradient, model::Model) = update!(memory, model, concatenateparameters(model))

"""
	update!(model, memory, concatenatedθ)

Update the model and the memory quantities according to new parameter values

MODIFIED ARGUMENT
-`memory`: structure containing variables memory between computations of the model's log-likelihood and its gradient
-`model`: structure with information concerning a factorial hidden Markov drift-diffusion model

ARGUMENT
-`concatenatedθ`: newest values of the model's parameters

RETURN
-`P`: an instance of `Probabilityvector`
```
"""
function update!(memory::Memoryforgradient, model::Model, concatenatedθ::Vector{<:Real})
	@unpack options, θnative, θreal = model
	@unpack Δt, K, minpa,  Ξ = options
	memory.concatenatedθ .= concatenatedθ
	sortparameters!(model, memory.concatenatedθ, memory.indexθ)
	real2native!(θnative, options, θreal)
	if !isempty(memory.p𝐘𝑑[1][1][1])
	    scaledlikelihood!(memory.p𝐘𝑑, memory.p𝑑_a, model.trialsets, θnative.ψ[1])
	end
	P = update_for_∇latent_dynamics!(memory, options, θnative)
	return P
end

"""
	update_for_∇latent_dynamics!(memory, options, θnative)

Update quantities for computing the gradient of the prior and transition probabilities of the latent variables

MODIFIED ARGUMENT
-`memory`: structure containing variables memory between computations of the model's log-likelihood and its gradient

UNMODIFIED ARGUMENT
-`options`: settings of the model
-`θnative`: values of the parameters that control the latent variables, in the parameters' native space

RETURN
-`P`: an instance of `Probabilityvector`
"""
function update_for_∇latent_dynamics!(memory::Memoryforgradient, options::Options, θnative::Latentθ)
	P = Probabilityvector(options.Δt, options.minpa, θnative, options.Ξ)
	update_for_∇transition_probabilities!(P)
	∇transitionmatrix!(memory.∇Aᵃsilent, memory.Aᵃsilent, P)
	updatecoupling!(memory, θnative)
	return P
end

"""
	updatecoupling!(memory, θnative)

Update quantities for computing the prior and transition probability of the coupling variables

MODIFIED ARGUMENT
-`memory`: structure containing variables memory between computations of the model's log-likelihood and its gradient

UNMODIFIED ARGUMENT
-`θnative`: values of the parameters that control the latent variables, in the parameters' native space
"""
function updatecoupling!(memory::Memoryforgradient, θnative::Latentθ)
	if memory.K == 2
		Aᶜ₁₁ = θnative.Aᶜ₁₁[1]
		Aᶜ₂₂ = θnative.Aᶜ₂₂[1]
		πᶜ₁ = θnative.πᶜ₁[1]
		memory.Aᶜ .= [Aᶜ₁₁ 1-Aᶜ₂₂; 1-Aᶜ₁₁ Aᶜ₂₂]
		memory.πᶜ .= [πᶜ₁, 1-πᶜ₁]
	end
	return nothing
end

"""
	scale_factor_choiceLL(model)

Scaling factor for the log-likelihood of behavioral choices
"""
function scale_factor_choiceLL(model::Model)
	a = model.options.choiceLL_scaling_exponent
	if a==0
		1.0
	else
		ntimesteps_neurons = sum(collect(trialset.ntimesteps*length(trialset.mpGLMs) for trialset in model.trialsets))
		ntrials = sum(collect(trialset.ntrials for trialset in model.trialsets))
		(ntimesteps_neurons/ntrials)^a
	end
end
