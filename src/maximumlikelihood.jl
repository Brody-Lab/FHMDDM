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

EXAMPLE
```julia-repl
julia> using FHMDDM, LineSearches, Optim
julia> datapath = "/mnt/cup/labs/brody/tzluo/analysis_data/analysis_2022_05_05_test/data.mat"
julia> model = Model(datapath)
julia> losses, gradientnorms = maximizelikelihood!(model, LBFGS(linesearch = LineSearches.BackTracking()))
```
"""
function maximizelikelihood!(model::Model,
							 optimizer::Optim.FirstOrderOptimizer;
			                 extended_trace::Bool=false,
			                 f_tol::AbstractFloat=0.0,
			                 g_tol::AbstractFloat=1e-8,
			                 iterations::Integer=1000,
			                 show_every::Integer=10,
			                 show_trace::Bool=true,
							 store_trace::Bool=true,
			                 x_tol::AbstractFloat=0.0)
	memory = Memoryforgradient(model)
    f(concatenatedθ) = -loglikelihood!(model, memory, concatenatedθ)
    g!(∇, concatenatedθ) = ∇negativeloglikelihood!(∇, memory, model, concatenatedθ)
    Optim_options = Optim.Options(extended_trace=extended_trace,
								  f_tol=f_tol,
                                  g_tol=g_tol,
                                  iterations=iterations,
                                  show_every=show_every,
                                  show_trace=show_trace,
								  store_trace=store_trace,
                                  x_tol=x_tol)
	θ₀ = concatenateparameters(model)[1]
	optimizationresults = [] # so that the variable is not confined to the scope of the while loop
	ongoing = true
	while ongoing
		optimizationresults = Optim.optimize(f, g!, θ₀, optimizer, Optim_options)
		ongoing = isnan(Optim.minimum(optimizationresults))
	end
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
    maximizelikelihood!(model, optimizer)

Optimize the parameters of the factorial hidden Markov drift-diffusion model using an algorithm implemented by Flux.jl

MODIFIED ARGUMENT
- a structure containing information for a factorial hidden Markov drift-diffusion model

UNMODIFIED ARGUMENT
-`optimizer`: optimization algorithm implemented by Flux.jl

OPTIONAL ARGUMENT
-`iterations`: number of inner iterations that will be run before the optimizer gives up

RETURN
`losses`: a vector of floating-point numbers indicating the value of the loss function (negative of the model's log-likelihood) across iterations. If `store_trace` were set to false, then these are NaN's
`gradientnorms`: a vector of floating-point numbers indicating the 2-norm of the gradient of  of the loss function across iterations. If `store_trace` were set to false, then these are NaN's

EXAMPLE
```julia-repl
julia> using FHMDDM
julia> import Flux
julia> datapath = "/mnt/cup/labs/brody/tzluo/analysis_data/analysis_2022_04_27_test/data.mat"
julia> model = Model(datapath)
julia> losses, gradientnorms = FHMDDM.maximizelikelihood!(model, Flux.ADAM())
```
"""
function maximizelikelihood!(model::Model,
							optimizer::Flux.Optimise.AbstractOptimiser;
							iterations::Integer = 3000)
	memory = Memoryforgradient(model)
	θ = concatenateparameters(model)[1]
	∇ = similar(θ)
	local x, min_err, min_θ
	min_err = typemax(eltype(θ)) #dummy variables
	min_θ = copy(θ)
	losses, gradientnorms = fill(NaN, iterations+1), fill(NaN, iterations+1)
	optimizationtime = 0.0
	for i = 1:iterations
		iterationtime = @timed begin
			x = -loglikelihood!(model, memory, θ)
			∇negativeloglikelihood!(∇, memory, model, θ)
			losses[i] = x
			gradientnorms[i] = norm(∇)
			if x < min_err  # found a better solution
				min_err = x
				min_θ = copy(θ)
			end
			Flux.update!(optimizer, θ, ∇)
		end
		optimizationtime += iterationtime[2]
		println("iteration=", i, ", loss= ", losses[i], ", gradient norm= ", gradientnorms[i], ", time(s)= ", optimizationtime)
	end
	sortparameters!(model, min_θ, memory.indexθ)
	real2native!(model.θnative, model.options, model.θreal)
    return losses, gradientnorms
end

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

EXAMPLE
```julia-repl
julia> using FHMDDM
julia> model = Model("/mnt/cup/labs/brody/tzluo/analysis_data/analysis_2022_04_27_test/data.mat"; randomize=true);
julia> concatenatedθ, indexθ = FHMDDM.concatenateparameters(model)
julia> memory = FHMDDM.Memoryforgradient(model)
julia> ℓ = loglikelihood!(model, memory, memory.concatenatedθ)
julia> ℓ = loglikelihood!(model, memory, rand(length(memory.concatenatedθ)))
```
"""
function loglikelihood!(model::Model,
						memory::Memoryforgradient,
					    concatenatedθ::Vector{<:Real})
	if concatenatedθ != memory.concatenatedθ
		P = update!(memory, model, concatenatedθ)
		memory.ℓ[1] = 0.0
		@inbounds for s in eachindex(model.trialsets)
			for m in eachindex(model.trialsets[s].trials)
				memory.ℓ[1] += loglikelihood(memory.p𝐘𝑑[s][m], memory, P, model.θnative, model.trialsets[s].trials[m])
			end
		end
	end
	memory.ℓ[1]
end

"""
	loglikelihood(p𝐘𝑑, θnative, trial)

Compute the log-likelihood of the data from one trial

ARGUMENT
-`p𝐘𝑑`: a matrix whose element `p𝐘𝑑[t][i,j]` represents the conditional likelihood `p(𝐘ₜ, d ∣ 𝐚ₜ=i, 𝐜ₜ=j)`
-`θnative`: model parameters in their native space
-`trial`: stimulus and behavioral information of one trial

RETURN
-`ℓ`: log-likelihood of the data from one trial
"""
function loglikelihood(p𝐘𝑑::Vector{<:Matrix{<:Real}},
   					   memory::Memoryforgradient,
					   P::Probabilityvector,
					   θnative::Latentθ,
					   trial::Trial)
	@unpack clicks = trial
	@unpack Aᵃinput, Aᵃsilent, Aᶜᵀ, πᶜᵀ = memory
    if length(clicks.time) > 0
		adaptedclicks = adapt(clicks, θnative.k[1], θnative.ϕ[1])
	end
	priorprobability!(P, trial.previousanswer)
	pa₁ = P.𝛑
	f = p𝐘𝑑[1] .* pa₁ .* πᶜᵀ
	D = sum(f)
	f ./= D
	ℓ = log(D)
	@inbounds for t = 2:trial.ntimesteps
		if isempty(clicks.inputindex[t])
			Aᵃ = Aᵃsilent
		else
			Aᵃ = Aᵃinput[clicks.inputindex[t][1]]
			update_for_transition_probabilities!(P, adaptedclicks, clicks, t)
			transitionmatrix!(Aᵃ, P)
		end
		f = p𝐘𝑑[t].*(Aᵃ * f * Aᶜᵀ)
		D = sum(f)
		f ./= D
		ℓ += log(D)
	end
	return ℓ
end

"""
    loglikelihood(concatenatedθ, indexθ, model)

ForwardDiff-compatible computation of the log-likelihood

UNMODIFIED ARGUMENT
-`concatenatedθ`: a vector of concatenated parameter values
-`indexθ`: struct indexing of each parameter in the vector of concatenated values
-`model`: an instance of FHM-DDM

RETURN
-log-likelihood

EXAMPLE
```julia-repl
julia> using FHMDDM
julia> model = Model("/mnt/cup/labs/brody/tzluo/analysis_data/analysis_2022_05_05_test/data.mat"; randomize=true);
julia> concatenatedθ, indexθ = FHMDDM.concatenateparameters(model)
julia> ℓ = loglikelihood(concatenatedθ, indexθ, model)
julia> using ForwardDiff
julia> f(x) = loglikelihood(x, indexθ, model)
julia> g = ForwardDiff.gradient(f, concatenatedθ)
julia> memory = FHMDDM.Memoryforgradient(model)
julia> ℓ2 = loglikelihood!(model, memory, concatenatedθ) #ForwardDiff-incompatible
julia> abs(ℓ2-ℓ)
```
"""
function loglikelihood(	concatenatedθ::Vector{T},
					    indexθ::Indexθ,
						model::Model) where {T<:Real}
	model = Model(concatenatedθ, indexθ, model)
	@unpack options, θnative, θreal, trialsets = model
	@unpack Δt, K, Ξ = options
	p𝐘𝑑=map(model.trialsets) do trialset
			map(trialset.trials) do trial
				map(1:trial.ntimesteps) do t
					ones(T,Ξ,K)
				end
			end
		end
    likelihood!(p𝐘𝑑, trialsets, θnative.ψ[1])
	Aᵃinput, Aᵃsilent = zeros(T,Ξ,Ξ), zeros(T,Ξ,Ξ)
	expλΔt = exp(θnative.λ[1]*Δt)
	dμ_dΔc = differentiate_μ_wrt_Δc(Δt, θnative.λ[1])
	d𝛏_dB = (2 .*collect(1:Ξ) .- Ξ .- 1)./(Ξ-2)
	𝛏 = θnative.B[1].*d𝛏_dB
	transitionmatrix!(Aᵃsilent, expλΔt.*𝛏, √(Δt*θnative.σ²ₐ[1]), 𝛏)
	Aᶜ₁₁ = θnative.Aᶜ₁₁[1]
	Aᶜ₂₂ = θnative.Aᶜ₂₂[1]
	πᶜ₁ = θnative.πᶜ₁[1]
	if K == 2
		Aᶜᵀ = [Aᶜ₁₁ 1-Aᶜ₁₁; 1-Aᶜ₂₂ Aᶜ₂₂]
		πᶜᵀ = [πᶜ₁ 1-πᶜ₁]
	else
		Aᶜᵀ = ones(T,1,1)
		πᶜᵀ = ones(T,1,1)
	end
	ℓ = zero(T)
	@inbounds for s in eachindex(trialsets)
		for m in eachindex(trialsets[s].trials)
			trial = trialsets[s].trials[m]
			pa₁ = probabilityvector(θnative.μ₀[1]+θnative.wₕ[1]*trial.previousanswer, √θnative.σ²ᵢ[1], 𝛏)
			f = p𝐘𝑑[s][m][1] .* pa₁ .* πᶜᵀ
			D = sum(f)
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
					transitionmatrix!(Aᵃinput, 𝛍, σ, 𝛏)
					Aᵃ = Aᵃinput
				else
					Aᵃ = Aᵃsilent
				end
				f = p𝐘𝑑[s][m][t] .* (Aᵃ * f * Aᶜᵀ)
				D = sum(f)
				f./=D
				ℓ+=log(D)
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

EXAMPLE
```julia-repl
julia> using FHMDDM
julia> model = Model("/mnt/cup/labs/brody/tzluo/analysis_data/analysis_2022_05_19_test/T176_2018_05_03/data.mat");
julia> concatenatedθ, indexθ = FHMDDM.concatenateparameters(model)
julia> ∇nℓ = similar(concatenatedθ)
julia> memory = FHMDDM.Memoryforgradient(model)
julia> FHMDDM.∇negativeloglikelihood!(∇nℓ, memory, model, concatenatedθ)
julia> using ForwardDiff
julia> f(x) = -FHMDDM.loglikelihood(x, indexθ, model)
julia> ℓ_auto = f(concatenatedθ)
julia> ∇nℓ_auto = ForwardDiff.gradient(f, concatenatedθ)
julia> abs(ℓ_auto + memory.ℓ[1])
julia> maximum(abs.(∇nℓ_auto .- ∇nℓ))
```
"""
function ∇negativeloglikelihood!(∇nℓ::Vector{<:Real},
 								 memory::Memoryforgradient,
								 model::Model,
								 concatenatedθ::Vector{<:AbstractFloat})
	if concatenatedθ != memory.concatenatedθ
		P = update!(memory, model, concatenatedθ)
	else
		P = Probabilityvector(model.options.Δt, model.θnative, model.options.Ξ)
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
	for ∇ℓglms in memory.∇ℓglm
		for ∇ℓglm in ∇ℓglms
			for u in ∇ℓglm.𝐮
				indexfit+=1
				∇nℓ[indexfit] = -u
			end
			for 𝐯ₖ in ∇ℓglm.𝐯
				for v in 𝐯ₖ
					indexfit+=1
					∇nℓ[indexfit] = -v
				end
			end
		end
	end
	return nothing
end

"""
	∇loglikelihood!(memory, model, P)

MODIFIED ARGUMENT
-`memory`: memory allocated for computing the gradient. The log-likelihood is updated.
-`model`: structure containing the data, parameters, and hyperparameters of the model
-`P`: a structure containing allocated memory for computing the accumulator's initial and transition probabilities as well as the partial derivatives of these probabilities
"""
function ∇loglikelihood!(memory::Memoryforgradient,
						 model::Model,
						 P::Probabilityvector)
	memory.ℓ .= 0.0
	memory.∇ℓlatent .= 0.0
	@inbounds for s in eachindex(model.trialsets)
		for m in eachindex(model.trialsets[s].trials)
			∇loglikelihood!(memory, model, P, s, m)
		end
	end
	@inbounds for s in eachindex(model.trialsets)
		for n in eachindex(model.trialsets[s].mpGLMs)
			expectation_∇loglikelihood!(memory.∇ℓglm[s][n], memory.γ[s], model.trialsets[s].mpGLMs[n])
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
	@unpack θnative = model
	@unpack clicks = trial
	@unpack inputtimesteps, inputindex = clicks
	@unpack Aᵃinput, ∇Aᵃinput, Aᵃsilent, ∇Aᵃsilent, Aᶜ, Aᶜᵀ, ∇Aᶜ, D, f, indexθ_pa₁, indexθ_paₜaₜ₋₁, indexθ_pc₁, indexθ_pcₜcₜ₋₁, indexθ_ψ, K, ℓ, ∇ℓlatent, nθ_pa₁, nθ_paₜaₜ₋₁, nθ_pc₁, nθ_pcₜcₜ₋₁, ∇pa₁, πᶜ, ∇πᶜ, Ξ = memory
	t = 1
	∇priorprobability!(∇pa₁, P, trial.previousanswer)
	pa₁ = copy(P.𝛑) # save for later
	@inbounds for j=1:Ξ
		for k = 1:K
			f[t][j,k] = p𝐘𝑑[t][j,k] * pa₁[j] * πᶜ[k]
		end
	end
	D[t] = sum(f[t])
	f[t] ./= D[t]
	ℓ[1] += log(D[t])
	if length(clicks.time) > 0
		adaptedclicks = ∇adapt(trial.clicks, θnative.k[1], θnative.ϕ[1])
	end
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
		f[t] ./= D[t]
		ℓ[1] += log(D[t])
	end
	b = ones(Ξ,K)
	f⨀b = f # reuse memory
	∇ℓlatent[indexθ_ψ[1]] += expectation_derivative_logp𝑑_wrt_ψ(trial.choice, f⨀b[trial.ntimesteps], θnative.ψ[1])
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
		end
	end
	t = 1
	@inbounds for i = 1:nθ_pa₁
		∇ℓlatent[indexθ_pa₁[i]] += sum_product_over_states(D[t], b, p𝐘𝑑[t], ∇pa₁[i], πᶜ)
	end
	@inbounds for i = 1:nθ_pc₁
		∇ℓlatent[indexθ_pc₁[i]] += sum_product_over_states(D[t], b, p𝐘𝑑[t], pa₁, ∇πᶜ[i])
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
-`P`: an instance of `Probabilityvector`

EXAMPLE
```julia-repl
julia> using FHMDDM
julia> model = Model("/mnt/cup/labs/brody/tzluo/analysis_data/analysis_2022_04_27_test/data.mat"; randomize=true);
julia> memory = FHMDDM.Memoryforgradient(model)
```
"""
function Memoryforgradient(model::Model; choicemodel::Bool=false)
	@unpack options, θnative = model
	@unpack Δt, K, Ξ = options
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
		concatenatedθ, indexθ = concatenateparameters(model)
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
					GLMθ(mpGLM.θ, eltype(mpGLM.θ.𝐮))
				end
			end
	Aᵃinput=map(1:maxclicks) do t
				A = zeros(Ξ,Ξ)
				A[1,1] = A[Ξ,Ξ] = 1.0
				return A
			end
	∇Aᵃinput = collect(collect(zeros(Ξ,Ξ) for q=1:nθ_paₜaₜ₋₁) for t=1:maxclicks)
	Aᵃsilent = zeros(Ξ, Ξ)
	∇Aᵃsilent = map(i->zeros(Ξ,Ξ), 1:nθ_paₜaₜ₋₁)
	Aᵃsilent[1,1] = Aᵃsilent[Ξ, Ξ] = 1.0
	p𝐘𝑑=map(model.trialsets) do trialset
			map(trialset.trials) do trial
				map(1:trial.ntimesteps) do t
					ones(Ξ,K)
				end
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
								concatenatedθ=similar(concatenatedθ),
								D = zeros(maxtimesteps),
								Δt=options.Δt,
								f=f,
								indexθ=indexθ,
								indexθ_pa₁=indexθ_pa₁,
								indexθ_paₜaₜ₋₁=indexθ_paₜaₜ₋₁,
								indexθ_pc₁=indexθ_pc₁,
								indexθ_pcₜcₜ₋₁=indexθ_pcₜcₜ₋₁,
								indexθ_ψ=indexθ_ψ,
								γ=γ,
								K=K,
								∇ℓglm=∇ℓglm,
								∇ℓlatent=zeros(13),
								∇pa₁ = collect(zeros(Ξ) for q=1:nθ_pa₁),
								πᶜ=πᶜ,
								∇πᶜ=∇πᶜ,
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
	update!(model, memory, concatenatedθ)

Update the model and the memory quantities according to new parameter values

MODIFIED ARGUMENT
-`memory`: structure containing variables memory between computations of the model's log-likelihood and its gradient
-`model`: structure with information concerning a factorial hidden Markov drift-diffusion model

ARGUMENT
-`concatenatedθ`: newest values of the model's parameters

RETURN
-`P`: an instance of `Probabilityvector`

EXAMPLE
```julia-repl
julia> using FHMDDM
julia> model = Model("/mnt/cup/labs/brody/tzluo/analysis_data/analysis_2022_04_27_test/data.mat"; randomize=true);
julia> memory, P = FHMDDM.Memoryforgradient(model)
julia> P = update!(model, memory, rand(length(memory.concatenatedθ)))
```
"""
function update!(memory::Memoryforgradient,
				 model::Model,
				 concatenatedθ::Vector{<:Real})
	memory.concatenatedθ .= concatenatedθ
	sortparameters!(model, memory.concatenatedθ, memory.indexθ)
	real2native!(model.θnative, model.options, model.θreal)
	if !isempty(memory.p𝐘𝑑[1][1][1])
	    likelihood!(memory.p𝐘𝑑, model.trialsets, model.θnative.ψ[1])
	end
	@unpack options, θnative = model
	@unpack Δt, K, Ξ = options
	P = Probabilityvector(Δt, θnative, Ξ)
	update_for_∇transition_probabilities!(P)
	∇transitionmatrix!(memory.∇Aᵃsilent, memory.Aᵃsilent, P)
	if K == 2
		Aᶜ₁₁ = θnative.Aᶜ₁₁[1]
		Aᶜ₂₂ = θnative.Aᶜ₂₂[1]
		πᶜ₁ = θnative.πᶜ₁[1]
		memory.Aᶜ .= [Aᶜ₁₁ 1-Aᶜ₂₂; 1-Aᶜ₁₁ Aᶜ₂₂]
		memory.πᶜ .= [πᶜ₁, 1-πᶜ₁]
	end
	return P
end
