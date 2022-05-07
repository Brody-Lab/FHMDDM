"""
	maximize_choice_posterior!(model)

Learn the parameters that maximize the L2-regularized log-likelihood of the behavioral choices

OPTIONAL ARGUMENT
-`extended_trace`: save additional information
-`f_tol`: threshold for determining convergence in the objective value
-`g_tol`: threshold for determining convergence in the gradient
-`iterations`: number of inner iterations that will be run before the algorithm gives up
-`outer_iterations`: number of outer iterations that will be run before the algorithm gives up
-`show_every`: trace output is printed every `show_every`th iteration.
-`show_trace`: should a trace of the optimization algorithm's state be shown?
-`x_tol`: threshold for determining convergence in the input vector

EXAMPLE
```julia-repl
julia> using FHMDDM
julia> datapath = "/mnt/cup/labs/brody/tzluo/analysis_data/analysis_2022_05_07a/T176_2018_05_03/data.mat"
julia> model = Model(datapath)
julia> FHMDDM.maximize_choice_posterior!(model)
```
"""
function maximize_choice_posterior!(model::Model;
						 L2coefficient::Real=0.1,
		                 extended_trace::Bool=true,
		                 f_tol::AbstractFloat=1e-9,
		                 g_tol::AbstractFloat=1e-8,
		                 iterations::Integer=1000,
		                 show_every::Integer=10,
		                 show_trace::Bool=true,
		                 x_tol::AbstractFloat=1e-5)
	memory = Memoryforgradient(model; choicemodel=true)
    f(concatenatedθ) = -choiceLL!(memory, model, concatenatedθ) + L2coefficient*dot(concatenatedθ,concatenatedθ)
	function g!(∇, concatenatedθ)
		∇negativechoiceLL!(∇, memory, model, concatenatedθ)
		∇ .+= 2.0.*L2coefficient.*concatenatedθ
		return nothing
	end
    Optim_options = Optim.Options(extended_trace=extended_trace,
								  f_tol=f_tol,
                                  g_tol=g_tol,
                                  iterations=iterations,
                                  show_every=show_every,
                                  show_trace=show_trace,
                                  x_tol=x_tol)
	algorithm = LBFGS(linesearch = LineSearches.BackTracking())
	θ₀ = concatenate_choice_related_parameters(model)[1]
	optimizationresults = Optim.optimize(f, g!, θ₀, algorithm, Optim_options)
    println(optimizationresults)
	θₘₗ = Optim.minimizer(optimizationresults)
	sortparameters!(model, θₘₗ, memory.indexθ.latentθ)
end

"""
	maximizechoiceLL!(model)

Learn the parameters that maximize the log-likelihood of the behavioral choices

OPTIONAL ARGUMENT
-see documentation of 'maximize_choice_posterior'
"""
function maximizechoiceLL!(model::Model;
		                 extended_trace::Bool=true,
		                 f_tol::AbstractFloat=1e-9,
		                 g_tol::AbstractFloat=1e-8,
		                 iterations::Integer=1000,
		                 show_every::Integer=10,
		                 show_trace::Bool=true,
		                 x_tol::AbstractFloat=1e-5)
	memory = Memoryforgradient(model; choicemodel=true)
    f(concatenatedθ) = -choiceLL!(memory, model, concatenatedθ)
    g!(∇, concatenatedθ) = ∇negativechoiceLL!(∇, memory, model, concatenatedθ)
    Optim_options = Optim.Options(extended_trace=extended_trace,
								  f_tol=f_tol,
                                  g_tol=g_tol,
                                  iterations=iterations,
                                  show_every=show_every,
                                  show_trace=show_trace,
                                  x_tol=x_tol)
	algorithm = LBFGS(linesearch = LineSearches.BackTracking())
	θ₀ = concatenate_choice_related_parameters(model)[1]
	optimizationresults = Optim.optimize(f, g!, θ₀, algorithm, Optim_options)
    println(optimizationresults)
	θₘₗ = Optim.minimizer(optimizationresults)
	sortparameters!(model, θₘₗ, memory.indexθ.latentθ)
end

"""
	choiceLL!(memory, model, concatenatedθ)

Log-likelihood of the choices

MODIFIED ARGUMENT
-`model`: an instance of FHM-DDM
-`memory`: a container of variables used by both the log-likelihood and gradient computation

UNMODIFIED ARGUMENT
-`concatenatedθ`: a vector of concatenated parameter values

RETURN
-log-likelihood
"""
function choiceLL!(memory::Memoryforgradient,
					model::Model,
					concatenatedθ::Vector{<:Real})
	if concatenatedθ != memory.concatenatedθ
		P = update_for_choiceLL!(memory, model, concatenatedθ)
		memory.ℓ[1] = 0.0
		@inbounds for trialset in model.trialsets
			for trial in trialset.trials
				choiceLL!(memory, P, model.θnative, trial)
			end
		end
	end
	memory.ℓ[1]
end

"""
	choiceLL!(memory, P, θnative, trial)

Log-likelihood of the choice in a single trial

MODIFIED ARGUMENT
-`memory`: memory allocated for computing the gradient. The log-likelihood is updated.
-`P`: a structure containing allocated memory for computing the accumulator's initial and transition probabilities as well as the partial derivatives of these probabilities

UNMODIFIED ARGUMENT
-`θnative`: parameters of the latent variables in native space
-`trial`: a struct containing the stimuli and behavioral response of one trial

RETURN
-log-likelihood
"""
function choiceLL!(memory::Memoryforgradient, P::Probabilityvector, θnative::Latentθ, trial::Trial)
	@unpack clicks = trial
	@unpack Aᵃinput, Aᵃsilent, ℓ, πᶜᵀ = memory
	priorprobability!(P, trial.previousanswer)
	f = copy(P.𝛑)
	if length(clicks.time) > 0
		adaptedclicks = adapt(clicks, θnative.k[1], θnative.ϕ[1])
	end
	@inbounds for t = 2:trial.ntimesteps
		if isempty(clicks.inputindex[t])
			Aᵃ = Aᵃsilent
		else
			Aᵃ = Aᵃinput[clicks.inputindex[t][1]]
			update_for_transition_probabilities!(P, adaptedclicks, clicks, t)
			transitionmatrix!(Aᵃ, P)
		end
		f = Aᵃ*f
	end
	conditional_probability_of_choice!(f, trial.choice, θnative.ψ[1])
	ℓ[1] += log(sum(f))
	return nothing
end

"""
	choiceLL(concatenatedθ, indexθ, model)

ForwardDiff-compatible computation of the log-likelihood of the choices

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
julia> model = Model("/mnt/cup/labs/brody/tzluo/analysis_data/analysis_2022_05_05_test/data.mat"; randomize=true);
julia> concatenatedθ, indexθ = FHMDDM.concatenate_choice_related_parameters(model)
julia> ℓ = FHMDDM.choiceLL(concatenatedθ, indexθ.latentθ, model)
julia> memory = FHMDDM.Memoryforgradient(model; choicemodel=true)
julia> ℓ2 = FHMDDM.choiceLL!(memory, model, concatenatedθ) #ForwardDiff-incompatible
julia> abs(ℓ2-ℓ)
```
"""
function choiceLL(concatenatedθ::Vector{T},
				indexθ::Latentθ,
				model::Model) where {T<:Real}
	model = Model(concatenatedθ, indexθ, model)
	@unpack options, θnative, θreal, trialsets = model
	@unpack Δt, Ξ = options
	Aᵃinput, Aᵃsilent = zeros(T,Ξ,Ξ), zeros(T,Ξ,Ξ)
	expλΔt = exp(θnative.λ[1]*Δt)
	dμ_dΔc = differentiate_μ_wrt_Δc(Δt, θnative.λ[1])
	d𝛏_dB = (2 .*collect(1:Ξ) .- Ξ .- 1)./(Ξ-2)
	𝛏 = θnative.B[1].*d𝛏_dB
	transitionmatrix!(Aᵃsilent, expλΔt.*𝛏, √(Δt*θnative.σ²ₐ[1]), 𝛏)
	ℓ = zero(T)
	@inbounds for s in eachindex(trialsets)
		for m in eachindex(trialsets[s].trials)
			trial = trialsets[s].trials[m]
			f = probabilityvector(θnative.μ₀[1]+θnative.wₕ[1]*trial.previousanswer, √θnative.σ²ᵢ[1], 𝛏)
			if length(trial.clicks.time) > 0
				adaptedclicks = adapt(trial.clicks, θnative.k[1], θnative.ϕ[1])
			end
			for t=2:trial.ntimesteps
				if t ∈ trial.clicks.inputtimesteps
					cL = sum(adaptedclicks.C[trial.clicks.left[t]])
					cR = sum(adaptedclicks.C[trial.clicks.right[t]])
					𝛍 = expλΔt.*𝛏 .+ (cR-cL)*dμ_dΔc
					σ = √((cR+cL)*θnative.σ²ₛ[1] + Δt*θnative.σ²ₐ[1])
					transitionmatrix!(Aᵃinput, 𝛍, σ, 𝛏)
					Aᵃ = Aᵃinput
				else
					Aᵃ = Aᵃsilent
				end
				f = Aᵃ*f
			end
			conditional_probability_of_choice!(f, trial.choice, θnative.ψ[1])
			ℓ+=log(sum(f))
		end
	end
	ℓ
end


"""
	∇negativechoiceLL!(∇nℓ, memory, model, concatenatedθ)

Update the gradient of the negative log-likelihood of choices

MODIFIED ARGUMENT
-`∇nℓ`: the vector representing the gradient
-`memory`: memory allocated for computing the gradient. The log-likelihood is updated.
-`model`: structure containing the data, parameters, and hyperparameters of the model

ARGUMENT
-`concatenatedθ`: values of the model's choice-related parameters concatenated into a vector

EXAMPLE
```julia-repl
julia> using FHMDDM
julia> model = Model("/mnt/cup/labs/brody/tzluo/analysis_data/analysis_2022_05_05_test/data.mat"; randomize=true);
julia> concatenatedθ, indexθ = FHMDDM.concatenate_choice_related_parameters(model)
julia> ∇nℓ = similar(concatenatedθ)
julia> memory = FHMDDM.Memoryforgradient(model; choicemodel=true)
julia> FHMDDM.∇negativechoiceLL!(∇nℓ, memory, model, concatenatedθ)
julia> using ForwardDiff
julia> f(x) = -FHMDDM.choiceLL(x, indexθ.latentθ, model)
julia> ∇nℓ_auto = ForwardDiff.gradient(f, concatenatedθ)
julia> maximum(abs.(∇nℓ_auto .- ∇nℓ))

julia> using FHMDDM
julia> model = Model("/mnt/cup/labs/brody/tzluo/analysis_data/analysis_2022_05_05_test/data.mat"; randomize=true);
julia> concatenatedθ, indexθ = FHMDDM.concatenate_choice_related_parameters(model)
julia> concatenatedθ = rand(length(concatenatedθ))
julia> ℓ = FHMDDM.choiceLL(concatenatedθ, indexθ.latentθ, model)
julia> ∇nℓ = similar(concatenatedθ)
julia> memory = FHMDDM.Memoryforgradient(model; choicemodel=true)
julia> FHMDDM.∇negativechoiceLL!(∇nℓ, memory, model, concatenatedθ)
julia> ℓ = FHMDDM.choiceLL(concatenatedθ, indexθ.latentθ, model)
julia> abs(ℓ - memory.ℓ[1])
```

```compare speeds of automatic and hand-coded gradients
julia> using FHMDDM, ForwardDiff, BenchmarkTools
julia> model = Model("/mnt/cup/labs/brody/tzluo/analysis_data/analysis_2022_05_07a/T176_2018_05_03/data.mat"; randomize=true);
julia> concatenatedθ, indexθ = FHMDDM.concatenate_choice_related_parameters(model);
julia> memory = FHMDDM.Memoryforgradient(model; choicemodel=true);
julia> ghand!(∇, concatenatedθ) = FHMDDM.∇negativechoiceLL!(∇, memory, model, concatenatedθ);
julia> f(x) = FHMDDM.choiceLL(x, indexθ.latentθ, model);
julia> gauto!(∇, x) = ForwardDiff.gradient!(∇, f, x);
julia> g1, g2 = similar(concatenatedθ), similar(concatenatedθ);
julia> ghand!(g1, concatenatedθ);
julia> ghand!(g2, concatenatedθ);
julia> maximum(abs.(g1.-g2))
julia> @benchmark ghand!(g1, concatenatedθ) #4.6s
julia> @benchmark gauto!(g2, concatenatedθ) #9.2s
```
"""
function ∇negativechoiceLL!(∇nℓ::Vector{<:Real},
							memory::Memoryforgradient,
							model::Model,
						    concatenatedθ::Vector{<:Real})
	if concatenatedθ != memory.concatenatedθ
		P = update_for_choiceLL!(memory, model, concatenatedθ)
	else
		P = Probabilityvector(model.options.Δt, model.θnative, model.options.Ξ)
	end
	∇choiceLL!(memory,model,P)
	indexall = 0
	indexfit = 0
	for field in fieldnames(Latentθ)
		indexall+=1
		if (getfield(memory.indexθ.latentθ, field)[1] > 0) && (field != :Aᶜ₁₁) && (field != :Aᶜ₂₂) && (field != :πᶜ₁)
			indexfit +=1
			∇nℓ[indexfit] = -memory.∇ℓlatent[indexall]
		end
	end
	native2real!(∇nℓ, memory.indexθ.latentθ, model)
end

"""
	∇choiceLL!(memory, model, P)

Update the gradient of the log-likelihood of the choices across trialsets

MODIFIED ARGUMENT
-`memory`: memory allocated for computing the gradient. The log-likelihood is updated.
-`P`: a structure containing allocated memory for computing the accumulator's initial and transition probabilities as well as the partial derivatives of these probabilities

UNMODIFIED ARGUMENT
-`θnative`: parameters of the latent variables in native space
-`trial`: a struct containing the stimuli and behavioral response of one trial
"""
function ∇choiceLL!(memory::Memoryforgradient, model::Model, P::Probabilityvector)
	memory.ℓ .= 0.0
	memory.∇ℓlatent .= 0.0
	@inbounds for trialset in model.trialsets
		for trial in trialset.trials
			∇choiceLL!(memory, P, model.θnative, trial)
		end
	end
	return nothing
end

"""
	∇choiceLL!(memory, P, θnative, trial)

Update the gradient of the log-likelihood of the choice in one trial

MODIFIED ARGUMENT
-`memory`: memory allocated for computing the gradient. The log-likelihood is updated.
-`P`: a structure containing allocated memory for computing the accumulator's initial and transition probabilities as well as the partial derivatives of these probabilities

UNMODIFIED ARGUMENT
-`θnative`: parameters of the latent variables in native space
-`trial`: a struct containing the stimuli and behavioral response of one trial
"""
function ∇choiceLL!(memory::Memoryforgradient,
					P::Probabilityvector,
					θnative::Latentθ,
					trial::Trial)
	@unpack clicks = trial
	@unpack inputtimesteps, inputindex = clicks
	@unpack Aᵃinput, ∇Aᵃinput, Aᵃsilent, ∇Aᵃsilent, D, f, indexθ_pa₁, indexθ_paₜaₜ₋₁, indexθ_ψ, ℓ, ∇ℓlatent, nθ_pa₁, nθ_paₜaₜ₋₁, ∇pa₁, Ξ = memory
	t = 1
	∇priorprobability!(∇pa₁, P, trial.previousanswer)
	f[t] .= P.𝛑
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
		f[t] = Aᵃ * f[t-1]
	end
	p𝑑_a = conditional_probability_of_choice(trial.choice, θnative.ψ[1], Ξ)
	p𝑑 = dot(p𝑑_a, f[trial.ntimesteps])
	ℓ[1] += log(p𝑑)
	b = p𝑑_a./p𝑑 # backward term for the last time step
	γ = b.*f[trial.ntimesteps] # posterior probability for the last time step
	∇ℓlatent[indexθ_ψ[1]] += expectation_derivative_logp𝑑_wrt_ψ(trial.choice, γ, θnative.ψ[1])
	@inbounds for t = trial.ntimesteps:-1:1
		if t < trial.ntimesteps
			if t+1 ∈ clicks.inputtimesteps
				clickindex = clicks.inputindex[t+1][1]
				Aᵃₜ₊₁ = Aᵃinput[clickindex]
			else
				Aᵃₜ₊₁ = Aᵃsilent
			end
			b = transpose(Aᵃₜ₊₁) * b
		end
		if t > 1
			if t ∈ clicks.inputtimesteps
				clickindex = clicks.inputindex[t][1]
				∇Aᵃ = ∇Aᵃinput[clickindex]
			else
				∇Aᵃ = ∇Aᵃsilent
			end
			for i = 1:nθ_paₜaₜ₋₁
				∇ℓlatent[indexθ_paₜaₜ₋₁[i]] += (transpose(b)*∇Aᵃ[i]*f[t-1])[1]
			end
		end
	end
	@inbounds for i = 1:nθ_pa₁
		∇ℓlatent[indexθ_pa₁[i]] += dot(b, ∇pa₁[i])
	end
	return nothing
end

"""
    conditional_probability_of_choice!(f, choice, ψ)

Probability of a choice conditioned on the accumulator state

ARGUMENT
-`choice`: the observed choice, either right (`choice`=true) or left.
-`ψ`: the prior probability of a lapse state

RETURN
`p`: conditional probability of the choice
"""
function conditional_probability_of_choice(choice::Bool, ψ::T, Ξ::Integer) where {T<:Real}
	p = zeros(T, Ξ)
	zeroindex = cld(Ξ,2)
    p[zeroindex] = 0.5
    if choice
        p[1:zeroindex-1]   .= ψ/2
        p[zeroindex+1:end] .= 1-ψ/2
    else
        p[1:zeroindex-1]   .= 1-ψ/2
        p[zeroindex+1:end] .= ψ/2
    end
	p
end

"""
    conditional_probability_of_choice!(f, choice, ψ)

Probability of a choice conditioned on the accumulator state

MODIFIED ARGUMENT
-`f`: the forward term

ARGUMENT
-`choice`: the observed choice, either right (`choice`=true) or left.
-`ψ`: the prior probability of a lapse state
"""
function conditional_probability_of_choice!(f::Array{<:Real}, choice::Bool, ψ::Real)
	Ξ = length(f)
	zeroindex = cld(Ξ,2)
    f[zeroindex] *= 0.5
    if choice
        f[1:zeroindex-1]   .*= ψ/2
        f[zeroindex+1:end] .*= 1-ψ/2
    else
        f[1:zeroindex-1]   .*= 1-ψ/2
        f[zeroindex+1:end] .*= ψ/2
    end
    return nothing
end

"""
	update_for_choiceLL!(model, memory, concatenatedθ)

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
julia> memory, P = FHMDDM.Memoryforgradient(model; choicemodel=true)
julia> P = update_for_choiceLL!(model, memory, rand(length(memory.concatenatedθ)))
"""
function update_for_choiceLL!(memory::Memoryforgradient,
							 model::Model,
							 concatenatedθ::Vector{<:Real})
	memory.concatenatedθ .= concatenatedθ
	sortparameters!(model, memory.concatenatedθ, memory.indexθ.latentθ)
	real2native!(model.θnative, model.options, model.θreal)
	@unpack options, θnative = model
	@unpack Δt, K, Ξ = options
	P = Probabilityvector(Δt, θnative, Ξ)
	update_for_∇transition_probabilities!(P)
	∇transitionmatrix!(memory.∇Aᵃsilent, memory.Aᵃsilent, P)
	return P
end
