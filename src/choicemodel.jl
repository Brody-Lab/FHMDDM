"""
	fitonlychoices!(model)

Learn the parameters that specify the likelihood of the behavioral choices

MODIFIED ARGUMENT
-`model`: structure containing the parameters, hyperparameters, and data of a factorial hidden Markov drift-diffusion model. The fields 'θnative' and 'θreal' are updated
"""
function fitonlychoices!(model::Model)
	if isnan(model.options.α₀_choices)
		maximizechoiceLL!(model)
	else
		maximize_evidence_choices!(model)
	end
	return nothing
end

"""
	maximize_evidence_choices!(model)

Learn the parameters that govern the likelihood of the behavioral choices and the precisions of the priors on these parameters

MODIFIED ARGUMENT
-`model`: structure containing the parameters, hyperparameters, and data of a factorial hidden Markov drift-diffusion model. The parameters are updated.

EXAMPLE
```julia-repl
julia> using FHMDDM
julia> model = Model("/mnt/cup/labs/brody/tzluo/analysis_data/analysis_2022_07_06a_test/T176_2018_05_03_b5K1K1/data.mat")
julia> FHMDDM.maximize_evidence_choices!(model)
julia> λΔt, pchoice = expectedemissions(model;nsamples=2)
julia> save(model, λΔt, pchoice)
julia>
```
"""
function maximize_evidence_choices!(model::Model;
								iterations::Int = 500,
								max_consecutive_failures::Int=2,
								outer_iterations::Int=10,
								verbose::Bool=true,
								g_tol::Real=1e-8,
								x_reltol::Real=1e-1)
	memory = FHMDDM.Memoryforgradient(model; choicemodel=true)
	best𝛉, index𝛉 = FHMDDM.concatenate_choice_related_parameters(model)
	best𝐸 = -Inf
	𝛂, index𝛂 = FHMDDM.choice_related_precisions(model)
	best𝛂 = copy(𝛂)
	n_consecutive_failures = 0
	posteriorconverged = false
	for i = 1:outer_iterations
	    results = FHMDDM.maximize_choice_posterior!(model; 𝛂=𝛂, iterations=iterations, g_tol=g_tol)
		if !Optim.converged(results)
			if Optim.iteration_limit_reached(results)
				new_α = min(100.0, 2geomean(𝛂))
				verbose && println("Outer iteration: ", i, ": because the maximum number of iterations was reached, the values of the precisions are set to be twice the geometric mean of the hyperparameters. New 𝛂  → ", new_α)
				𝛂 .= new_α
			else
				verbose && println("Outer iteration: ", i, ": because of a line search failure, the values of latent-variable parameters are randomized")
				randomize_latent_parameters!(model)
			end
		else
			verbose && println("Outer iteration: ", i, ": the MAP values of the parameters converged")
			𝛉₀ = FHMDDM.concatenate_choice_related_parameters(model)[1] # exact posterior mode
			stats = @timed FHMDDM.∇∇choiceLL(model)
			𝐇 = stats.value[3][index𝛂, index𝛂]
			verbose && println("Outer iteration: ", i, ": computing the Hessian of the log-likelihood took ", stats.time, " seconds")
			FHMDDM.choiceLL!(memory, model, 𝛉₀)
			𝐸 = FHMDDM.logevidence(𝐇, memory.ℓ[1], Diagonal(𝛂), 𝛉₀)
			if 𝐸 > best𝐸
				if verbose
					if posteriorconverged
						println("Outer iteration: ", i, ": the log-evidence (best: ", best𝐸, "; new:", 𝐸, ") is improved by the new values of the precisions found in the previous outer iteration")
					else
						println("Outer iteration: ", i, ": initial value of log-evidence: ", 𝐸, " is set as the best log-evidence")
					end
				end
				best𝐸 = 𝐸
				best𝛂 .= 𝛂
				best𝛉 .= 𝛉₀
				n_consecutive_failures = 0
			else
				n_consecutive_failures += 1
				verbose && println("Outer iteration: ", i, ": because the log-evidence (best: ", best𝐸, "; new:", 𝐸, ") was not improved by the new precisions, subsequent learning of the precisions will be begin at the midpoint between the current values of the precisions and the values that gave the best evidence so far.")
				for j in eachindex(𝛂)
					𝛂[j] = (𝛂[j] + best𝛂[j])/2
				end
			end
			posteriorconverged = true
			if n_consecutive_failures == max_consecutive_failures
				verbose && println("Outer iteration: ", i, ": optimization halted early due to ", max_consecutive_failures, " consecutive failures in improving evidence")
				break
			end
			𝛂, normΔ = maximize_evidence_choices!(memory, model, 𝛂, 𝐇, 𝛉₀)
			if verbose
				println("Outer iteration ", i, ": new 𝛂 → ", 𝛂)
			end
			if normΔ < x_reltol
				verbose && println("Outer iteration: ", i, ": optimization halted after relative difference in the norm of the hyperparameters (in real space) decreased below ", x_reltol)
				break
			else
				randomize_latent_parameters!(model)
			end
		end
		if (i==outer_iterations) && verbose
			println("Optimization halted after reaching the last of ", outer_iterations, " allowed outer iterations.")
		end
	end
	println("Best log-evidence: ", best𝐸)
	println("Best shrinkage coefficients: ", best𝛂)
	println("Best parameters: ", best𝛉)
	sortparameters!(model, best𝛉, index𝛉.latentθ)
	real2native!(model.θnative, model.options, model.θreal)
	return nothing
end

"""
	choice_related_precisions(model)

Concatenate the precisions of the priors on each drift-diffusion parameter that is being fit

ARGUMENT
-`model`: structure containing the data, hyperparameters, and parameters

RETURN
-a vector concatenating the precisions of the priors on the drift-diffusion parameters that are being fit
-a 10-element BitVector indicating which of the precisions are being learned
"""
function choice_related_precisions(model::Model)
	concatenatedθ, indexθ = concatenate_choice_related_parameters(model)
	𝛂 = similar(concatenatedθ)
	index𝛂 = falses(10)
	j = 0
	k = 0
	for parametername in fieldnames(Latentθ)
		if parametername == :Aᶜ₁₁ || parametername == :Aᶜ₂₂ || parametername == :πᶜ₁
 		else
			j = j + 1
			if getfield(indexθ.latentθ, parametername)[1] > 0
				k = k + 1
				𝛂[k] = model.options.α₀_choices
				index𝛂[j] = true
			end
		end
	end
	return 𝛂, index𝛂
end

"""
	maximize_evidence_choices!(memory, model, 𝐇, 𝛉₀)

Learn hyperparameters by fixing the parameters of the model and maximizing the evidence of only the choices

MODIFIED ARGUMENT
-`𝛂`: precision of the priors on the drift-diffusion parameters
-`memory`: structure containing variables to be modified during computations
-`model`: structure containing the parameters, hyperparameters, and data. The parameter values are modified, but the hyperparameters are not modified
-`𝐇`: Hessian of the log-likelihood evaluated at the MAP solution `𝛉₀`, containing only the parameters associated with hyperparameters that are being optimized
-`𝛉₀`: exact MAP solution

RETURN
-Euclidean of the normalized difference in the log of the hyperparameters being optimized
"""
function maximize_evidence_choices!(memory::Memoryforgradient,
						model::Model,
						𝛂₀::Vector{<:Real},
						𝐇::Matrix{<:Real},
						𝐰₀::Vector{<:Real};
						αrange::Vector{<:Real}=[1e-2, 1e2],
						optimizationoptions::Optim.Options=Optim.Options(iterations=15, show_trace=true, show_every=1),
						optimizer::Optim.FirstOrderOptimizer=LBFGS(linesearch=LineSearches.BackTracking()))
	𝚽 = Diagonal(𝛂₀)
	𝐁₀𝐰₀ = (𝚽-𝐇)*𝐰₀
	𝐱₀ = similar(𝛂₀)
	for i in eachindex(𝛂₀)
		𝐱₀[i] = FHMDDM.native2real(𝛂₀[i], αrange[1], αrange[2])
	end
	function f(𝐱)
		𝛂 = similar(𝐱)
		for i in eachindex(𝐱)
			𝛂[i] = real2native(𝐱[i], αrange[1], αrange[2])
		end
		𝚽 = Diagonal(𝛂)
	    𝐰 = (𝚽-𝐇) \ 𝐁₀𝐰₀ # LAPACK.sysv! uses less memory but is slower
		choiceLL!(memory, model, 𝐰)
		return -logevidence(𝐇, memory.ℓ[1], 𝚽, 𝐰)
	end
	∇nℓ = similar(𝐰₀)
	function g!(∇n𝐸, 𝐱)
		𝛂 = similar(𝐱)
		for i in eachindex(𝐱)
			𝛂[i] = real2native(𝐱[i], αrange[1], αrange[2])
		end
		∇negativelogevidence_choices!(memory, model, ∇n𝐸, ∇nℓ, 𝛂, 𝐁₀𝐰₀, 𝐇)
		for i in eachindex(𝐱)
			∇n𝐸[i] *= differentiate_native_wrt_real(𝐱[i], αrange[1], αrange[2])
		end
		return nothing
	end
	optimizationresults = Optim.optimize(f, g!, 𝐱₀, optimizer, optimizationoptions)
	# function f(𝐱)
	# 	𝛂 = similar(𝐱)
	# 	for i in eachindex(𝐱)
	# 		𝛂[i] = real2native(𝐱[i], αrange[1], αrange[2])
	# 	end
	# 	-log_evidence_choices(𝛂, 𝐁₀𝐰₀, 𝐇, model)
	# end
	# optimizationresults = Optim.optimize(f, 𝐱₀, optimizer, optimizationoptions; autodiff = :forward)
	𝐱̂ = Optim.minimizer(optimizationresults)
	normΔ = 0.0
	for i in eachindex(𝐱̂)
		normΔ += (𝐱̂[i]/𝐱₀[i] - 1.0)^2
	end
	𝛂̂ = similar(𝐱̂)
	for i in eachindex(𝐱̂)
		𝛂̂[i] = real2native(𝐱̂[i], αrange[1], αrange[2])
	end
	return 𝛂̂, √normΔ
end

"""
	∇negativelogevidence_choices!(memory, model, ∇n𝐸, 𝛂, 𝐇, 𝐁₀𝛉ₘₐₚ)

gradient of the negative log of the marginal likelihood

MODIFIED ARGUMENT
-`memory`: a structure containing memory for in-place computation
-`model`: structure containing the parameters, hyperparameters, and data
-`∇n𝐸`: memory for in-place computation of the gradient of the negative of the log-evidence
-`∇nℓ`: memory for in-place computation of the gradient of the negative of the log-evidence
-`𝛉`: memory for in-place computation of the approximate posterior mode as a function of the hyperparameters

UNMODIFIED ARGUMENT
-`𝛂𝐬`: concatenated values of the L2 penalties coefficients
-`𝐁₀𝐰₀`: Hessian of the log-posterior evalued at the MAP values of the parameters multiplied by the MAP value of the parameters, containing only the parameters associated with the hyperparameters being optimized
-`𝐇`: Hessian of the log-likelihood evaluated at the MAP values of the parameters, containing only the parameters associated with the hyperparameters being optimized
"""
function ∇negativelogevidence_choices!(memory::Memoryforgradient,
								model::Model,
								∇n𝐸::Vector{<:Real},
								∇nℓ::Vector{<:Real},
								𝛂::Vector{<:Real},
								𝐁₀𝐰₀::Vector{<:Real},
								𝐇::Matrix{<:Real})
	𝚽 = Diagonal(𝛂)
	𝐁 = 𝚽-𝐇
	C = factorize(𝐁)
	𝐰 = C \ 𝐁₀𝐰₀
	∇negativechoiceLL!(∇nℓ, memory, model, 𝐰)
	𝐦 = C \ (𝚽*𝐰 + ∇nℓ)
	𝛀 = (C \ (𝚽 \ 𝐇)')'
	@inbounds for i in eachindex(𝐰)
		∇n𝐸[i] = 0.5*(𝐰[i]^2 + 𝛀[i,i]) - 𝐰[i]*𝐦[i]
	end
	return nothing
end

"""
	check_∇logevidence(model)

Check whether the hand-coded gradient of the log-evidence matches the automatic gradient

ARGUMENT
-`model`: structure containing the parameters and hyperparameters

OPTIONAL ARGUMENT
-`simulate`: whether to simulate Hessian and MAP solution. If not, the model is first fitted before a Hessian is computed

RETURN
-maximum absolute normalized difference between the gradients
-absolute normalized difference between the log-evidence functions

EXAMPLE
```julia-repl
julia> using FHMDDM
julia> model = Model("/mnt/cup/labs/brody/tzluo/analysis_data/analysis_2022_07_06a_test/T176_2018_05_03_b5K1K1/data.mat")
julia> max_abs_norm_diff_∇𝐸, abs_norm_diff_𝐸 = FHMDDM.check_∇logevidence_choices(model; simulate=true)
julia>
julia> using FHMDDM
julia> model = Model("/mnt/cup/labs/brody/tzluo/analysis_data/analysis_2022_07_06a_test/T176_2018_05_03_b5K1K1/data.mat")
julia> max_abs_norm_diff_∇𝐸, abs_norm_diff_𝐸 = FHMDDM.check_∇logevidence_choices(model; simulate=false)
julia>
```
"""
function check_∇logevidence_choices(model::Model; simulate::Bool=true)
	𝛉, index𝛉 = FHMDDM.concatenate_choice_related_parameters(model)
	𝛂, index𝛂 = FHMDDM.choice_related_precisions(model)
	𝚽 = Diagonal(𝛂)
	N = length(𝛂)
	if simulate
		𝐑 = 1 .- 2rand(N,N)
		𝐁₀ = transpose(𝐑)*𝐑 # simulate a positive-definite Hessian of the posterior
		𝐇 = 𝚽 - 𝐁₀
		𝐰₀ = 1 .- 2rand(N)
		𝐰₀ ./= norm(𝐰₀)
		𝐁₀𝐰₀ = 𝐁₀*𝐰₀
		𝛉 .= 𝐰₀
		FHMDDM.sortparameters!(model, 𝛉, index𝛉.latentθ)
		FHMDDM.real2native!(model.θnative, model.options, model.θreal)
	else
		FHMDDM.maximize_choice_posterior!(model)
		𝐇 = FHMDDM.∇∇choiceLL(model)[3][index𝛂, index𝛂]
		𝐰₀ = FHMDDM.concatenate_choice_related_parameters(model)[1]
		𝐁₀𝐰₀ = (𝚽-𝐇)*𝐰₀
	end
	𝛂 .= rand(N)
	∇nℓ = similar(𝛉)
	memory = FHMDDM.Memoryforgradient(model;choicemodel=true)
	𝚽 = Diagonal(𝛂)
	𝛉 = (𝚽-𝐇) \ 𝐁₀𝐰₀
	FHMDDM.choiceLL!(memory, model, 𝛉)
	handcoded_evidence = FHMDDM.logevidence(𝐇, memory.ℓ[1], 𝚽, 𝛉)
	handcoded_gradient = fill(NaN,N)
	FHMDDM.∇negativelogevidence_choices!(memory, model, handcoded_gradient, ∇nℓ, 𝛂, 𝐁₀𝐰₀, 𝐇)
    f(x) = FHMDDM.log_evidence_choices(x, 𝐁₀𝐰₀, 𝐇, model)
	automatic_evidence = f(𝛂)
	automatic_gradient = ForwardDiff.gradient(f, 𝛂)
	return maximum(abs.((automatic_gradient .+ handcoded_gradient)./automatic_gradient)), abs((automatic_evidence-handcoded_evidence)/automatic_evidence)
end

"""
	logevidence(𝛂, 𝐁₀𝛉ₘₐₚ, 𝐇, indexθ, model)

ForwardDiff-computation evaluation of the log-evidence

ARGUMENT
-`𝛂`: precisions being learned
-`𝐁₀𝛉ₘₐₚ`: Hessian of the log-posterior evalued at the MAP values of the parameters multiplied by the MAP value of the parameters
-`𝐇`: Hessian of the log-likelihood evaluated at the MAP values of the parameters
-`index𝛂`: index of the precisions being fit within the full vector of concatenated precisions
-`indexθ`: index of the parameters
-`model`: structure containing the parameters, hyperparameters, and data

RETURN
-log of the marginal likelihood
"""
function log_evidence_choices(𝛂::Vector{type},
					𝐁₀𝐰₀::Vector{<:Real},
					𝐇::Matrix{<:Real},
					model::Model) where{type<:Real}
	𝚽 = Diagonal(𝛂)
	𝐁 = 𝚽-𝐇
    𝐰 = 𝐁 \ 𝐁₀𝐰₀
	index𝛉 = concatenate_choice_related_parameters(model)[2]
	ℓ = choiceLL(𝐰, index𝛉.latentθ, model)
	logevidence(𝐇, ℓ, 𝚽, 𝐰)
end

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

RETURN
-results from the optimization assembled by the Optim module


EXAMPLE
```julia-repl
julia> using FHMDDM
julia> datapath = "/mnt/cup/labs/brody/tzluo/analysis_data/analysis_2022_05_07a/T176_2018_05_03/data.mat"
julia> model = Model(datapath)
julia> FHMDDM.maximize_choice_posterior!(model)
```
"""
function maximize_choice_posterior!(model::Model;
						 𝛂::Vector{<:AbstractFloat}=[model.options.α₀_choices],
		                 extended_trace::Bool=true,
		                 f_tol::AbstractFloat=0.0,
		                 g_tol::AbstractFloat=1e-8,
		                 iterations::Integer=1000,
		                 show_every::Integer=10,
		                 show_trace::Bool=true,
		                 x_tol::AbstractFloat=0.0)
	memory = Memoryforgradient(model; choicemodel=true)
    f(concatenatedθ) = -choiceLL!(memory, model, concatenatedθ) + 0.5*dot(𝛂.*concatenatedθ, concatenatedθ)
	function g!(∇, concatenatedθ)
		∇negativechoiceLL!(∇, memory, model, concatenatedθ)
		∇ .+= 𝛂.*concatenatedθ
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
	real2native!(model.θnative, model.options, model.θreal)
	return optimizationresults
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
	real2native!(model.θnative, model.options, model.θreal)
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
function choiceLL(concatenatedθ::Vector{T}, indexθ::Latentθ, model::Model) where {T<:Real}
	model = Model(concatenatedθ, indexθ, model)
	@unpack options, θnative, θreal, trialsets = model
	@unpack Δt, minpa, Ξ = options
	Aᵃinput, Aᵃsilent = zeros(T,Ξ,Ξ), zeros(T,Ξ,Ξ)
	expλΔt = exp(θnative.λ[1]*Δt)
	dμ_dΔc = differentiate_μ_wrt_Δc(Δt, θnative.λ[1])
	d𝛏_dB = (2 .*collect(1:Ξ) .- Ξ .- 1)./(Ξ-2)
	𝛏 = θnative.B[1].*d𝛏_dB
	transitionmatrix!(Aᵃsilent, minpa, expλΔt.*𝛏, √(Δt*θnative.σ²ₐ[1]), 𝛏)
	ℓ = zero(T)
	@inbounds for s in eachindex(trialsets)
		for m in eachindex(trialsets[s].trials)
			trial = trialsets[s].trials[m]
			f = probabilityvector(minpa, θnative.μ₀[1]+θnative.wₕ[1]*trial.previousanswer, √θnative.σ²ᵢ[1], 𝛏)
			if length(trial.clicks.time) > 0
				adaptedclicks = adapt(trial.clicks, θnative.k[1], θnative.ϕ[1])
			end
			for t=2:trial.ntimesteps
				if t ∈ trial.clicks.inputtimesteps
					cL = sum(adaptedclicks.C[trial.clicks.left[t]])
					cR = sum(adaptedclicks.C[trial.clicks.right[t]])
					𝛍 = expλΔt.*𝛏 .+ (cR-cL)*dμ_dΔc
					σ = √((cR+cL)*θnative.σ²ₛ[1] + Δt*θnative.σ²ₐ[1])
					transitionmatrix!(Aᵃinput, minpa, 𝛍, σ, 𝛏)
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
julia> model = Model("/mnt/cup/labs/brody/tzluo/analysis_data/analysis_2022_05_21_test/T176_2018_05_03/data.mat");
julia> concatenatedθ, indexθ = FHMDDM.concatenate_choice_related_parameters(model)
julia> ∇nℓ = similar(concatenatedθ)
julia> memory = FHMDDM.Memoryforgradient(model; choicemodel=true)
julia> FHMDDM.∇negativechoiceLL!(∇nℓ, memory, model, concatenatedθ)
julia> using ForwardDiff
julia> f(x) = -FHMDDM.choiceLL(x, indexθ.latentθ, model)
julia> ∇nℓ_auto = ForwardDiff.gradient(f, concatenatedθ)
julia> maximum(abs.(∇nℓ_auto .- ∇nℓ))

julia> using FHMDDM
julia> model = Model("/mnt/cup/labs/brody/tzluo/analysis_data/analysis_2022_05_21_test/T176_2018_05_03/data.mat");
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
		P = Probabilityvector(model.options.Δt, model.options.minpa, model.θnative, model.options.Ξ)
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
    conditional_probability_of_choice(f, choice, ψ)

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
	@unpack Δt, K, minpa, Ξ = options
	P = Probabilityvector(Δt, minpa, θnative, Ξ)
	update_for_∇transition_probabilities!(P)
	∇transitionmatrix!(memory.∇Aᵃsilent, memory.Aᵃsilent, P)
	return P
end


"""
	update_drift_diffusion_transformation(model)

Update the transformation of the drift-diffusion parameters between real and native spaces.

Specifically, the hyperparameter that specifies the value of each drift-diffusion parameter in native space that corresponds to its value of zero in real space is updated.

ARGUMENT
-`model`: structure containing, the settings and hyperparameters of the model

RETURN
-`model`: structure containing the new settings and hyperparameters of the model

EXAMPLE
```julia-repl
julia> using FHMDDM
julia> datapath = "/mnt/cup/labs/brody/tzluo/analysis_data/analysis_2022_06_28b_test/T176_2018_05_03_b5K1K1/data.mat"
julia> model = Model(datapath)
julia> newmodel = FHMDDM.update_drift_diffusion_transformation(model)
```
"""
function update_drift_diffusion_transformation(model::Model)
	if model.options.updateDDtransformation
		dict = dictionary(model.options)
		dict["lqu_B"][2] = model.θnative.B[1]
		dict["lqu_k"][2] = model.θnative.k[1]
		dict["lqu_lambda"][2] = model.θnative.λ[1]
		dict["lqu_mu0"][2] = model.θnative.μ₀[1]
		dict["lqu_phi"][2] = model.θnative.ϕ[1]
		dict["lqu_psi"][2] = model.θnative.ψ[1]
		dict["lqu_sigma2_a"][2] = model.θnative.σ²ₐ[1]
		dict["lqu_sigma2_i"][2] = model.θnative.σ²ᵢ[1]
		dict["lqu_sigma2_s"][2] = model.θnative.σ²ₛ[1]
		dict["lqu_w_h"][2] = model.θnative.wₕ[1]
		Model(options=Options(dict),
			gaussianprior=model.gaussianprior,
			θnative = model.θnative,
			θreal = model.θreal,
			θ₀native = model.θ₀native,
			trialsets = model.trialsets)
	else
		model
	end
end

"""
	check_∇∇choiceLL(model)

Compare the automatically computed and hand-coded gradients and hessians with respect to the parameters being fitted in their real space

ARGUMENT
-`model`: a structure containing the data, parameters, and hyperparameters of a factorial hidden-Markov drift-diffusion model

RETURN
-`absdiffℓ`: absolute difference in the log-likelihood evaluted using the algorithm bein automatically differentiated and the hand-coded algorithm
-`absdiff∇`: absolute difference in the gradients
-`absdiff∇∇`: absolute difference in the hessians

EXAMPLE
```julia-repl
julia> using FHMDDM, ForwardDiff
julia> model = Model("/mnt/cup/labs/brody/tzluo/analysis_data/analysis_2022_07_06a_test/T176_2018_05_03_b5K1K1/data.mat")
julia> absdiffℓ, absdiff∇, absdiff∇∇ = FHMDDM.check_∇∇choiceLL(model)
julia> println("   max(|Δloss|): ", absdiffℓ)
julia> println("   max(|Δgradient|): ", maximum(absdiff∇))
julia> println("   max(|Δhessian|): ", maximum(absdiff∇∇))
julia>
```
"""
function check_∇∇choiceLL(model::Model)
	concatenatedθ, indexθ = concatenate_choice_related_parameters(model)
	ℓhand, ∇hand, ∇∇hand = ∇∇choiceLL(model)
	f(x) = choiceLL(x, indexθ.latentθ, model)
	ℓauto = f(concatenatedθ)
	∇auto = ForwardDiff.gradient(f, concatenatedθ)
	∇∇auto = ForwardDiff.hessian(f, concatenatedθ)
	return abs(ℓauto-ℓhand), abs.(∇auto .- ∇hand), abs.(∇∇auto .- ∇∇hand)
end

"""
	∇∇choiceLL(model)

Hessian of the log-likelihood of only the choices

ARGUMENT
-`model`: a structure containing the data and hyperparameters of the factorial hidden drift-diffusion model

RETURN
-`ℓ`: log-likelihood
-`∇ℓ`: gradient of the log-likelihood with respect to fitted parameters in real space
-`∇∇ℓ`: Hessian matrix of the log-likelihood with respect to fitted parameters in real space
"""
function ∇∇choiceLL(model::Model)
	@unpack trialsets = model
	memory = Memory_for_hessian_choiceLL(model)
	for trialset in trialsets
		for trial in trialset.trials
			∇∇choiceLL!(memory, model.θnative, trial)
		end
	end
	@unpack ℓ, ∇ℓ, ∇∇ℓ = memory
	for i = 1:size(∇∇ℓ,1)
		for j = i+1:size(∇∇ℓ,2)
			∇∇ℓ[j,i] = ∇∇ℓ[i,j]
		end
	end
	firstderivatives = differentiate_native_wrt_real(model)
	secondderivatives = differentiate_twice_native_wrt_real(model)
	for i = 1:memory.nθ
		d1 = getfield(firstderivatives, memory.parameternames[i])[1]
		d2 = getfield(secondderivatives, memory.parameternames[i])[1]
		∇∇ℓ[i,:] .*= d1
		∇∇ℓ[:,i] .*= d1
		∇∇ℓ[i,i] += d2*∇ℓ[i]
		∇ℓ[i] *= d1
	end
	return ℓ[1], ∇ℓ, ∇∇ℓ
end

"""
	 ∇∇choiceLL!(memory, θnative, trial)

Compute the hessian of the log-likelihood of the choice in one trial

MODIFIED ARGUMENT
-`memory`: a structure containing pre-allocated memory for in-place computation and also pre-computed quantities that are identical across trials. The hessian corresponds to the field `∇∇ℓ` within `memory.`

UNMODIFIED ARGUMENT
-`θnative`: values of the parameters controlling the latent varables in their native space
-`trial`: structure containing the auditory stimuli and behavioral choice of one trial
"""
function ∇∇choiceLL!(memory::Memory_for_hessian_choiceLL, θnative::Latentθ, trial::Trial)
	@unpack clicks = trial
	@unpack ℓ, ∇ℓ, ∇∇ℓ, f, ∇f, ∇D, ∇b, ∂p𝑑_∂ψ, P, ∇pa₁, ∇∇pa₁, indexθ_pa₁, indexθ_paₜaₜ₋₁, indexθ_ψ, nθ, nθ_pa₁, nθ_paₜaₜ₋₁, nθ_ψ, index_pa₁_in_θ, index_paₜaₜ₋₁_in_θ, index_ψ_in_θ, p𝑑, ∂p𝑑_∂ψ = memory
	∇∇priorprobability!(∇∇pa₁, ∇pa₁, P, trial.previousanswer)
	f[1] .= P.𝛑
	for q = 1:nθ
		i = index_pa₁_in_θ[q]
		if i == 0
			∇f[1][q] .= 0
		else
			∇f[1][q] = ∇pa₁[i]
		end
	end
	adaptedclicks = ∇∇adapt(clicks, θnative.k[1], θnative.ϕ[1])
	for t=2:trial.ntimesteps-1
		Aᵃ, ∇Aᵃ, ∇∇Aᵃ = ∇∇transitionmatrices!(memory, adaptedclicks, clicks, t)
		f[t] = Aᵃ * f[t-1]
		for q in eachindex(∇ℓ)
			i = index_paₜaₜ₋₁_in_θ[q]
			if i != 0
				∇f[t][q] = ∇Aᵃ[i] * f[t-1] .+ Aᵃ * ∇f[t-1][q]
			else
				∇f[t][q] = Aᵃ * ∇f[t-1][q]
			end
		end
	end
	t = trial.ntimesteps
	Aᵃ, ∇Aᵃ, ∇∇Aᵃ = ∇∇transitionmatrices!(memory, adaptedclicks, clicks, t)
	conditional_choice_likelihood!(p𝑑, trial.choice, θnative.ψ[1])
	differentiate_conditional_choice_likelihood_wrt_ψ!(∂p𝑑_∂ψ, trial.choice)
	f[t] = p𝑑.* (Aᵃ * f[t-1])
	D = sum(f[t])
	ℓ[1] += log(D)
	f[t] ./= D
	for q in eachindex(∇ℓ)
		i_aₜ = index_paₜaₜ₋₁_in_θ[q]
		i_ψ = index_ψ_in_θ[q]
		if i_aₜ > 0
			∇f[t][q] = p𝑑 .* (∇Aᵃ[i_aₜ] * f[t-1] .+ Aᵃ * ∇f[t-1][q])
		elseif i_ψ > 0
			∇f[t][q] = ∂p𝑑_∂ψ .* (Aᵃ * f[t-1]) .+ p𝑑 .* (Aᵃ * ∇f[t-1][q])
		else
			∇f[t][q] = p𝑑 .* (Aᵃ * ∇f[t-1][q])
		end
	end
	for i in eachindex(∇f[t])
		∇D[i] = sum(∇f[t][i])
		for j in eachindex(∇f[t][i])
			∇f[t][i][j] = (∇f[t][i][j] - f[t][j]*∇D[i])/D
		end
	end
	γ = f[trial.ntimesteps]
	∇γ = ∇f[trial.ntimesteps]
	q = indexθ_ψ[1]
	∇ℓ[q] += expectation_derivative_logp𝑑_wrt_ψ(trial.choice, γ, θnative.ψ[1])
	∇∇ℓ[q,q] += expectation_second_derivative_logp𝑑_wrt_ψ(trial.choice, γ, θnative.ψ[1])
	for r = q:nθ
		∇∇ℓ[q,r] += expectation_derivative_logp𝑑_wrt_ψ(trial.choice, ∇γ[r], θnative.ψ[1])
	end
	for q in eachindex(∇b)
		∇b[q] .= 0
	end
	Aᵃ, ∇Aᵃ, ∇∇Aᵃ = ∇∇transitionmatrices(memory, adaptedclicks, clicks, t)
	for i = 1:nθ_paₜaₜ₋₁
		q = indexθ_paₜaₜ₋₁[i]
		η = (p𝑑'*∇Aᵃ[i]*f[t-1])[1]/D
		∇ℓ[q] += η
		for r = q:nθ
			∇∇ℓ[q,r] += (p𝑑'*∇Aᵃ[i]*∇f[t-1][r])[1]/D - η/D*∇D[r]
			j = index_paₜaₜ₋₁_in_θ[r]
			if j > 0
				∇∇ℓ[q,r] += (p𝑑'*∇∇Aᵃ[i,j]*f[t-1])[1]/D
			end
			j = index_ψ_in_θ[r]
			if j > 0
				∇∇ℓ[q,r] += (∂p𝑑_∂ψ'*∇Aᵃ[i]*f[t-1])[1]/D
			end
		end
	end
	b = nothing # so that updates of b in inside the for loop is accessible outside of the loop
	for t = trial.ntimesteps-1:-1:1
		Aᵃₜ₊₁, ∇Aᵃₜ₊₁, ∇∇Aᵃₜ₊₁ = ∇∇transitionmatrices(memory, adaptedclicks, clicks, t+1)
		Aᵃₜ₊₁ᵀ = transpose(Aᵃₜ₊₁)
		if t == trial.ntimesteps-1
			for q = 1:nθ
				i_aₜ = index_paₜaₜ₋₁_in_θ[q]
				i_ψ = index_ψ_in_θ[q]
				if i_ψ != 0
					∇b[q] = Aᵃₜ₊₁ᵀ*(∂p𝑑_∂ψ./D .-  p𝑑./D^2 .*∇D[q])
				elseif i_aₜ != 0
					∇b[q] = transpose(∇Aᵃₜ₊₁[i_aₜ])*(p𝑑./D) .-  Aᵃₜ₊₁ᵀ*(p𝑑./D^2 .*∇D[q])
				else
					∇b[q] = -Aᵃₜ₊₁ᵀ*(p𝑑./D^2 .*∇D[q])
				end
			end
			b = Aᵃₜ₊₁ᵀ*(p𝑑./D)
		else
			for q in eachindex(∇ℓ)
				i = index_paₜaₜ₋₁_in_θ[q]
				if i > 0
					∇b[q] = (transpose(∇Aᵃₜ₊₁[i])*b) .+ (Aᵃₜ₊₁ᵀ*∇b[q])
				else
					∇b[q] = Aᵃₜ₊₁ᵀ*∇b[q]
				end
			end
			b = Aᵃₜ₊₁ᵀ * b
		end
		if t > 1
			Aᵃ, ∇Aᵃ, ∇∇Aᵃ = ∇∇transitionmatrices(memory, adaptedclicks, clicks, t)
			bᵀ = transpose(b)
			for i = 1:nθ_paₜaₜ₋₁
				q = indexθ_paₜaₜ₋₁[i]
				∇ℓ[q] += (bᵀ*∇Aᵃ[i]*f[t-1])[1]
				for r = q:nθ
					∇∇ℓ[q,r] += (transpose(∇b[r])*∇Aᵃ[i]*f[t-1])[1] + (bᵀ*∇Aᵃ[i]*∇f[t-1][r])[1]
					j = index_paₜaₜ₋₁_in_θ[r]
					if j > 0
						∇∇ℓ[q,r] += (bᵀ*∇∇Aᵃ[i,j]*f[t-1])[1]
					end
				end
			end
		end
	end
	for i = 1:nθ_pa₁
		q = indexθ_pa₁[i]
		∇ℓ[q] += dot(b, ∇pa₁[i])
		for r = q:nθ
			∇∇ℓ[q,r] += dot(∇b[r], ∇pa₁[i])
			j = index_pa₁_in_θ[r]
			if j > 0
				∇∇ℓ[q,r] += dot(b, ∇∇pa₁[i,j])
			end
		end
	end
	return nothing
end

"""
	∇∇transitionmatrices!(memory, adaptedclicks, clicks, t)

Obtain the transition matrices and their first and second partial derivatives for a time step

If the time step has input, the transition matrix and its derivatives are computed in-place

INPUT
-`memory`: a structure containing the memory used for computing the hessian of the log-likelihood of only the choices
-`adaptedclicks`: a structure containing the information on the post-adaptation strengths of the clicks as well as their first and second derivatives
-`clicks`: a structure containing information on the timing of the auditory inputs
-`t`: time step

OUTPUT
-`Aᵃ`: transition matrix. Element Aᵃ[i,j] corresponds to `p(aₜ = ξᵢ ∣ aₜ₋₁ = ξⱼ)`
-`∇Aᵃ`: first deriative of the transition matrix. Element ∇Aᵃ[m][i,j] corresponds to `∂{p(aₜ = ξᵢ ∣ aₜ₋₁ = ξⱼ)}/∂θₘ`
-`∇Aᵃ`: first deriative of the transition matrix. Element ∇∇Aᵃ[m,n][i,j] corresponds to `∂²{p(aₜ = ξᵢ ∣ aₜ₋₁ = ξⱼ)}/∂θₘ∂θₙ`
"""
function ∇∇transitionmatrices!(memory::Memory_for_hessian_choiceLL, adaptedclicks::Adaptedclicks, clicks::Clicks, t::Integer)
	@unpack P, Aᵃsilent, ∇Aᵃsilent, ∇∇Aᵃsilent, Aᵃinput, ∇Aᵃinput, ∇∇Aᵃinput = memory
	if t ∈ clicks.inputtimesteps
		update_for_∇∇transition_probabilities!(P, adaptedclicks, clicks, t)
		clickindex = clicks.inputindex[t][1]
		Aᵃ = Aᵃinput[clickindex]
		∇Aᵃ = ∇Aᵃinput[clickindex]
		∇∇Aᵃ = ∇∇Aᵃinput[clickindex]
		∇∇transitionmatrix!(∇∇Aᵃ, ∇Aᵃ, Aᵃ, P)
	else
		Aᵃ = Aᵃsilent
		∇Aᵃ = ∇Aᵃsilent
		∇∇Aᵃ = ∇∇Aᵃsilent
	end
	return Aᵃ, ∇Aᵃ, ∇∇Aᵃ
end

"""
	∇∇transitionmatrices(memory, adaptedclicks, clicks, t)

Obtain the transition matrices and their first and second partial derivatives for a time step

INPUT
-`memory`: a structure containing the memory used for computing the hessian of the log-likelihood of only the choices
-`adaptedclicks`: a structure containing the information on the post-adaptation strengths of the clicks as well as their first and second derivatives
-`clicks`: a structure containing information on the timing of the auditory inputs
-`t`: time step

OUTPUT
-`Aᵃ`: transition matrix. Element Aᵃ[i,j] corresponds to `p(aₜ = ξᵢ ∣ aₜ₋₁ = ξⱼ)`
-`∇Aᵃ`: first deriative of the transition matrix. Element ∇Aᵃ[m][i,j] corresponds to `∂{p(aₜ = ξᵢ ∣ aₜ₋₁ = ξⱼ)}/∂θₘ`
-`∇Aᵃ`: first deriative of the transition matrix. Element ∇∇Aᵃ[m,n][i,j] corresponds to `∂²{p(aₜ = ξᵢ ∣ aₜ₋₁ = ξⱼ)}/∂θₘ∂θₙ`
"""
function ∇∇transitionmatrices(memory::Memory_for_hessian_choiceLL, adaptedclicks::Adaptedclicks, clicks::Clicks, t::Integer)
	@unpack P, Aᵃsilent, ∇Aᵃsilent, ∇∇Aᵃsilent, Aᵃinput, ∇Aᵃinput, ∇∇Aᵃinput = memory
	if t ∈ clicks.inputtimesteps
		clickindex = clicks.inputindex[t][1]
		Aᵃ = Aᵃinput[clickindex]
		∇Aᵃ = ∇Aᵃinput[clickindex]
		∇∇Aᵃ = ∇∇Aᵃinput[clickindex]
	else
		Aᵃ = Aᵃsilent
		∇Aᵃ = ∇Aᵃsilent
		∇∇Aᵃ = ∇∇Aᵃsilent
	end
	return Aᵃ, ∇Aᵃ, ∇∇Aᵃ
end

"""
    conditional_choice_likelihood!(p, choice, ψ)

In-place computation of the condition likelihood a choice given the accumulator state

MODIFIED ARGUMENT
-`p`: after modidication, element `p[i]` corresponds to `p(choice ∣ a=ξᵢ)`

ARGUMENT
-`choice`: the observed choice, either right (`choice`=true) or left.
-`ψ`: the prior probability of a lapse state
"""
function conditional_choice_likelihood!(p::Vector{<:Real}, choice::Bool, ψ::Real)
	Ξ = length(p)
	zeroindex = cld(Ξ,2)
    p[zeroindex] = 0.5
    if choice
        p[1:zeroindex-1]   .= ψ/2
        p[zeroindex+1:end] .= 1-ψ/2
    else
        p[1:zeroindex-1]   .= 1-ψ/2
        p[zeroindex+1:end] .= ψ/2
    end
    return nothing
end

"""
	differentiate_conditional_choice_likelihood_wrt_ψ!(∂p𝑑_∂ψ, 𝑑)

Derivative of the conditional likelihood of the choice with respect to the lapse rate

ARGUMENT
-`𝑑`: left (false) or right (true) choice of the animal

MODIFIED ARGUMENT
-`∂p𝑑_∂ψ`: derivative of the conditional likelihood of the choice with respect to the lapse rate. Element `∂p𝑑_∂ψ[i,j]` represents:
	∂p{𝑑 ∣ a(T)=ξ(i)}/∂ψ
"""
function differentiate_conditional_choice_likelihood_wrt_ψ!(∂p𝑑_∂ψ::Vector{<:Real}, 𝑑::Bool)
	if 𝑑
		∂p𝑑_ξ⁻_∂ψ = 0.5
		∂p𝑑_ξ⁺_∂ψ = -0.5
	else
		∂p𝑑_ξ⁻_∂ψ = -0.5
		∂p𝑑_ξ⁺_∂ψ = 0.5
	end
	Ξ = length(∂p𝑑_∂ψ)
	zeroindex = cld(Ξ,2)
	for i = 1:zeroindex-1
		∂p𝑑_∂ψ[i] = ∂p𝑑_ξ⁻_∂ψ
	end
	∂p𝑑_∂ψ[zeroindex] = 0.0
	for i = zeroindex+1:Ξ
		∂p𝑑_∂ψ[i] = ∂p𝑑_ξ⁺_∂ψ
	end
end

"""
	Memory_for_hessian_choiceLL(model)

Create a structure for computing the hessian of the log-likelihood of the choices

ARGUMENT
-`model`: structure containing the data, hyperparameters, and parameters of a factorial hidden-Markov drift-diffusion model

OUTPUT
-a structure containing the memory and pre-computed quantities
"""
function Memory_for_hessian_choiceLL(model::Model)
	@unpack options, θnative, θreal, trialsets = model
	@unpack Δt, minpa, Ξ = options
	# B, k, λ, μ₀, ϕ, ψ, σ²ₐ, σ²ᵢ, σ²ₛ, wₕ
	parameternames = [:B, :k, :λ, :μ₀, :ϕ, :ψ, :σ²ₐ, :σ²ᵢ, :σ²ₛ, :wₕ]
	nθ = length(parameternames)
	indexθ_pa₁ = [1,4,8,10]
	indexθ_paₜaₜ₋₁ = [1,2,3,5,7,9]
	indexθ_ψ = [6]
	nθ_pa₁ = length(indexθ_pa₁)
	nθ_paₜaₜ₋₁ = length(indexθ_paₜaₜ₋₁)
	P = Probabilityvector(Δt, minpa, θnative, Ξ)
	update_for_∇∇transition_probabilities!(P)
	∇∇Aᵃsilent = map(i->zeros(Ξ,Ξ), CartesianIndices((nθ_paₜaₜ₋₁,nθ_paₜaₜ₋₁)))
	∇Aᵃsilent = map(i->zeros(Ξ,Ξ), 1:nθ_paₜaₜ₋₁)
	Aᵃsilent = ones(typeof(θnative.B[1]), Ξ, Ξ).*minpa
	one_minus_Ξminpa = 1.0-Ξ*minpa
	Aᵃsilent[1,1] += one_minus_Ξminpa
	Aᵃsilent[Ξ, Ξ] += one_minus_Ξminpa
	∇∇transitionmatrix!(∇∇Aᵃsilent, ∇Aᵃsilent, Aᵃsilent, P)
	maxclicks = maximum_number_of_clicks(model)
	maxtimesteps = maximum_number_of_time_steps(model)
	f = collect(zeros(Ξ) for t=1:maxtimesteps)
	∇f = collect(collect(zeros(Ξ) for q=1:nθ) for t=1:maxtimesteps)
	Aᵃinput=map(1:maxclicks) do t
				A = ones(Ξ,Ξ).*minpa
				A[1,1] += one_minus_Ξminpa
				A[Ξ,Ξ] += one_minus_Ξminpa
				return A
			end
	∇Aᵃinput = collect(collect(zeros(Ξ,Ξ) for q=1:nθ_paₜaₜ₋₁) for t=1:maxclicks)
	∇∇Aᵃinput = map(1:maxclicks) do t
					map(CartesianIndices((nθ_paₜaₜ₋₁,nθ_paₜaₜ₋₁))) do ij
						zeros(Ξ,Ξ)
					end
				end
	∇pa₁ = collect(zeros(Ξ) for q=1:nθ_pa₁)
	∇∇pa₁ = map(CartesianIndices((nθ_pa₁,nθ_pa₁))) do q
				zeros(Ξ)
			end
	Memory_for_hessian_choiceLL(Ξ=Ξ,
								parameternames=parameternames,
								indexθ_pa₁=indexθ_pa₁,
								indexθ_paₜaₜ₋₁=indexθ_paₜaₜ₋₁,
								indexθ_ψ=indexθ_ψ,
								P=P,
								Aᵃsilent=Aᵃsilent,
								∇Aᵃsilent=∇Aᵃsilent,
								∇∇Aᵃsilent=∇∇Aᵃsilent,
								Aᵃinput=Aᵃinput,
								∇Aᵃinput=∇Aᵃinput,
								∇∇Aᵃinput=∇∇Aᵃinput,
								∇pa₁=∇pa₁,
								∇∇pa₁=∇∇pa₁,
								f=f,
								∇f=∇f)
end
