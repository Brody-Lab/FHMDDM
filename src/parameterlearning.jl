"""
	learnparameters!(model)

Optimize the parameters of the factorial hidden Markov drift-diffusion model

The value maximized in the optimization is specified in `model.options.objective`

MODIFIED ARGUMENT
-`model`

RETURN
-depending on the `model.options.objective`, output of `maximizeevidence!`, `maximizeposterior!`, or `maximizelikelihood!`

EXAMPLE
```julia-repl
julia> using FHMDDM
julia> model = Model("/mnt/cup/labs/brody/tzluo/analysis_data/analysis_2022_07_20e_test/T249_2020_02_12/data.mat")
julia> learnparameters!(model)
julia> λΔt, pchoice = expectedemissions(model;nsamples=10)
julia> fbz = posterior_first_state(model)
julia> save(model, fbz, λΔt, pchoice)
julia>
julia> using FHMDDM
julia> model = Model("/mnt/cup/labs/brody/tzluo/analysis_data/analysis_2022_07_20j_test/T224_2020_01_03/data.mat")
julia> learnparameters!(model)
julia>
```
"""
function learnparameters!(model::Model; initialize::Bool=true)
	@unpack objective = model.options
	initialize && initializeparameters!(model)
	if objective == "evidence"
		output = maximizeevidence!(model)
	elseif objective == "posterior"
		output = maximizeposterior!(model)
	elseif objective == "likelihood"
		output = maximizelikelihood!(model, Optim.LBFGS(linesearch = LineSearches.BackTracking()))
	else
		error(objective, " is not a recognized objective.")
	end
	return output
end

"""
	initializeparameters!(model)

Initialize the values of a subset of the parameters by maximizing the likelihood of only the choices.

The parameters specifying the transition probability of the coupling variable are not modified. The weights of the GLM are computed by maximizing the expectation of complete-data log-likelihood across accumulator states.

MODIFIED ARGUMENT
-`model`: an instance of the factorial hidden Markov drift-diffusion model

EXAMPLE
```julia-repl
julia> using FHMDDM
julia> datapath = "/mnt/cup/labs/brody/tzluo/analysis_data/analysis_2022_07_15b_test/T176_2018_05_03/data.mat"
julia> model = Model(datapath)
julia> FHMDDM.initializeparameters!(model)
```
"""
function initializeparameters!(model::Model; verbose::Bool=true)
	fitonlychoices!(model)
	if model.options.updateDDtransformation
		model = update_drift_diffusion_transformation(model)
	end
	verbose && println("Initializing GLM parameters")
	stats = @timed initialize_GLM_parameters!(model)
	verbose && println("Initializing the GLM parameters took ", stats.time, " seconds")
	return nothing
end

"""
	initialize_GLM_parameters!(model)

Initialize the GLM parameters

MODIFIED ARGUMENT
-`model`: an instance of the factorial hidden Markov drift-diffusion model
"""
function initialize_GLM_parameters!(model::Model)
	memory = FHMDDM.Memoryforgradient(model)
	choiceposteriors!(memory, model)
	for i in eachindex(model.trialsets)
	    for mpGLM in model.trialsets[i].mpGLMs
	        maximize_expectation_of_loglikelihood!(mpGLM, memory.γ[i])
	    end
	end
	if model.options.gain_state_dependent
		for i in eachindex(model.trialsets)
		    for mpGLM in model.trialsets[i].mpGLMs
		        gmean = mean(mpGLM.θ.𝐠)
				mpGLM.θ.𝐠[1] .= 3.0.*gmean
				mpGLM.θ.𝐠[2] .= -gmean
		    end
		end
	end
	if model.options.tuning_state_dependent
		for i in eachindex(model.trialsets)
			for mpGLM in model.trialsets[i].mpGLMs
				vmean = mean(mpGLM.θ.𝐯)
				mpGLM.θ.𝐯[1] .= 3.0.*vmean
				mpGLM.θ.𝐯[2] .= -vmean
			end
		end
	end
end

"""
	initialize_for_stochastic_transition!(model)

Initialize the parameters of the model such that the state transitions over time

MODIFIED ARGUMENT
-`model`: a structure containing the data, parameters, and the hyperparameters of the model. The prior and transition probabilities of the coupling variable are modified. The drift-diffusion parameters are optimized to maximize the posterior likelihood of the choices. The state-independent filters are optimized by maximizing the likelihood of each neuron's GLM as though it does not dependent on the accumulator. The filters of the accumulator dependent input are optimized by performing a single M-step using the posterior probabilities of the latent variable conditioned on only the choices. The data are split such that a different subset of the time steps in each trial used to optimize the filters in each state. The first subset of the time steps are used to optimize the filters of the accumulator in the first state, and the last subset of time steps are used to optimize the filters in the last state.

EXAMPLE
```julia-repl
julia> using FHMDDM
julia> datapath = "/mnt/cup/labs/brody/tzluo/analysis_data/analysis_2022_05_11_test/T176_2018_05_03_001/data.mat"
julia> model = Model(datapath)
julia> ℓs = FHMDDM.initialize_for_stochastic_transition!(model;EMiterations=100)
```
"""
function initialize_for_stochastic_transition!(model::Model; EMiterations::Integer=1, relativeΔℓ::Real=1e-6)
	@unpack options, θnative, θ₀native, θreal, trialsets = model
	@unpack K = model.options
	θ₀native.πᶜ₁[1] = θnative.πᶜ₁[1] = 0.999
	θ₀native.Aᶜ₁₁[1] = θnative.Aᶜ₁₁[1] = 0.95
	θ₀native.Aᶜ₂₂[1] = θnative.Aᶜ₂₂[1] = 0.999
	native2real!(θreal, options, θnative)
	initializeparameters!(model)
end

"""
	maximizeECDLL!(model, ∑χ, ∑γ)

Optimize the parameters of the coupling variable by maximizing the expectation of the complete-data log-likelihood

MODIFIED ARGUMENT
-`model`: structure containing the data, parameters, and hyperparameters

UNMODIFIED ARGUMENT
-`∑χ`: sum of the joint posterior probability of the coupling variable across consecutive time steps
-`∑γ`: sum of the posterior probability of the coupling variable across consecutive time steps
"""
function maximizeECDLL!(model::Model, ∑χ::Matrix{<:Real}, ∑γ::Vector{<:Real})
	@unpack options, θnative, θreal = model
	θnative.Aᶜ₁₁[1] = max(min(∑χ[1,1]/(∑χ[1,1]+∑χ[2,1]), options.lqu_Aᶜ₁₁[3]), options.lqu_Aᶜ₁₁[1])
	θnative.Aᶜ₂₂[1] = max(min(∑χ[2,2]/(∑χ[2,2]+∑χ[1,2]), options.lqu_Aᶜ₂₂[3]), options.lqu_Aᶜ₂₂[1])
	θnative.πᶜ₁[1] = max(min(∑γ[1]/(∑γ[1]+∑γ[2]),options.lqu_πᶜ₁[3]), options.lqu_πᶜ₁[1])
	native2real!(θreal, options, θnative)
end

"""
    maximizeposterior!(model)

Optimize the parameters of the factorial hidden Markov drift-diffusion model using a first-order optimizer in Optim

MODIFIED ARGUMENT
-`model`

OPTIONAL ARGUMENT
-`extended_trace`: save additional information
-`f_tol`: threshold for determining convergence in the objective value
-`g_tol`: threshold for determining convergence in the gradient
-`iterations`: number of inner iterations that will be run before the optimizer gives up
-`outer_iterations`: number of outer iterations that will be run before the optimizer gives up
-`show_every`: trace output is printed every `show_every`th iteration.
-`show_trace`: should a trace of the optimizer's state be shown?
-`store_trace`: whether to store the information in each optimization. It is useful for visualizing the change in gradient norm and cost function across iterations. To avoid slow down due to memory usage, it is best to set `extended_trace` to false.
-`x_tol`: threshold for determining convergence in the input vector

RETURN
`losses`: value of the loss function (negative of the un-normalized posterior probability of the parameters) across iterations. If `store_trace` were set to false, then these are NaN's
`gradientnorms`: 2-norm of the gradient of  of the loss function across iterations. If `store_trace` were set to false, then these are NaN's
`optimizationresults`: structure containing the results of the optimization

EXAMPLE
```julia-repl
julia> using FHMDDM
julia> model = Model("/mnt/cup/labs/brody/tzluo/analysis_data/analysis_2022_05_21_test/T176_2018_05_03/data.mat")
julia> FHMDDM.initializeparameters!(model)
julia> losses, gradientnorms, optimizationresults = maximizeposterior!(model)
```
"""
function maximizeposterior!(model::Model;
							extended_trace::Bool=false,
							f_tol::AbstractFloat=0.0,
							g_tol::AbstractFloat=1e-8,
							iterations::Integer=1000,
							optimizer::Optim.FirstOrderOptimizer = LBFGS(linesearch = LineSearches.BackTracking()),
							show_every::Integer=10,
							show_trace::Bool=true,
							store_trace::Bool=true,
							x_tol::AbstractFloat=0.0)
	memory = Memoryforgradient(model)
    f(concatenatedθ) = -logposterior!(model, memory, concatenatedθ)
	g!(∇,concatenatedθ) = ∇negativelogposterior!(∇, model, memory, concatenatedθ)
    Optim_options = Optim.Options(extended_trace=extended_trace,
								  f_tol=f_tol,
                                  g_tol=g_tol,
                                  iterations=iterations,
                                  show_every=show_every,
                                  show_trace=show_trace,
								  store_trace=store_trace,
                                  x_tol=x_tol)
	θ₀ = concatenateparameters(model)[1]
	optimizationresults = Optim.optimize(f, g!, θ₀, optimizer, Optim_options)
    θₘₐₚ = Optim.minimizer(optimizationresults)
	sortparameters!(model, θₘₐₚ, memory.indexθ)
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
    return losses, gradientnorms, optimizationresults
end

"""
	logposterior!(model, memory, concatenatedθ)

Log of the posterior probability, minus the terms independent of the parameters

MODIFIED ARGUMENT
-`model`: structure containing the data, parameters, and hyperparameters of a factorial hidden Markov drift-diffusion model
-`memory`: structure for in-place computation of the gradient of the log-likelihood

UNMODIFIED ARGUMENT
-`concatenatedθ`: Values of the parameters being fitted concatenated into a vector

RETURN
-log of the posterior probability of the parameters, minus the parameter-independent terms
"""
function logposterior!(model::Model, memory::Memoryforgradient, concatenatedθ::Vector{<:Real})
	loglikelihood!(model, memory, concatenatedθ) - 0.5dot(concatenatedθ, model.gaussianprior.𝚲, concatenatedθ)
end

"""
	logposterior(concatenatedθ, indexθ, model)

ForwardDiff-compatiable computation of the log of the posterior probability, minus the terms independent of the parameters

ARGUMENT
-`concatenatedθ`: a vector of concatenated parameter values
-`indexθ`: struct indexing of each parameter in the vector of concatenated values
-`model`: an instance of FHM-DDM

RETURN
-log of the posterior probability of the parameters, minus the parameter-independent terms

"""
function logposterior(concatenatedθ::Vector{T}, indexθ::Indexθ, model::Model) where {T<:Real}
	loglikelihood(concatenatedθ, indexθ, model) - 0.5dot(concatenatedθ, model.gaussianprior.𝚲, concatenatedθ)
end

"""
	∇negativelogposterior!(∇, model, memory, concatenatedθ)

Gradient of the negative of the log of the posterior probability, minus the terms independent of the parameters

MODIFIED ARGUMENT
-`∇`: the gradient
-`model`: structure containing the data, parameters, and hyperparameters of a factorial hidden Markov drift-diffusion model
-`memory`: structure for in-place computation of the gradient of the log-likelihood

UNMODIFIED ARGUMENT
-`concatenatedθ`: Values of the parameters being fitted concatenated into a vector
"""
function ∇negativelogposterior!(∇::Vector{<:Real}, model::Model, memory::Memoryforgradient, concatenatedθ::Vector{<:Real})
	∇negativeloglikelihood!(∇, memory, model, concatenatedθ)
	mul!(∇, model.gaussianprior.𝚲, concatenatedθ, 1, 1) # same as `∇ .+= 𝚲*concatenatedθ` but allocates no memory; not much faster though
	return nothing
end

"""
	check_∇negativelogposterior(model)

Compare the hand-computed and automatically-differentiated gradients

ARGUMENT
-`model`: a structure containing the data, parameters, and hyperparameters of a factorial hidden-Markov drift-diffusion model

RETURN
-`absdiffℓ`: absolute difference in the log-posterior evaluted using the algorithm bein automatically differentiated and the hand-coded algorithm
-`absdiff∇`: absolute difference in the gradients

EXAMPLE
```julia-repl
julia> using FHMDDM
julia> datapath = "/mnt/cup/labs/brody/tzluo/analysis_data/analysis_2022_07_18a_test/T176_2018_05_03_scaled/data.mat"
julia> model = Model(datapath)
julia> absdiffℓ, absdiff∇ = FHMDDM.check_∇negativelogposterior(model)
julia> println("")
julia> println(datapath)
julia> println("   max(|Δloss|): ", absdiffℓ)
julia> println("   max(|Δgradient|): ", maximum(absdiff∇))
julia>
```
"""
function check_∇negativelogposterior(model::Model)
	concatenatedθ, indexθ = FHMDDM.concatenateparameters(model)
	memory = Memoryforgradient(model)
	ℓhand = logposterior!(model, memory, concatenatedθ)
	∇hand = similar(concatenatedθ)
	∇negativelogposterior!(∇hand, model, memory, concatenatedθ)
	f(x) = logposterior(x, indexθ, model)
	ℓauto = f(concatenatedθ)
	∇auto = ForwardDiff.gradient(f, concatenatedθ)
	return abs(ℓauto-ℓhand), abs.(∇auto .+ ∇hand)
end
