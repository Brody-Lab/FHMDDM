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
julia> datapath = "/mnt/cup/labs/brody/tzluo/analysis_data/analysis_2022_08_08c_test/T176_2018_05_03/data.mat"
julia> model = Model(datapath)
julia> learnparameters!(model)
julia>
```
"""
function learnparameters!(model::Model; initialize::Bool=true, iterations::Integer=500)
	@unpack objective = model.options
	initialize && initializeparameters!(model)
	if objective == "evidence"
		output = maximizeevidence!(model;iterations=iterations)
	elseif objective == "posterior"
		output = maximizeposterior!(model; iterations=iterations)
	elseif objective == "likelihood"
		output = maximizelikelihood!(model, Optim.LBFGS(linesearch = LineSearches.BackTracking()); iterations=iterations)
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
function initializeparameters!(model::Model; show_trace::Bool=true, verbose::Bool=true)
	if !isempty(concatenate_latent_parameters(model)[1])
		fitonlychoices!(model; show_trace=show_trace)
	end
	verbose && println("Initializing GLM parameters")
	stats = @timed initialize_GLM_parameters!(model; show_trace=false)
	verbose && println("Initializing the GLM parameters took ", stats.time, " seconds")
	return nothing
end

"""
    maximizeposterior!(model)

Optimize the parameters of the factorial hidden Markov drift-diffusion model using a first-order optimizer in Optim

MODIFIED ARGUMENT
-`model`

OPTIONAL ARGUMENT
-`extended_trace`: save additional information
-`f_tol`: threshold for determining convergence in the objective value
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
julia> model = Model("/mnt/cup/labs/brody/tzluo/analysis_data/analysis_2022_08_08a_test/T176_2018_05_03/data.mat")
julia> FHMDDM.initializeparameters!(model)
julia> optimizationresults, losses, gradientnorms = maximizeposterior!(model; store_trace = true)
```
"""
function maximizeposterior!(model::Model;
							extended_trace::Bool=false,
							f_tol::AbstractFloat=0.0,
							iterations::Integer=500,
							optimizer::Optim.FirstOrderOptimizer = LBFGS(linesearch = LineSearches.BackTracking()),
							show_every::Integer=10,
							show_trace::Bool=true,
							store_trace::Bool=false,
							x_tol::AbstractFloat=0.0)
	memory = Memoryforgradient(model)
    f(concatenatedθ) = -logposterior!(model, memory, concatenatedθ)
	g!(∇,concatenatedθ) = ∇negativelogposterior!(∇, model, memory, concatenatedθ)
    Optim_options = Optim.Options(extended_trace=extended_trace,
								  f_tol=f_tol,
                                  g_tol=model.options.g_tol,
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
    return optimizationresults, losses, gradientnorms
end

"""
	logposterior(model)

Log of the posterior probability of the parameters given the data
"""
logposterior(model::Model) = logposterior!(model, Memoryforgradient(model), concatenateparameters(model)[1])

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
