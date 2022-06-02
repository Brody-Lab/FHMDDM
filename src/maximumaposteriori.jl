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

EXAMPLE
```julia-repl
julia> using FHMDDM
julia> model = Model("/mnt/cup/labs/brody/tzluo/analysis_data/analysis_2022_05_21_test/T176_2018_05_03/data.mat")
julia> FHMDDM.initializeparameters!(model)
julia> losses, gradientnorms = maximizeposterior!(model)
```
"""
function maximizeposterior!(model::Model;
							extended_trace::Bool=false,
							f_tol::AbstractFloat=0.0,
							g_tol::AbstractFloat=1e-8,
							iterations::Integer=1000,
							show_every::Integer=10,
							show_trace::Bool=true,
							store_trace::Bool=true,
							x_tol::AbstractFloat=0.0)
	optimizer = LBFGS(linesearch = LineSearches.BackTracking())
	memory = Memoryforgradient(model)
	𝐀 = model.precisionmatrix
    f(concatenatedθ) = -loglikelihood!(model, memory, concatenatedθ) + 0.5(concatenatedθ' * 𝐀 * concatenatedθ)
    function g!(∇, concatenatedθ)
		∇negativeloglikelihood!(∇, memory, model, concatenatedθ)
		∇ .+= 𝐀*concatenatedθ
		return nothing
	end
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
    return losses, gradientnorms
end
