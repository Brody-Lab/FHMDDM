"""
    maximizelikelihood!(model)

Learn the parameters of the factorial hidden Markov drift-diffusion model by maximizing the likelihood of the data

MODIFIED ARGUMENT
- a structure containing information for a factorial hidden Markov drift-diffusion model

UNMODIFIED ARGUMENT
-`λ`: L2 regularization weight. This is equivalent to a Gaussian prior with zero mean and a covariance matrix equal to `λI`

OPTIONAL ARGUMENT
-`algorithm`: an algorithm implemented by Optim.jl. The limited memory pseudo-Newton algorithm `LFBGS()` does pretty well, and when using L-BFGS the `HagerZhang()` line search seems to do better than `BackTracking()`
-`extended_trace`: save additional information
-`f_tol`: threshold for determining convergence in the objective value
-`g_tol`: threshold for determining convergence in the gradient
-`iterations`: number of inner iterations that will be run before the algorithm gives up
-`outer_iterations`: number of outer iterations that will be run before the algorithm gives up
-`show_every`: trace output is printed every `show_every`th iteration.
-`show_trace`: should a trace of the optimization algorithm's state be shown?
-`store_trace`: whether to store the information in each optimization. It is useful for visualizing the change in gradient norm and cost function across iterations. To avoid slow down due to memory usage, it is best to set `extended_trace` to false.
-`x_tol`: threshold for determining convergence in the input vector

RETURN
`losess`: value of the loss function (negative of the model's log-likelihood) across iterations. If `store_trace` were set to false, then these are NaN's
`gradientnorms`: 2-norm of the gradient of  of the loss function across iterations. If `store_trace` were set to false, then these are NaN's
"""
function maximizeposterior!(model::Model,
							λ::AbstractFloat;
							 algorithm=LBFGS(linesearch = LineSearches.BackTracking()),
			                 extended_trace::Bool=false,
			                 f_tol::AbstractFloat=1e-9,
			                 g_tol::AbstractFloat=1e-8,
			                 iterations::Integer=1000,
			                 show_every::Integer=10,
			                 show_trace::Bool=true,
							 store_trace::Bool=true,
			                 x_tol::AbstractFloat=1e-5)
	shared = Shared(model)
	@unpack K, Ξ = model.options
	γ =	map(model.trialsets) do trialset
			map(CartesianIndices((Ξ,K))) do index
				zeros(trialset.ntimesteps)
			end
		end
    f(concatenatedθ) = -loglikelihood!(model, shared, concatenatedθ) + λ*(concatenatedθ ⋅ concatenatedθ)
    function g!(∇, concatenatedθ)
		∇negativeloglikelihood!(∇, γ, model, shared, concatenatedθ)
		∇ .+= 2.0.*λ.*concatenatedθ
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
	θ₀ = copy(shared.concatenatedθ)
	optimizationresults = Optim.optimize(f, g!, θ₀, algorithm, Optim_options)
    MAPθ = Optim.minimizer(optimizationresults)
	sortparameters!(model, MAPθ, shared.indexθ)
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
