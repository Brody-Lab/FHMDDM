"""
    maximizeposterior!(model, λ, optimizer)

Optimize the parameters of the factorial hidden Markov drift-diffusion model using a first-order optimizer in Optim

MODIFIED ARGUMENT
- a structure containing information for a factorial hidden Markov drift-diffusion model

UNMODIFIED ARGUMENT
-`λ`: L2 regularization weight. This is equivalent to a Gaussian prior with zero mean and a precision matrix equal to `λI`
-`optimizer`: an optimizer implemented by Optim.jl. The limited memory quasi-Newton algorithm `LBFGS()` does pretty well, and when using L-BFGS the `HagerZhang()` line search seems to do better than `BackTracking()`

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
`losses`: value of the loss function (negative of the model's log-likelihood) across iterations. If `store_trace` were set to false, then these are NaN's
`gradientnorms`: 2-norm of the gradient of  of the loss function across iterations. If `store_trace` were set to false, then these are NaN's

EXAMPLE
```julia-repl
julia> using FHMDDM, LineSearches, Optim
julia> datapath = "/mnt/cup/labs/brody/tzluo/analysis_data/analysis_2022_04_27_test/data.mat"
julia> model = Model(datapath)
julia> losses, gradientnorms = maximizeposterior!(model, 0.2, LBFGS(linesearch = LineSearches.BackTracking()))
"""
function maximizeposterior!(model::Model,
							λ::AbstractFloat,
							optimizer::Optim.FirstOrderOptimizer;
							extended_trace::Bool=false,
							f_tol::AbstractFloat=0.0,
							g_tol::AbstractFloat=1e-8,
							iterations::Integer=1000,
							show_every::Integer=10,
							show_trace::Bool=true,
							store_trace::Bool=true,
							x_tol::AbstractFloat=0.0)
	𝛌 = L2regularizer(λ, model)
	maximizeposterior!(model, 𝛌, optimizer; extended_trace=extended_trace, f_tol=f_tol,g_tol=g_tol, iterations=iterations, show_every=show_every, show_trace=show_trace, store_trace=store_trace, x_tol=x_tol)
end

"""
    maximizeposterior!(model, 𝛌, optimizer)

Optimize the parameters of the factorial hidden Markov drift-diffusion model using a first-order optimizer in Optim

MODIFIED ARGUMENT
-`model`

UNMODIFIED ARGUMENT
-`𝛌`: a vector L2 regularization weight. This is equivalent to a Gaussian prior with zero mean and a precision matrix equal to `Diag(𝛌)`
-`optimizer`: an optimizer implemented by Optim.jl. The limited memory quasi-Newton algorithm `LBFGS()` does pretty well, and when using L-BFGS the `HagerZhang()` line search seems to do better than `BackTracking()`

OPTIONAL ARGUMENT
-see above

RETURN
-see above
"""
function maximizeposterior!(model::Model,
							𝛌::Vector{<:AbstractFloat},
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
    f(concatenatedθ) = -loglikelihood!(model, memory, concatenatedθ) + ((𝛌.*concatenatedθ) ⋅ concatenatedθ)
    function g!(∇, concatenatedθ)
		∇negativeloglikelihood!(∇, memory, model, concatenatedθ)
		∇ .+= 2.0.*𝛌.*concatenatedθ
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
	L2regularizer(λ, model)

Create a vector of L2 regularization weights

The parameters of the coupling variables are not regularized, and the baseline firing rate of the GLM in each state is not regularized

ARGUMENT
-`λ`: constant used to regularize all parameters that are regularized
-`model`: struct containing the parameters, data, and hyperparameters

OUTPUT
-a vector of L2 regularization weights

```julia-repl
julia> using FHMDDM
julia> datapath = "/mnt/cup/labs/brody/tzluo/analysis_data/analysis_2022_04_27_test/data.mat"
julia> model = Model(datapath; randomize=true)
julia> 𝛌 = FHMDDM.L2regularizer(0.2, model)
```
"""
function L2regularizer(λ::Real, model::Model)
	θ, index = concatenateparameters(model)
	𝛌 = zeros(length(θ))
	for field in (:B, :k, :λ, :μ₀, :ϕ, :ψ, :σ²ₐ, :σ²ᵢ, :σ²ₛ, :wₕ)
		i = getfield(index.latentθ, field)[1]
		if i != 0
			𝛌[i] = λ
		end
	end
	for glmθ in index.glmθ
		for glmθ in glmθ
			for h in glmθ.𝐡
				𝛌[h] = λ
			end
			for 𝐮ₖ in glmθ.𝐮
				for u in 𝐮ₖ
					𝛌[u] = λ
				end
			end
			for 𝐯ₖ in glmθ.𝐯
				for u in 𝐯ₖ
					𝛌[u] = λ
				end
			end
		end
	end
	𝛌
end
