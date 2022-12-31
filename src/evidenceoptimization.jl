"""
	maximizeevidence!(model)

Learn both the parameters and hyperparameters by maximizing the evidence

The optimization procedure alternately fixes the hyperparameters and learn the parameters by maximizing the posterior and fixes the parameters and learn the hyperparameters by maximing the evidence.

MODIFIED ARGUMENT
-`model`: structure containing the parameters, hyperparameters, and data. The parameters and the precision matrix are updated maximize the evidence of the data

OPTIONAL ARGUMET
-`iterations`: maximum number of iterations for optimizing the posterior probability of the parameters
-`outer_iterations`: maximum number of iterations for alternating between maximizing the posterior probability and maximizing the evidence
-`verbose`: whether to display messages

EXAMPLE
```julia-repl
julia> using FHMDDM
julia> model = Model("/mnt/cup/labs/brody/tzluo/analysis_data/analysis_2022_08_08a_test/T176_2018_05_03/data.mat")
julia> initializeparameters!(model)
julia> maximizeevidence!(model)
julia>
```
"""
function maximizeevidence!(model::Model;
						iterations::Int=500,
						max_consecutive_failures::Int=2,
						outer_iterations::Int=5,
						verbose::Bool=true,
						x_reltol::Real=1e-1)
	@unpack index𝚽, 𝛂min, 𝛂max = model.gaussianprior
	memory = FHMDDM.Memoryforgradient(model)
	best𝛉 = concatenateparameters(model)
	index𝛉 = indexparameters(model)
	best𝐸 = -Inf
	best𝛂 = copy(model.gaussianprior.𝛂)
	n_consecutive_failures = 0
	posteriorconverged = false
	for i = 1:outer_iterations
	    results = maximizeposterior!(model;iterations=iterations)[1]
		if !Optim.converged(results)
			if Optim.iteration_limit_reached(results)
				for i = 1:length(model.gaussianprior.𝛂)
					model.gaussianprior.𝛂[i] = min(𝛂max[i], 2model.gaussianprior.𝛂[i])
				end
				if verbose
					println("Outer iteration: ", i, ": because the MAP optimization did not converge after reaching the maximum number of iterations, the values of the precisions are doubled")
					println("Outer iteration ", i, ": new 𝛂 → ", model.gaussianprior.𝛂)
				end
			else
				verbose && println("Outer iteration: ", i, ": because of a line search failure, Gaussian noise is added to the parameter values")
				𝛉 = concatenateparameters(model)
				𝛉 .+= randn(length(𝛉))
				sortparameters!(model, 𝛉, index𝛉)
			end
		else
			verbose && println("Outer iteration: ", i, ": the MAP values of the parameters converged")
			𝛉₀ = concatenateparameters(model) # exact posterior mode
			stats = @timed ∇∇loglikelihood(model)[3][index𝚽, index𝚽]
			𝐇 = stats.value
			verbose && println("Outer iteration: ", i, ": computing the Hessian of the log-likelihood took ", stats.time, " seconds")
			𝐸 = logevidence!(memory, model, 𝐇, 𝛉₀)
			if 𝐸 > best𝐸
				if verbose
					if posteriorconverged
						println("Outer iteration: ", i, ": the log-evidence (best: ", best𝐸, "; new:", 𝐸, ") is improved by the new values of the precisions found in the previous outer iteration")
					else
						println("Outer iteration: ", i, ": initial value of log-evidence: ", 𝐸, " is set as the best log-evidence")
					end
				end
				best𝐸 = 𝐸
				best𝛂 .= model.gaussianprior.𝛂
				best𝛉 .= 𝛉₀
				n_consecutive_failures = 0
			else
				n_consecutive_failures += 1
				verbose && println("Outer iteration: ", i, ": because the log-evidence (best: ", best𝐸, "; new:", 𝐸, ") was not improved by the new precisions, subsequent learning of the precisions will be begin at the midpoint between the current values of the precisions and the values that gave the best evidence so far.")
				for j in eachindex(model.gaussianprior.𝛂)
					model.gaussianprior.𝛂[j] = (model.gaussianprior.𝛂[j] + best𝛂[j])/2
				end
			end
			posteriorconverged = true
			if n_consecutive_failures == max_consecutive_failures
				verbose && println("Outer iteration: ", i, ": optimization halted early due to ", max_consecutive_failures, " consecutive failures in improving evidence")
				break
			end
			normΔ = maximizeevidence!(memory, model, 𝐇, 𝛉₀)
			if verbose
				println("Outer iteration ", i, ": new 𝛂 → ", model.gaussianprior.𝛂)
			end
			if normΔ < x_reltol
				verbose && println("Outer iteration: ", i, ": optimization halted after relative difference in the norm of the hyperparameters (in real space) decreased below ", x_reltol)
				break
			else
				sortparameters!(model, 𝛉₀, index𝛉)
			end
		end
		if (i==outer_iterations) && verbose
			println("Optimization halted after reaching the last of ", outer_iterations, " allowed outer iterations.")
		end
	end
	println("Best log-evidence: ", best𝐸)
	println("Best L2 penalty coefficients: ", best𝛂)
	println("Best parameters: ", best𝛉)
	model.gaussianprior.𝛂 .= best𝛂
	precisionmatrix!(model.gaussianprior)
	sortparameters!(model, best𝛉, index𝛉)
	real2native!(model.θnative, model.options, model.θreal)
	return nothing
end

"""
	logevidence!(memory, model, 𝐇, 𝛉)

Log of the approximate marginal likelihood

MODIFIED ARGUMENT
-`memory`: a structure containing memory for in-place computation
-`model`: structure containing the parameters, hyperparameters, and data

UNMODIFIED ARGUMENT
-`𝛉`: posterior mode
"""
function logevidence!(memory::Memoryforgradient, model::Model, 𝐇::Matrix{<:Real}, 𝛉::Vector{<:Real})
	loglikelihood!(model, memory, 𝛉)
	𝐰 = 𝛉[model.gaussianprior.index𝚽]
	logevidence(𝐇, memory.ℓ[1], model.gaussianprior.𝚽, 𝐰)
end

"""
	logevidence(𝐇, ℓ, 𝚽, 𝐰)

Evaluate the log-evidence

ARGUMENT
-`𝚽`: subset of the precision matrix corresponding to only the parametes with finite covariance
-`𝐇`: subset of the Hessian of the log-likelihood evaluated at the MAP values of the parameters, containing only the elements with finite variance
-`ℓ`: log-likelihood evaluated at the approximate posterior mode 𝐰
-`𝐰`: subset of parameters corresponding to dimensions with finite covariance

RETURN
-log evidence
"""
function logevidence(𝐇::Matrix{<:Real}, ℓ::Real, 𝚽::AbstractMatrix{<:Real}, 𝐰::Vector{<:Real})
	𝐌 = I - (𝚽 \ 𝐇)
	logdet𝐌, signdet𝐌 = logabsdet(𝐌)
	if signdet𝐌 < 0
		println("negative determinant")
		-Inf
	else
		ℓ - 0.5dot(𝐰, 𝚽, 𝐰) - 0.5logdet𝐌
	end
end

"""
	maximizeevidence!(memory, model, 𝐇, 𝛉₀)

Learn hyperparameters by fixing the parameters of the model and maximizing the evidence

MODIFIED ARGUMENT
-`memory`: structure containing variables to be modified during computations
-`model`: structure containing the parameters, hyperparameters, and data. The parameter values are modified, but the hyperparameters are not modified

UNMODIFIED ARGUMENT
-`𝐇`: Hessian of the log-likelihood evaluated at the MAP solution `𝛉₀`, containing only the parameters associated with hyperparameters that are being optimized
-`𝛉₀`: exact MAP solution

RETURN
-Euclidean of the normalized difference in the log of the hyperparameters being optimized
"""
function maximizeevidence!(memory::Memoryforgradient,
						model::Model,
						𝐇::Matrix{<:Real},
						𝛉₀::Vector{<:Real};
						optimizationoptions::Optim.Options=Optim.Options(iterations=15, show_trace=false, show_every=1),
						optimizer::Optim.FirstOrderOptimizer=LBFGS(linesearch=LineSearches.BackTracking()))
	@unpack gaussianprior = model
	@unpack 𝛂min, 𝛂max, index𝚽, 𝚽 = gaussianprior
	𝐰₀ = 𝛉₀[index𝚽]
	𝐁₀𝐰₀ = (𝚽-𝐇)*𝐰₀
	𝐱₀ = native2real(gaussianprior)
	𝛉 = concatenateparameters(model)
	∇nℓ = similar(𝛉)
	f(𝐱) = -logevidence!(memory, model, 𝛉, 𝐁₀𝐰₀, 𝐇, 𝐱)
	g!(∇n𝐸, 𝐱) = ∇negativelogevidence!(memory, model, ∇n𝐸, ∇nℓ, 𝛉, 𝐁₀𝐰₀, 𝐇, 𝐱)
	optimizationresults = Optim.optimize(f, g!, 𝐱₀, optimizer, optimizationoptions)
	𝐱̂ = Optim.minimizer(optimizationresults)
	normΔ = 0.0
	for i in eachindex(𝐱̂)
		normΔ += (𝐱̂[i]/𝐱₀[i] - 1.0)^2
	end
	real2native!(gaussianprior, 𝐱̂)
	precisionmatrix!(gaussianprior)
	return √normΔ
end

"""
	logevidence!(memory, model, 𝛉, 𝐁₀𝐰₀, 𝐇, 𝐱)

Log of the marginal likelihood

MODIFIED ARGUMENT
-`memory`: a structure containing memory for in-place computation
-`model`: structure containing the parameters, hyperparameters, and data
-`𝛉`: preallocated memory for the parameters of the model

UNMODIFIED ARGUMENT
-`𝐁₀𝐰₀`: Hessian of the log-posterior evalued at the MAP values of the parameters multiplied by the MAP value of the parameters, containing only the parameters associated with the hyperparameters being optimized
-`𝐇`: Hessian of the log-likelihood evaluated at the MAP values of the parameters, containing only the parameters associated with the hyperparameters being optimized
-`𝐱`: concatenated values of the L2 penalties coefficients in real space

RETURN
-log of the evidence
"""
function logevidence!(memory::Memoryforgradient,
					model::Model,
					𝛉::Vector{<:Real},
					𝐁₀𝐰₀::Vector{<:Real},
					𝐇::Matrix{<:Real},
					𝐱::Vector{<:Real})
	FHMDDM.real2native!(model.gaussianprior, 𝐱)
	FHMDDM.precisionmatrix!(model.gaussianprior)
	@unpack index𝚽, 𝚽 = model.gaussianprior
    𝐰 = (𝚽-𝐇) \ 𝐁₀𝐰₀ # LAPACK.sysv! uses less memory but is slower
	𝛉[index𝚽] .= 𝐰
	FHMDDM.loglikelihood!(model, memory, 𝛉)
	FHMDDM.logevidence(𝐇, memory.ℓ[1], 𝚽, 𝐰)
end

"""
	∇negativelogevidence!(∇n𝐸, memory, model, 𝐇, 𝐁₀𝛉₀, 𝐱)

Gradient of the negative log of the marginal likelihood with respect to the coefficients in real space

MODIFIED ARGUMENT
-`memory`: a structure containing memory for in-place computation
-`model`: structure containing the parameters, hyperparameters, and data
-`∇n𝐸`: memory for in-place computation of the gradient of the negative of the log-evidence
-`∇nℓ`: memory for in-place computation of the gradient of the negative of the log-evidence
-`𝛉`: memory for in-place computation of the approximate posterior mode as a function of the hyperparameters

UNMODIFIED ARGUMENT
-`𝐁₀𝐰₀`: Hessian of the log-posterior evalued at the MAP values of the parameters multiplied by the MAP value of the parameters, containing only the parameters associated with the hyperparameters being optimized
-`𝐇`: Hessian of the log-likelihood evaluated at the MAP values of the parameters, containing only the parameters associated with the hyperparameters being optimized
-`𝐱`: concatenated values of the L2 penalties coefficients in real space
"""
function ∇negativelogevidence!(memory::Memoryforgradient,
								model::Model,
								∇n𝐸::Vector{<:Real},
								∇nℓ::Vector{<:Real},
								𝛉::Vector{<:Real},
								𝐁₀𝐰₀::Vector{<:Real},
								𝐇::Matrix{<:Real},
								𝐱::Vector{<:Real})
	real2native!(model.gaussianprior, 𝐱)
	@unpack 𝐀, 𝛂, 𝛂max, 𝛂min, index𝚽, 𝚽, index𝐀_in_index𝚽 = model.gaussianprior
	𝐁 = 𝚽-𝐇
	C = factorize(𝐁)
	𝐰 = C \ 𝐁₀𝐰₀
	𝛉[index𝚽] .= 𝐰
	∇negativeloglikelihood!(∇nℓ, memory, model, 𝛉)
	𝐦 = C \ (𝚽*𝐰 + ∇nℓ[index𝚽])
	𝛀 = (C \ (𝚽 \ 𝐇)')'
	@inbounds for j = 1:length(𝛂)
		𝐰ⱼ = 𝐰[index𝐀_in_index𝚽[j]]
		𝐦ⱼ = 𝐦[index𝐀_in_index𝚽[j]]
		𝛀ⱼ = 𝛀[index𝐀_in_index𝚽[j],index𝐀_in_index𝚽[j]]
		𝐀ⱼ = 𝐀[j]
		𝐰ⱼᵀ𝐀ⱼ = transpose(𝐰ⱼ)*𝐀ⱼ
		∇n𝐸[j] = 0.5*(𝐰ⱼᵀ𝐀ⱼ*𝐰ⱼ + tr(𝐀ⱼ*𝛀ⱼ)) - 𝐰ⱼᵀ𝐀ⱼ*𝐦ⱼ
		∇n𝐸[j] *= differentiate_native_wrt_real(𝐱[j], 𝛂min[j], 𝛂max[j])
	end
	return nothing
end

"""
	logevidence(𝛂, 𝐁₀𝛉ₘₐₚ, 𝐇, indexθ, model)

ForwardDiff-computation evaluation of the log-evidence

ARGUMENT
-`𝛂`: precisions being learned
-`𝐁₀𝛉ₘₐₚ`: Hessian of the log-posterior evalued at the MAP values of the parameters multiplied by the MAP value of the parameters
-`𝐇`: Hessian of the log-likelihood evaluated at the MAP values of the parameters
-`indexθ`: index of the parameters
-`model`: structure containing the parameters, hyperparameters, and data

RETURN
-log of the marginal likelihood
"""
function logevidence(𝐁₀𝐰₀::Vector{<:Real},
					 𝐇::Matrix{<:Real},
					 model::Model,
					 𝐱::Vector{type},) where{type<:Real}
	gaussianprior = real2native(model.gaussianprior, 𝐱)
	@unpack index𝚽, 𝚽 = gaussianprior
	𝐁 = 𝚽-𝐇
    𝐰 = 𝐁 \ 𝐁₀𝐰₀
	𝛉 = concatenateparameters(model)
	index𝛉 = indexparameters(model)
	𝛉 = 𝛉 .- zero(type)
	𝛉[index𝚽] .= 𝐰
	ℓ = loglikelihood(𝛉, index𝛉, model)
	logevidence(𝐇, ℓ, 𝚽, 𝐰)
end
