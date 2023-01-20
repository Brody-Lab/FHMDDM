"""
	maximizeevidence!(model)

Learn both the parameters and hyperparameters by maximizing the evidence

This optimization procedure alternately fixes the hyperparameters and learn the parameters by maximizing the posterior, and fixes the parameters and learn the hyperparameters by maximizing the evidence.

MODIFIED ARGUMENT
-`model`: structure containing the parameters, hyperparameters, and data. The parameters and hyperparameters are updated.

OPTIONAL ARGUMET
-`iterations`: maximum number of iterations for alternating between maximizing the posterior probability and maximizing the evidence
-`MAP_iterations`: maximum number of iterations for optimizing the posterior probability of the parameters
-`store_trace`: should a trace of the optimization algorithm's state be saved at each iteration?
-`verbose`: whether to display messages
-`x_reltol`: the relative difference in the norm of the L2 penalty coefficients, in real space, below which the optimization procedure aborts
"""
function maximizeevidence!(model::Model;
						iterations::Int=2,
						MAP_iterations::Int=500,
						store_trace::Bool=true,
						verbose::Bool=true,
						x_reltol::Real=1e-1)
	@unpack index𝚽, 𝛂min, 𝛂max = model.gaussianprior
	memory = Memoryforgradient(model)
	index𝛉 = indexparameters(model)
	best𝐸 = -Inf
	best𝛉 = concatenateparameters(model)
	best𝛂 = copy(model.gaussianprior.𝛂)
	for i = 1:iterations
		verbose && printseparator()
		verbose && println("Evidence optimization iteration: ", i, ": maximizing the log-posterior.")
	    Optim_results = maximizeposterior!(model;iterations=MAP_iterations)[1]
		MAP_values_converged = Optim.converged(Optim_results)
		𝛉₀ = concatenateparameters(model)
		stats = @timed ∇∇loglikelihood(model)
		𝐇 = stats.value[3]
		verbose && println("Evidence optimization iteration: ", i, ": computing the Hessian of the log-likelihood took ", stats.time, " seconds")
		if store_trace
			savetrace(MAP_values_converged, 𝐇, i, model)
			verbose && println("Evidence optimization iteration: ", i, ": saved trace")
		end
		if MAP_values_converged
			verbose && println("Evidence optimization iteration: ", i, ": the MAP values of the parameters converged")
		else
			verbose && println("Evidence optimization iteration: ", i, ": the MAP values of the parameters did not converge, and therefore the optimization procedure is aborting.")
			if i == 1
				best𝛉 = concatenateparameters(model)
			end
			break
		end
		𝐇finite = 𝐇[index𝚽, index𝚽]
		𝐸 = logevidence!(memory, model, 𝐇finite, 𝛉₀)
		if 𝐸 > best𝐸
			verbose && println("Evidence optimization iteration: ", i, ": the current log-evidence ( ", 𝐸, ") is greater than its previous value (", best𝐸, ").")
			best𝐸 = 𝐸
			best𝛉 = 𝛉₀
			best𝛂 = copy(model.gaussianprior.𝛂)
		else
			verbose && println("Evidence optimization iteration: ", i, ": the current log-evidence ( ", 𝐸, ") is not greater than its previous value (", best𝐸, "), and therefore the optimization procedure is aborting.")
			break
		end
		if i==iterations
			verbose && println("Evidence optimization iteration ", i, ": the last iteration has been reached, and the optimization procedure is aborting.")
			break
		end
		normΔ = maximizeevidence!(memory, model, 𝐇finite, index𝛉, 𝛉₀)
		verbose && println("Evidence optimization iteration ", i, ": new L2 penalty coefficients (𝛂) → ", model.gaussianprior.𝛂)
		if normΔ < x_reltol
			verbose && println("Evidence optimization iteration: ", i, ": optimization halted after the relative difference in the norm of the L2 penalty coefficients (in real space) decreased below ", x_reltol)
			break
		end
	end
	𝛉₀ = concatenateparameters(model)
	if 𝛉₀ != best𝛉
		sortparameters!(model, best𝛉, index𝛉)
		real2native!(model.θnative, model.options, model.θreal)
	end
	if model.gaussianprior.𝛂 != best𝛂
		model.gaussianprior.𝛂 .= best𝛂
		precisionmatrix!(model.gaussianprior)
	end
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
	maximizeevidence!(memory, model, 𝐇, index𝛉, 𝛉₀)

Learn hyperparameters by fixing the parameters of the model and maximizing the evidence

MODIFIED ARGUMENT
-`memory`: structure containing variables to be modified during computations
-`model`: structure containing the parameters, hyperparameters, and data. Only the hyperparameters are modified

UNMODIFIED ARGUMENT
-`index𝛉`: composite containing the indices of the model parameters if they were concatenated into a vector
-`𝐇`: Hessian of the log-likelihood evaluated at the MAP solution `𝛉₀`, containing only the parameters associated with hyperparameters that are being optimized
-`𝛉₀`: exact MAP solution

RETURN
-Euclidean of the normalized difference in the log of the hyperparameters being optimized
"""
function maximizeevidence!(memory::Memoryforgradient,
						model::Model,
						𝐇::Matrix{<:Real},
						index𝛉::Indexθ,
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
	sortparameters!(model, 𝛉₀, index𝛉) #restore paramter values
	real2native!(model.θnative, model.options, model.θreal)
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

"""
	savetrace(MAP_values_converged, 𝐇, iteration, model)

Save the hessian of the log-likelihood with the
"""
function savetrace(MAP_values_converged::Bool,
					𝐇::Matrix{<:AbstractFloat},
					iteration::Integer,
					model::Model;
					folderpath::String=dirname(model.options.datapath))
	modelsummary = dictionary(ModelSummary(model))
	dict = Dict((key=>modelsummary[key] for key in keys(modelsummary))...,
				"MAP_values_converged"=>MAP_values_converged,
				"hessian_loglikelihood"=>𝐇,
				"hessian_logposterior"=>𝐇-modelsummary["precisionmatrix"])
	filename = "evidence_optimization_iteration_"*string(iteration)*".mat"
	filepath = joinpath(folderpath, filename)
	matwrite(filepath, dict)
end
