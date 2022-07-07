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
julia> model = Model("/mnt/cup/labs/brody/tzluo/analysis_data/analysis_2022_06_10a_test/T176_2018_05_03_b3K2K1/data.mat")
julia> initializeparameters!(model)
julia> maximizeevidence!(model)
```
"""
function maximizeevidence!(model::Model;
						iterations::Int = 500,
						max_consecutive_failures::Int=2,
						outer_iterations::Int=10,
						verbose::Bool=true,
						g_tol::Real=1e-3,
						x_reltol::Real=1e-1)
	@unpack index𝚽 = model.gaussianprior
	memory = Memoryforgradient(model)
	best𝛉, index𝛉 = concatenateparameters(model)
	best𝐸 = -Inf
	best𝛂 = copy(model.gaussianprior.𝛂)
 	best𝐬 = copy(model.gaussianprior.𝐬)
	n_consecutive_failures = 0
	posteriorconverged = false
	for i = 1:outer_iterations
	    results = maximizeposterior!(model; iterations=iterations, g_tol=g_tol)[3]
		if !Optim.converged(results)
			if Optim.iteration_limit_reached(results)
				new_α = min(100.0, 2geomean(model.gaussianprior.𝛂))
				model.gaussianprior.𝛂 .= new_α
				new_s = min(100.0, 2geomean(model.gaussianprior.𝐬))
				model.gaussianprior.𝐬 .= new_s
				verbose && println("Outer iteration: ", i, ": because the maximum number of iterations was reached, the values of the precisions are set to be twice the geometric mean of the hyperparameters. New (𝛂, 𝐬) → (", new_α, ", ", new_s, ")")
			else
				verbose && println("Outer iteration: ", i, ": because of a line search failure, Gaussian noise is added to the parameter values")
				𝛉 = concatenateparameters(model)[1]
				𝛉 .+= randn(length(𝛉))
				sortparameters!(model, 𝛉, index𝛉)
			end
		else
			verbose && println("Outer iteration: ", i, ": the MAP values of the parameters converged")
			𝛉₀ = concatenateparameters(model)[1] # exact posterior mode
			stats = @timed ∇∇loglikelihood(model)[index𝚽, index𝚽]
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
			 	best𝐬 .= model.gaussianprior.𝐬
				best𝛉 .= 𝛉₀
				n_consecutive_failures = 0
			else
				n_consecutive_failures += 1
				verbose && println("Outer iteration: ", i, ": because the log-evidence (best: ", best𝐸, "; new:", 𝐸, ") was not improved by the new precisions, subsequent learning of the precisions will be begin at the midpoint between the current values of the precisions and the values that gave the best evidence so far.")
				for j in eachindex(model.gaussianprior.𝛂)
					model.gaussianprior.𝛂[j] = (model.gaussianprior.𝛂[j] + best𝛂[j])/2
				end
				for j in eachindex(model.gaussianprior.𝐬)
					model.gaussianprior.𝐬[j] = (model.gaussianprior.𝐬[j] + best𝐬[j])/2
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
				println("Outer iteration ", i, ": new 𝐬 → ", model.gaussianprior.𝐬)
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
	println("Best shrinkage coefficients: ", best𝛂)
	println("Best smoothing coefficients: ", best𝐬)
	println("Best parameters: ", best𝛉)
	precisionmatrix!(model.gaussianprior, best𝛂, best𝐬)
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
-`𝐇`: Hessian of the log-likelihood evaluated at the MAP solution `𝛉₀`, containing only the parameters associated with hyperparameters that are being optimized
-`𝛉₀`: exact MAP solution

RETURN
-Euclidean of the normalized difference in the log of the hyperparameters being optimized
"""
function maximizeevidence!(memory::Memoryforgradient,
						model::Model,
						𝐇::Matrix{<:Real},
						𝛉₀::Vector{<:Real};
						αrange::Vector{<:Real}=[1e-1, 1e2],
						srange::Vector{<:Real}=[1e-8, 1e2],
						optimizationoptions::Optim.Options=Optim.Options(iterations=15, show_trace=true, show_every=1),
						optimizer::Optim.FirstOrderOptimizer=LBFGS(linesearch=LineSearches.BackTracking()))
	𝐰₀ = 𝛉₀[model.gaussianprior.index𝚽]
	𝐁₀𝐰₀ = (model.gaussianprior.𝚽-𝐇)*𝐰₀
	𝛂₀𝐬₀ = vcat(model.gaussianprior.𝛂, model.gaussianprior.𝐬)
	𝐱₀ = 𝛂₀𝐬₀
	N𝛂 = length(model.gaussianprior.𝛂)
	N𝐬 = length(model.gaussianprior.𝐬)
	for i = 1:N𝛂
		𝐱₀[i] = native2real(𝛂₀𝐬₀[i], αrange[1], αrange[2])
	end
	for i = N𝛂+1:N𝛂+N𝐬
		𝐱₀[i] = native2real(𝛂₀𝐬₀[i], srange[1], srange[2])
	end
	𝛉 = concatenateparameters(model)[1]
	∇nℓ = similar(𝛉)
	function f(𝐱)
		𝛂𝐬 = copy(𝐱)
		for i = 1:N𝛂
			𝛂𝐬[i] = real2native(𝐱[i], αrange[1], αrange[2])
		end
		for i = N𝛂+1:N𝛂+N𝐬
			𝛂𝐬[i] = real2native(𝐱[i], srange[1], srange[2])
		end
		-logevidence!(memory, model, 𝛉, 𝛂𝐬, 𝐁₀𝐰₀, 𝐇)
	end
	function g!(∇n𝐸, 𝐱)
		𝛂𝐬 = copy(𝐱)
		for i = 1:N𝛂
			𝛂𝐬[i] = real2native(𝐱[i], αrange[1], αrange[2])
		end
		for i = N𝛂+1:N𝛂+N𝐬
			𝛂𝐬[i] = real2native(𝐱[i], srange[1], srange[2])
		end
		∇negativelogevidence!(memory, model, ∇n𝐸, ∇nℓ, 𝛉, 𝛂𝐬, 𝐁₀𝐰₀, 𝐇)
		for i = 1:N𝛂
			∇n𝐸[i] *= differentiate_native_wrt_real(𝐱[i], αrange[1], αrange[2])
		end
		for i = N𝛂+1:N𝛂+N𝐬
			∇n𝐸[i] *= differentiate_native_wrt_real(𝐱[i], srange[1], srange[2])
		end
		return nothing
	end
	optimizationresults = Optim.optimize(f, g!, 𝐱₀, optimizer, optimizationoptions)
	𝐱̂ = Optim.minimizer(optimizationresults)
	normΔ = 0.0
	for i in eachindex(𝐱̂)
		normΔ += (𝐱̂[i]/𝐱₀[i] - 1.0)^2
	end
	𝛂̂𝐬̂ = 𝐱̂
	for i = 1:N𝛂
		𝛂̂𝐬̂[i] = real2native(𝐱̂[i], αrange[1], αrange[2])
	end
	for i = N𝛂+1:N𝛂+N𝐬
		𝛂̂𝐬̂[i] = real2native(𝐱̂[i], srange[1], srange[2])
	end
	precisionmatrix!(model.gaussianprior, 𝛂̂𝐬̂)
	return √normΔ
end

"""
	logevidence!(memory, model, 𝛉, 𝛂𝐬, 𝐁₀𝐰₀, 𝐇)

Log of the marginal likelihood

MODIFIED ARGUMENT
-`memory`: a structure containing memory for in-place computation
-`model`: structure containing the parameters, hyperparameters, and data
-`𝛉`: preallocated memory for the parameters of the model

UNMODIFIED ARGUMENT
-`𝛂𝐬`: concatenated values of the L2 penalties coefficients
-`𝐁₀𝐰₀`: Hessian of the log-posterior evalued at the MAP values of the parameters multiplied by the MAP value of the parameters, containing only the parameters associated with the hyperparameters being optimized
-`𝐇`: Hessian of the log-likelihood evaluated at the MAP values of the parameters, containing only the parameters associated with the hyperparameters being optimized

RETURN
-log of the evidence
"""
function logevidence!(memory::Memoryforgradient,
					model::Model,
					𝛉::Vector{<:Real},
					𝛂𝐬::Vector{<:Real},
					𝐁₀𝐰₀::Vector{<:Real},
					𝐇::Matrix{<:Real})
	precisionmatrix!(model.gaussianprior, 𝛂𝐬)
	@unpack index𝚽, 𝚽 = model.gaussianprior
    𝐰 = (𝚽-𝐇) \ 𝐁₀𝐰₀ # LAPACK.sysv! uses less memory but is slower
	𝛉[index𝚽] .= 𝐰
	loglikelihood!(model, memory, 𝛉)
	logevidence(𝐇, memory.ℓ[1], 𝚽, 𝐰)
end

"""
	∇negativelogevidence!(∇n𝐸, memory, model, 𝛂, 𝐇, 𝐁₀𝛉ₘₐₚ)

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
function ∇negativelogevidence!(memory::Memoryforgradient,
								model::Model,
								∇n𝐸::Vector{<:Real},
								∇nℓ::Vector{<:Real},
								𝛉::Vector{<:Real},
								𝛂𝐬::Vector{<:Real},
								𝐁₀𝐰₀::Vector{<:Real},
								𝐇::Matrix{<:Real})
	precisionmatrix!(model.gaussianprior, 𝛂𝐬)
	@unpack index𝚽, 𝚽, 𝐒, index𝛂_in_index𝚽, index𝐒_in_index𝚽 = model.gaussianprior
	𝐁 = 𝚽-𝐇
	C = factorize(𝐁)
	𝐰 = C \ 𝐁₀𝐰₀
	𝛉[index𝚽] .= 𝐰
	∇negativeloglikelihood!(∇nℓ, memory, model, 𝛉)
	𝐦 = C \ (𝚽*𝐰 + ∇nℓ[index𝚽])
	𝛀 = (C \ (𝚽 \ 𝐇)')'
	N_𝛂 = length(model.gaussianprior.𝛂)
	@inbounds for i=1:N_𝛂
		k = index𝛂_in_index𝚽[i]
		∇n𝐸[i] = 0.5*(𝐰[k]^2 + 𝛀[k,k]) - 𝐰[k]*𝐦[k]
	end
	@inbounds for j = 1:length(model.gaussianprior.𝐬)
		𝐰ⱼ = 𝐰[index𝐒_in_index𝚽[j]]
		𝐦ⱼ = 𝐦[index𝐒_in_index𝚽[j]]
		𝛀ⱼ = 𝛀[index𝐒_in_index𝚽[j],index𝐒_in_index𝚽[j]]
		𝐒ⱼ = 𝐒[j]
		𝐰ⱼᵀ𝐒ⱼ = transpose(𝐰ⱼ)*𝐒ⱼ
		∇n𝐸[N_𝛂+j] = 0.5*(𝐰ⱼᵀ𝐒ⱼ*𝐰ⱼ + tr(𝐒ⱼ*𝛀ⱼ)) - 𝐰ⱼᵀ𝐒ⱼ*𝐦ⱼ
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
julia> model = Model("/mnt/cup/labs/brody/tzluo/analysis_data/analysis_2022_06_20a_test/T176_2018_05_03_b3K2K2/data.mat")
julia> max_abs_norm_diff_∇𝐸, abs_norm_diff_𝐸 = FHMDDM.check_∇logevidence(model; simulate=true)
julia>
julia> model = Model("/mnt/cup/labs/brody/tzluo/analysis_data/analysis_2022_06_20a_test/T176_2018_05_03_b3K2K2/data.mat")
julia> max_abs_norm_diff_∇𝐸, abs_norm_diff_𝐸 = FHMDDM.check_∇logevidence(model; simulate=false)
julia>
julia> using FHMDDM
julia> model = Model("/mnt/cup/labs/brody/tzluo/analysis_data/analysis_2022_06_20a_test/no_smoothing/data.mat")
julia> max_abs_norm_diff_∇𝐸, abs_norm_diff_𝐸 = FHMDDM.check_∇logevidence(model; simulate=true)
julia>
julia> model = Model("/mnt/cup/labs/brody/tzluo/analysis_data/analysis_2022_06_20a_test/no_smoothing/data.mat")
julia> max_abs_norm_diff_∇𝐸, abs_norm_diff_𝐸 = FHMDDM.check_∇logevidence(model; simulate=false)
julia>
```
"""
function check_∇logevidence(model::Model; simulate::Bool=true)
	@unpack index𝛂, index𝐒, index𝚽, 𝚽 = model.gaussianprior
	𝛉, index𝛉 = FHMDDM.concatenateparameters(model)
	∇nℓ = similar(𝛉)
	FHMDDM.precisionmatrix!(model.gaussianprior, rand(length(index𝛂)), rand(length(index𝐒)))
	if simulate
		N_𝚽 = length(index𝚽)
		𝐑 = 1 .- 2rand(N_𝚽,N_𝚽)
		𝐁₀ = transpose(𝐑)*𝐑 # simulate a positive-definite Hessian of the posterior
		𝐇 = 𝚽 - 𝐁₀
		𝐰₀ = 1 .- 2rand(N_𝚽)
		𝐰₀ ./= norm(𝐰₀)
		𝐁₀𝐰₀ = 𝐁₀*𝐰₀
		for i in eachindex(index𝚽)
			𝛉[index𝚽[i]] = 𝐰₀[i]
		end
		FHMDDM.sortparameters!(model, 𝛉, index𝛉)
		FHMDDM.real2native!(model.θnative, model.options, model.θreal)
	else
		FHMDDM.initializeparameters!(model)
		FHMDDM.maximizeposterior!(model)
		𝐇 = FHMDDM.∇∇loglikelihood(model)[index𝚽, index𝚽]
		𝐰₀ = FHMDDM.concatenateparameters(model)[1][index𝚽]
		𝐁₀𝐰₀ = (𝚽-𝐇)*𝐰₀
	end
	𝛂𝐬 = vcat(rand(length(index𝛂)), rand(length(index𝐒)))
	memory = FHMDDM.Memoryforgradient(model)
	handcoded_evidence = FHMDDM.logevidence!(memory, model, 𝛉, 𝛂𝐬, 𝐁₀𝐰₀, 𝐇)
	handcoded_gradient = fill(NaN,length(𝛂𝐬))
	FHMDDM.∇negativelogevidence!(memory, model, handcoded_gradient, ∇nℓ, 𝛉, 𝛂𝐬, 𝐁₀𝐰₀, 𝐇)
    f(x) = FHMDDM.logevidence(x, 𝐁₀𝐰₀, 𝐇, model)
	automatic_evidence = f(𝛂𝐬)
	automatic_gradient = ForwardDiff.gradient(f, 𝛂𝐬)
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
function logevidence(𝛂𝐬::Vector{type},
					𝐁₀𝐰₀::Vector{<:Real},
					𝐇::Matrix{<:Real},
					model::Model) where{type<:Real}
	gaussianprior = GaussianPrior(model.options, model.trialsets, 𝛂𝐬)
	@unpack index𝚽, 𝚽 = gaussianprior
	𝐁 = 𝚽-𝐇
    𝐰 = 𝐁 \ 𝐁₀𝐰₀
	𝛉, index𝛉 = concatenateparameters(model)
	𝛉 = 𝛉 .- zero(type)
	@inbounds for i in eachindex(index𝚽)
		𝛉[index𝚽[i]] = 𝐰[i]
	end
	ℓ = loglikelihood(𝛉, index𝛉, model)
	logevidence(𝐇, ℓ, 𝚽, 𝐰)
end

"""
    geomean(a)
Return the geometric mean of a real-valued vector.
"""
function geomean(a::Vector{<:Real})
    s = 0.0
    n = length(a)
    for i = 1 : n
        @inbounds s += log(a[i])
    end
    return exp(s / n)
end
