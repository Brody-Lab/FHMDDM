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
	    gradientnorms = maximizeposterior!(model; iterations=iterations, g_tol=g_tol)[2]
		if gradientnorms[findlast(x->!isnan(x), gradientnorms)] > g_tol
			new_α = min(100.0, 2geomean(model.gaussianprior.𝛂))
			model.gaussianprior.𝛂 .= new_α
			new_s = min(100.0, 2geomean(model.gaussianprior.𝐬))
			model.gaussianprior.𝐬 .= new_s
			verbose && println("Outer iteration: ", i, ": because a critical point could not be found, the values of the precisions are set to be twice the geometric mean of the hyperparameters. New (𝛂, 𝐬) → (", new_α, ", ", new_s, ")")
		else
			verbose && println("Outer iteration: ", i, ": the MAP values of the parameters converged")
			𝛉₀ = concatenateparameters(model)[1] # exact posterior mode
			stats = @timed ∇∇loglikelihood(model)[index𝚽, index𝚽]
			𝐇 = stats.value
			verbose && println("Outer iteration: ", i, ": computing the Hessian of the log-likelihood took ", stats.time, " seconds")
			𝐸 = logevidence!(memory, model, 𝛉₀)
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
	precisionmatrix!(model.gaussianprior, vcat(best𝛂,best𝐬))
	sortparameters!(model, best𝛉, index𝛉)
	return nothing
end

"""
	logevidence!(memory, model, 𝛉)

Log of the approximate marginal likelihood

MODIFIED ARGUMENT
-`memory`: a structure containing memory for in-place computation
-`model`: structure containing the parameters, hyperparameters, and data

UNMODIFIED ARGUMENT
-`𝛉`: posterior mode
"""
function logevidence!(memory::Memoryforgradient, model::Model, 𝛉::Vector{<:Real})
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
function logevidence(𝐇::Matrix{<:Real}, ℓ::Real, 𝚽::Matrix{<:Real}, 𝐰::Vector{<:Real})
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
	maximizeevidence!(memory, model, 𝐀₀, 𝐇, index𝛂, 𝛉ₘₐₚ)

Learn hyperparameters by fixing the parameters of the model and maximizing the evidence

MODIFIED ARGUMENT
-`memory`: structure containing variables to be modified during computations
-`model`: structure containing the parameters, hyperparameters, and data. The parameter values are modified, but the hyperparameters are not modified
-`𝐀₀`: the precision matrix used to compute the MAP solution `𝛉ₘₐₚ`
-`𝐇`: Hessian of the log-likelihood evaluated at the MAP solution `𝛉ₘₐₚ`
-`𝛉ₘₐₚ`: MAP solution computed using the precision matrix 𝐀₀

RETURN
-Euclidean of the normalized difference in the log-precisions being optimized
"""
function maximizeevidence!(memory::Memoryforgradient,
						model::Model,
						𝐇::Matrix{<:Real},
						𝛉₀::Vector{<:Real};
						αrange::Vector{<:Real}=[1e-1, 1e2],
						optimizationoptions::Optim.Options=Optim.Options(iterations=15, show_trace=true, show_every=1),
						optimizer::Optim.FirstOrderOptimizer=LBFGS(linesearch=LineSearches.BackTracking()))
	𝐰₀ = 𝛉₀[model.gaussianprior.index𝚽]
	𝐁₀𝐰₀ = (model.gaussianprior.𝚽-𝐇)*𝐰₀
	𝛂₀𝐬₀ = vcat(model.gaussianprior.𝛂, model.gaussianprior.𝐬)
	𝐱₀ = 𝛂₀𝐬₀
	for i in eachindex(𝛂₀𝐬₀)
		𝐱₀[i] = native2real(𝛂₀𝐬₀[i], αrange[1], αrange[2])
	end
	𝛉 = concatenateparameters(model)[1]
	function f(𝐱)
		𝛂𝐬 = real2native.(𝐱, αrange[1], αrange[2])
		-logevidence!(memory, model, 𝛉, 𝛂𝐬, 𝐁₀𝐰₀, 𝐇)
	end
	function g!(∇n𝐸, 𝐱)
		𝛂𝐬 = real2native.(𝐱, αrange[1], αrange[2])
		∇negativelogevidence!(memory, model, ∇n𝐸, 𝛉, 𝛂𝐬, 𝐁₀𝐰₀, 𝐇)
		for i in eachindex(∇n𝐸)
			∇n𝐸[i] *= differentiate_native_wrt_real(𝐱[i], αrange[1], αrange[2])
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
	for i in eachindex(𝐱̂)
		𝛂̂𝐬̂[i] = real2native(𝐱̂[i], αrange[1], αrange[2])
	end
	precisionmatrix!(model.gaussianprior, 𝛂̂𝐬̂)
	return √normΔ
end

"""
	logevidence!(memory, model, 𝛂, 𝐁₀𝛉ₘₐₚ, 𝐇, index𝛂)

Log of the marginal likelihood

MODIFIED ARGUMENT
-`𝐁₁`: approximate hessian matrix of the posterior probability, as a function of the hyperparameters. This contains only the subset of parameters with finite variance.
-`𝛉₁`: approximate posterior mode as a function, as a function of the hyperparameters.
-`memory`: a structure containing memory for in-place computation
-`model`: structure containing the parameters, hyperparameters, and data

UNMODIFIED ARGUMENT
-`𝐀₀`: precision matrix that was fixed and used to find the MAP values of the parameters
-`𝛂𝐬`: precisions being learned
-`𝐇`: Hessian of the log-likelihood evaluated at the MAP values of the parameters
-`index𝛂`: index of the precisions being fit within the full vector of concatenated precisions
-`𝐁₀𝛉ₘₐₚ`: Hessian of the log-posterior evalued at the MAP values of the parameters multiplied by the MAP value of the parameters

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
    𝐰 = (𝚽-𝐇) \ 𝐁₀𝐰₀
	𝛉[index𝚽] .= 𝐰
	loglikelihood!(model, memory, 𝛉)
	logevidence(𝚽, 𝐇, memory.ℓ[1], 𝐰)
end

"""
	∇negativelogevidence!(∇n𝐸, memory, model, 𝛂, 𝐇, 𝐁₀𝛉ₘₐₚ)

gradient of the negative log of the marginal likelihood

MODIFIED ARGUMENT
-`∇n𝐸`: gradient of the negative of the log-evidence
-`𝐀`: precision matrix
-`memory`: a structure containing memory for in-place computation
-`model`: structure containing the parameters, hyperparameters, and data
-`𝐰`: posterior mode as a function of the hyperparameters

UNMODIFIED ARGUMENT
-`𝛂`: precisions being learned
-`𝐇`: Hessian of the log-likelihood evaluated at the MAP values of the parameters, including only rows and columns corresponding to the hyperparameters being optimized
-`index𝛂`: index of the precisions being fit within the full vector of concatenated precisions
-`𝐁₀𝛉ₘₐₚ`: Hessian of the log-posterior evalued at the MAP values of the parameters multiplied by the MAP value of the parameters
"""
function ∇negativelogevidence!(memory::Memoryforgradient,
								model::Model,
								∇n𝐸::Vector{<:Real},
								𝛉::Vector{<:Real},
								𝛂𝐬::Vector{<:Real},
								𝐁₀𝐰₀::Vector{<:Real},
								𝐇::Matrix{<:Real})
	precisionmatrix!(model.gaussianprior, 𝛂𝐬)
	@unpack index𝚽, 𝚽, 𝐒, index𝛂_in_index𝚽, index𝐒_in_index𝚽 = model.gaussianprior
	𝐁 = 𝚽-𝐇
    𝐰 = 𝐁 \ 𝐁₀𝐰₀
	𝛉[index𝚽] .= 𝐰
	∇nℓ = similar(𝛉)
	∇negativeloglikelihood!(∇nℓ, memory, model, 𝛉)
	𝐦 = 𝐁 \ (𝚽*𝐰 + ∇nℓ[index𝚽]) # there might be a way to avoid the repeat inversions of B
	𝛀 = (𝐁 \ (𝚽 \ 𝐇)')'
	N_𝛂 = length(model.gaussianprior.𝛂)
	for i=1:N_𝛂
		k = index𝛂_in_index𝚽[i]
		∇n𝐸[i] = 0.5*(𝐰[k]^2 + 𝛀[k,k]) - 𝐰[k]*𝐦[k]
	end
	for j = 1:length(model.gaussianprior.𝐬)
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
julia> model = Model("/mnt/cup/labs/brody/tzluo/analysis_data/analysis_2022_06_14a_test/T176_2018_05_03_b2K2K2/data.mat")
julia> max_abs_norm_diff_∇𝐸, abs_norm_diff_𝐸 = FHMDDM.check_∇logevidence(model; simulate=true)
julia> model = Model("/mnt/cup/labs/brody/tzluo/analysis_data/analysis_2022_06_14a_test/T176_2018_05_03_b2K2K2/data.mat")
julia> max_abs_norm_diff_∇𝐸, abs_norm_diff_𝐸 = FHMDDM.check_∇logevidence(model; simulate=false)
```
"""
function check_∇logevidence(model::Model; simulate::Bool=true)
	index𝛂 = FHMDDM.indexprecisions(model)
	D = length(index𝛂)
	𝛂 = rand(D)
	concatenatedθ, indexθ = FHMDDM.concatenateparameters(model)
	memory = FHMDDM.Memoryforgradient(model)
	𝛂₀ = model.precisionmatrix.diag[index𝛂]
	𝐀₀ = Diagonal(𝛂₀)
	if simulate
		𝐑 = 1 .- 2rand(D,D)
		𝐇 = Diagonal(𝛂) - transpose(𝐑)*𝐑
		𝐀 = Diagonal(𝛂)
		𝐁 = 𝐀-𝐇
		𝐰 = 1 .- 2rand(D)
		𝐰 ./= norm(𝐰)
		𝐁₀𝛉ₘₐₚ = 𝐁*𝐰
		𝛉ₘₐₚ = (𝐀₀-𝐇) \ 𝐁₀𝛉ₘₐₚ
		for i in eachindex(index𝛂)
			concatenatedθ[index𝛂[i]] = 𝐰[i]
		end
		FHMDDM.sortparameters!(model, concatenatedθ, indexθ)
		FHMDDM.real2native!(model.θnative, model.options, model.θreal)
	else
		FHMDDM.initializeparameters!(model)
		FHMDDM.maximizeposterior!(model)
		𝐇 = ∇∇loglikelihood(model)[index𝛂, index𝛂]
		𝛉ₘₐₚ = concatenateparameters(model)[1][index𝛂]
		𝐁₀𝛉ₘₐₚ = (𝐀₀-𝐇)*𝛉ₘₐₚ
	end
	handcoded_evidence = FHMDDM.logevidence!(concatenatedθ, memory, model, 𝛂, 𝐁₀𝛉ₘₐₚ, 𝐇, index𝛂)
	handcoded_gradient = fill(NaN,D)
	FHMDDM.∇negativelogevidence!(concatenatedθ, memory, model, handcoded_gradient, 𝛂, 𝐁₀𝛉ₘₐₚ, 𝐇, index𝛂)
    f(x) = FHMDDM.logevidence(x, 𝐁₀𝛉ₘₐₚ, 𝐇, index𝛂, model)
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
function logevidence(𝛂::Vector{type},
					𝐁₀𝛉ₘₐₚ::Vector{<:Real},
					𝐇::Matrix{<:Real},
					index𝛂::Vector{<:Integer},
					model::Model) where{type<:Real}
	𝐀 = Diagonal(𝛂)
	𝐁 = 𝐀-𝐇
    𝐰 = 𝐁 \ 𝐁₀𝛉ₘₐₚ
	concatenatedθ, indexθ = concatenateparameters(model)
	concatenatedθ = concatenatedθ .- zero(type)
	@inbounds for i in eachindex(index𝛂)
		concatenatedθ[index𝛂[i]] = 𝐰[i]
	end
	ℓ = FHMDDM.loglikelihood(concatenatedθ, indexθ, model)
	logevidence(𝐀, 𝐇, ℓ, 𝐰)
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
