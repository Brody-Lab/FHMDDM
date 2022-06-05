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
julia> model = Model("/mnt/cup/labs/brody/tzluo/analysis_data/analysis_2022_06_04_test/T176_2018_05_03/data.mat")
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
	memory = FHMDDM.Memoryforgradient(model)
	initializeparameters!(memory, model)
	index𝛂 = FHMDDM.indexprecisions(model)
	𝛂₀ = model.precisionmatrix.diag[index𝛂]
	𝐀₀ = Diagonal(𝛂₀)
	best𝐸 = -Inf
	best𝛂 = copy(𝛂₀)
	concatenatedθ, indexθ = FHMDDM.concatenateparameters(model)
	best𝛉ₘₐₚ = concatenatedθ[index𝛂]
	n_consecutive_failures = 0
	posteriorconverged = false
	for i = 1:outer_iterations
		sortparameters!(model, concatenatedθ, indexθ)
	    gradientnorms = maximizeposterior!(model; iterations=iterations, g_tol=g_tol)[2]
		if gradientnorms[findlast(x->!isnan(x), gradientnorms)] > g_tol
			minα = minimum(model.precisionmatrix.diag[index𝛂])
			minα10 = minα*10
			for j in eachindex(index𝛂)
				model.precisionmatrix.diag[index𝛂[j]] = max(model.precisionmatrix.diag[index𝛂[j]], minα10)
			end
			concatenatedθ = FHMDDM.concatenateparameters(model)[1]
			verbose && println("Outer iteration: ", i, ": because a critical point could not be found, the values of the precisions at least ten times the minimum value. New 𝛂 → ", model.precisionmatrix.diag[index𝛂])
		else
			for j in eachindex(index𝛂)
				𝛂₀[j] = model.precisionmatrix.diag[index𝛂[j]]
			end
			stats = @timed Symmetric(FHMDDM.∇∇loglikelihood(model)[index𝛂, index𝛂])
			𝐇 = stats.value
			verbose && println("Outer iteration: ", i, ": computing the Hessian of the log-likelihood took ", stats.time, " seconds")
			concatenatedθ = FHMDDM.concatenateparameters(model)[1]
			𝛉ₘₐₚ = concatenatedθ[index𝛂]
			𝐁₀𝛉ₘₐₚ = (𝐀₀-𝐇)*𝛉ₘₐₚ
			𝐸 = FHMDDM.logevidence!(concatenatedθ, memory, model, 𝛂₀, 𝐁₀𝛉ₘₐₚ, 𝐇, index𝛂)
			if 𝐸 > best𝐸
				if verbose && posteriorconverged
					println("Outer iteration: ", i, ": the evidence (best: ", best𝐸, "; new:", 𝐸, ") is improved by the new values of the precisions found in the previous outer iteration")
				end
				best𝐸 = 𝐸
				best𝛂 .= 𝛂₀
				best𝛉ₘₐₚ .= 𝛉ₘₐₚ
				n_consecutive_failures = 0
			else
				n_consecutive_failures += 1
				verbose && println("Outer iteration: ", i, ": because the evidence (best: ", best𝐸, "; new:", 𝐸, ") was not improved by the new precisions, subsequent learning of the precisions will be begin at the midpoint between the current values of the precisions and the values that gave the best evidence so far.")
				for j in eachindex(index𝛂)
					model.precisionmatrix.diag[index𝛂[j]] = (model.precisionmatrix.diag[index𝛂[j]] + best𝛂[j])/2
				end
			end
			posteriorconverged = true
			if n_consecutive_failures == max_consecutive_failures
				verbose && println("Outer iteration: ", i, ": optimization halted early due to ", max_consecutive_failures, " consecutive failures in improving evidence")
				break
			end
		    normΔlog𝛂 = maximizeevidence!(memory, model, 𝐀₀, 𝐁₀𝛉ₘₐₚ, 𝐇, index𝛂, 𝛉ₘₐₚ)
			if verbose
				println("Outer iteration ", i, ": new 𝛂 → ", model.precisionmatrix.diag[index𝛂])
			end
			if normΔlog𝛂 < x_reltol
				if verbose
					println("Outer iteration: ", i, ": optimization halted after relative difference in the norm of the hyperparameters decreased below ", x_reltol)
				end
				break
			end
		end
		if (i==outer_iterations) && verbose
			println("Optimization halted after reaching the last of ", outer_iterations, " allowed outer iterations")
		end
	end
	concatenatedθ, indexθ = concatenateparameters(model)
	for j in eachindex(index𝛂)
		model.precisionmatrix.diag[index𝛂[j]] = best𝛂[j]
		concatenatedθ[index𝛂[j]] = best𝛉ₘₐₚ[j]
	end
	sortparameters!(model, concatenatedθ, indexθ)
	verbose && println("Approximate log-evidence = ", best𝐸)
	return nothing
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
						𝐀₀::Diagonal,
						𝐁₀𝛉ₘₐₚ::Vector{<:Real},
						𝐇::Symmetric,
						index𝛂::Vector{<:Integer},
						𝛉ₘₐₚ::Vector{<:Real};
						αrange::Vector{<:Real}=[0.05, 1e6],
						optimizationoptions::Optim.Options=Optim.Options(iterations=15),
						optimizer::Optim.FirstOrderOptimizer=LBFGS(linesearch=LineSearches.BackTracking()))
	concatentatedθ = FHMDDM.concatenateparameters(model)[1]
    f(log𝛂) = -FHMDDM.logevidence!(concatentatedθ, memory, model, exp.(log𝛂), 𝐁₀𝛉ₘₐₚ, 𝐇, index𝛂)
	function g!(∇n𝐸, log𝛂)
		𝛂 = exp.(log𝛂)
		FHMDDM.∇negativelogevidence!(concatentatedθ, memory, model, ∇n𝐸, 𝛂, 𝐁₀𝛉ₘₐₚ, 𝐇, index𝛂)
		∇n𝐸 .*= 𝛂
		return nothing
	end
	initial_log𝛂 = log.(model.precisionmatrix.diag[index𝛂]) # note that we do not want to start from 𝐀₀.diag
	onesvector = ones(length(initial_log𝛂))
	lower = log(minimum(αrange)).*onesvector
	upper = log(maximum(αrange)).*onesvector
	od = OnceDifferentiable(f, g!, initial_log𝛂)
	optimizationresults = Optim.optimize(od, lower, upper, initial_log𝛂, Fminbox(optimizer), optimizationoptions)
	log𝛂̂ = Optim.minimizer(optimizationresults)
	normΔlog𝛂 = 0.0
	for i in eachindex(log𝛂̂)
		normΔlog𝛂 += (log𝛂̂[i]/log(𝐀₀.diag[i]) - 1.0)^2
		model.precisionmatrix.diag[index𝛂[i]] = exp(log𝛂̂[i])
	end
	return √normΔlog𝛂
end

"""
	logevidence!(memory, model, 𝛂, 𝐁₀𝛉ₘₐₚ, 𝐇, index𝛂)

Log of the marginal likelihood

MODIFIED ARGUMENT
-`memory`: a structure containing memory for in-place computation
-`model`: structure containing the parameters, hyperparameters, and data

UNMODIFIED ARGUMENT
-`𝐀₀`: precision matrix that was fixed and used to find the MAP values of the parameters
-`𝛂`: precisions being learned
-`𝐇`: Hessian of the log-likelihood evaluated at the MAP values of the parameters
-`index𝛂`: index of the precisions being fit within the full vector of concatenated precisions
-`𝐁₀𝛉ₘₐₚ`: Hessian of the log-posterior evalued at the MAP values of the parameters multiplied by the MAP value of the parameters

RETURN
-log of the evidence
"""
function logevidence!(concatenatedθ::Vector{<:Real},
					memory::Memoryforgradient,
					model::Model,
					𝛂::Vector{<:Real},
					𝐁₀𝛉ₘₐₚ::Vector{<:Real},
					𝐇::Symmetric,
					index𝛂::Vector{<:Integer})
	𝐀 = Diagonal(𝛂)
	𝐁 = 𝐀-𝐇
    𝐰 = 𝐁 \ 𝐁₀𝛉ₘₐₚ
	for i in eachindex(index𝛂)
		concatenatedθ[index𝛂[i]] = 𝐰[i]
	end
	loglikelihood!(model, memory, concatenatedθ)
	logevidence(𝐀, 𝐇, memory.ℓ[1], 𝐰)
end

"""
	logevidence(𝐀, 𝐇, ℓ, 𝐰)

Evaluate the log-evidence

ARGUMENT
-`𝐀`: precision matrix, containing only the precisions being optimized
-`𝐇`: Hessian of the log-likelihood evaluated at the MAP values of the parameters
-`ℓ`: log-likelihood evaluated at the approximate posterior mode 𝐰
-`𝐰`: approximate posterior mode as a function of 𝐀ₒₚₜ for the precisions being optimized

RETURN
-log evidence
"""
function logevidence(𝐀::Diagonal, 𝐇::Symmetric, ℓ::Real, 𝐰::Vector{<:Real})
	𝐌 = I - 𝐀^-1*𝐇
	if det(𝐌) < 0
		println("negative determinant")
		-Inf
	else
		ℓ - 0.5𝐰'*𝐀*𝐰 - 0.5logdet(I - 𝐀^-1*𝐇)
	end
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
function ∇negativelogevidence!(concatenatedθ::Vector{<:Real},
								memory::Memoryforgradient,
								model::Model,
								∇n𝐸::Vector{<:Real},
								𝛂::Vector{<:Real},
								𝐁₀𝛉ₘₐₚ::Vector{<:Real},
								𝐇::Symmetric,
								index𝛂::Vector{<:Integer})
	𝐀 = Diagonal(𝛂)
	𝐁 = 𝐀-𝐇
    𝐰 = 𝐁 \ 𝐁₀𝛉ₘₐₚ
	for i in eachindex(index𝛂)
		concatenatedθ[index𝛂[i]] = 𝐰[i]
	end
	∇nℓ = similar(concatenatedθ)
	∇negativeloglikelihood!(∇nℓ, memory, model, concatenatedθ)
	𝐉 = -𝐁 \ Diagonal(𝐰) #Jacobian matrix of the posterior mode 𝐰 with respect to the precisions 𝛂
	∇n𝐸 .= 𝐉'*(∇nℓ[index𝛂] - 0.5𝐁₀𝛉ₘₐₚ + 𝐀*𝐰)
	𝚲 = I - 𝐀^-1*𝐇
	𝐐 = zeros(size(𝚲));
	for i in eachindex(𝛂)
		if i > 1
	    	𝐐[i-1,:] .= 0.0
		end
	    𝐐[i,:] = 𝛂[i]^-2 .* 𝐇[i,:]
	    ∇n𝐸[i] += 0.5tr(𝚲 \ 𝐐)
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
julia> model = Model("/mnt/cup/labs/brody/tzluo/analysis_data/analysis_2022_06_01_test/T176_2018_05_03/data.mat")
julia> max_abs_norm_diff_∇𝐸, abs_norm_diff_𝐸 = FHMDDM.check_∇logevidence(model; simulate=true)
julia> model = Model("/mnt/cup/labs/brody/tzluo/analysis_data/analysis_2022_06_01_test/T176_2018_05_03/data.mat")
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
		𝐇 = Symmetric(Diagonal(𝛂) - transpose(𝐑)*𝐑)
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
		FHMDDM.initializeparameters!(memory, model)
		FHMDDM.maximizeposterior!(model)
		𝐇 = Symmetric(∇∇loglikelihood(model)[index𝛂, index𝛂])
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
					𝐇::Symmetric,
					index𝛂::Vector{<:Integer},
					model::Model) where{type<:Real}
	𝐀 = Diagonal(𝛂)
	𝐁 = 𝐀-𝐇
    𝐰 = 𝐁 \ 𝐁₀𝛉ₘₐₚ
	concatenatedθ, indexθ = concatenateparameters(model)
	concatenatedθ = concatenatedθ .- zero(type)
	for i in eachindex(index𝛂)
		concatenatedθ[index𝛂[i]] = 𝐰[i]
	end
	ℓ = FHMDDM.loglikelihood(concatenatedθ, indexθ, model)
	logevidence(𝐀, 𝐇, ℓ, 𝐰)
end
