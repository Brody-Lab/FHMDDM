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
julia> model = Model("/mnt/cup/labs/brody/tzluo/analysis_data/analysis_2022_06_05b_test/T176_2018_05_03/data.mat")
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
	memory = FHMDDM.Memoryforgradient(model)
	index𝛂 = FHMDDM.indexprecisions(model)
	𝛂₀ = model.precisionmatrix.diag[index𝛂]
	𝐀₀ = Diagonal(𝛂₀)
	best𝐸 = -Inf
	best𝛂 = copy(𝛂₀)
	𝛟, indexθ = FHMDDM.concatenateparameters(model)
	best𝛟 = copy(𝛟)
	n_consecutive_failures = 0
	posteriorconverged = false
	for i = 1:outer_iterations
		sortparameters!(model, 𝛟, indexθ)
	    gradientnorms = maximizeposterior!(model; iterations=iterations, g_tol=g_tol)[2]
		concatenatedθ = FHMDDM.concatenateparameters(model)[1]
		𝛟 .= concatenatedθ
		if gradientnorms[findlast(x->!isnan(x), gradientnorms)] > g_tol
			two_times_geomean = 2geomean(model.precisionmatrix.diag[index𝛂])
			model.precisionmatrix.diag[index𝛂] .= two_times_geomean
			verbose && println("Outer iteration: ", i, ": because a critical point could not be found, the values of the precisions are set to be twice the geometric mean of the hyperparameters. New 𝛂 → ", two_times_geomean)
		else
			verbose && println("Outer iteration: ", i, ": the MAP values of the parameters converged")
			for j in eachindex(index𝛂)
				𝛂₀[j] = model.precisionmatrix.diag[index𝛂[j]]
			end
			stats = @timed ∇∇loglikelihood(model)[index𝛂, index𝛂]
			𝐇 = stats.value
			verbose && println("Outer iteration: ", i, ": computing the Hessian of the log-likelihood took ", stats.time, " seconds")
			𝛉ₘₐₚ = concatenatedθ[index𝛂]
			𝐁₀𝛉ₘₐₚ = (𝐀₀-𝐇)*𝛉ₘₐₚ
			𝐸 = logevidence!(concatenatedθ, memory, model, 𝛂₀, 𝐁₀𝛉ₘₐₚ, 𝐇, index𝛂)
			if 𝐸 > best𝐸
				if verbose
					if posteriorconverged
						println("Outer iteration: ", i, ": the log-evidence (best: ", best𝐸, "; new:", 𝐸, ") is improved by the new values of the precisions found in the previous outer iteration")
					else
						println("Outer iteration: ", i, ": initial value of log-evidence: ", 𝐸, " is set as the best log-evidence")
					end
				end
				best𝐸 = 𝐸
				best𝛂 .= 𝛂₀
				best𝛟 .= 𝛟
				n_consecutive_failures = 0
			else
				n_consecutive_failures += 1
				verbose && println("Outer iteration: ", i, ": because the log-evidence (best: ", best𝐸, "; new:", 𝐸, ") was not improved by the new precisions, subsequent learning of the precisions will be begin at the midpoint between the current values of the precisions and the values that gave the best evidence so far.")
				for j in eachindex(index𝛂)
					model.precisionmatrix.diag[index𝛂[j]] = (model.precisionmatrix.diag[index𝛂[j]] + best𝛂[j])/2
				end
			end
			posteriorconverged = true
			if n_consecutive_failures == max_consecutive_failures
				verbose && println("Outer iteration: ", i, ": optimization halted early due to ", max_consecutive_failures, " consecutive failures in improving evidence")
				break
			end
		    normΔ = maximizeevidence!(memory, model, 𝐀₀, 𝐁₀𝛉ₘₐₚ, 𝐇, index𝛂, 𝛉ₘₐₚ)
			if verbose
				println("Outer iteration ", i, ": new 𝛂 → ", model.precisionmatrix.diag[index𝛂])
			end
			if normΔ < x_reltol
				if verbose
					println("Outer iteration: ", i, ": optimization halted after relative difference in the norm of the hyperparameters (in real space) decreased below ", x_reltol)
				end
				break
			end
		end
		if (i==outer_iterations) && verbose
			println("Optimization halted after reaching the last of ", outer_iterations, " allowed outer iterations.")
		end
	end
	println("Best log-evidence: ", best𝐸)
	println("Best hyperparameters: ", best𝛂)
	println("Best parameters: ", best𝛟)
	for j in eachindex(index𝛂)
		model.precisionmatrix.diag[index𝛂[j]] = best𝛂[j]
	end
	sortparameters!(model, best𝛟, indexθ)
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
						𝐇::Matrix{<:Real},
						index𝛂::Vector{<:Integer},
						𝛉ₘₐₚ::Vector{<:Real};
						αrange::Vector{<:Real}=[1e-1, 1e2],
						optimizationoptions::Optim.Options=Optim.Options(iterations=15, show_trace=true, show_every=1),
						optimizer::Optim.FirstOrderOptimizer=LBFGS(linesearch=LineSearches.BackTracking()))
	concatentatedθ = FHMDDM.concatenateparameters(model)[1]
	function f(𝐱)
		𝛂 = real2native.(𝐱, αrange[1], αrange[2])
		-FHMDDM.logevidence!(concatentatedθ, memory, model, 𝛂, 𝐁₀𝛉ₘₐₚ, 𝐇, index𝛂)
	end
	function g!(∇n𝐸, 𝐱)
		𝛂 = real2native.(𝐱, αrange[1], αrange[2])
		FHMDDM.∇negativelogevidence!(concatentatedθ, memory, model, ∇n𝐸, 𝛂, 𝐁₀𝛉ₘₐₚ, 𝐇, index𝛂)
		for i in eachindex(∇n𝐸)
			∇n𝐸[i] *= differentiate_native_wrt_real(𝐱[i], αrange[1], αrange[2])
		end
		return nothing
	end
	𝐱₀ = native2real.(model.precisionmatrix.diag[index𝛂], αrange[1], αrange[2])
	optimizationresults = Optim.optimize(f, g!, 𝐱₀, optimizer, optimizationoptions)
	𝐱̂ = Optim.minimizer(optimizationresults)
	normΔ = 0.0
	for i in eachindex(𝐱̂)
		x₀ᵢ = native2real(𝐀₀.diag[i], αrange[1], αrange[2])
		normΔ += (𝐱̂[i]/x₀ᵢ - 1.0)^2
		model.precisionmatrix.diag[index𝛂[i]] = real2native(𝐱̂[i], αrange[1], αrange[2])
	end
	return √normΔ
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
					𝐇::Matrix{<:Real},
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
function logevidence(𝐀::Diagonal, 𝐇::Matrix{<:Real}, ℓ::Real, 𝐰::Vector{<:Real})
	𝐌 = I - 𝐀^-1*𝐇
	if det(𝐌) < 0
		println("negative determinant") # try/catch block is much slower than conditional branching
		-Inf
	else
		ℓ - 0.5dot(𝐰, 𝐀, 𝐰) - 0.5logdet(I - 𝐀^-1*𝐇)
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
								𝐇::Matrix{<:Real},
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
	∇n𝐸 .= 𝐉'*(∇nℓ[index𝛂] .- 0.5.*𝐁₀𝛉ₘₐₚ .+ 𝐀*𝐰)
	𝚲 = I - 𝐀^-1*𝐇
	𝐐 = zeros(size(𝚲));
	@inbounds for i in eachindex(𝛂)
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
julia> model = Model("/mnt/cup/labs/brody/tzluo/analysis_data/analysis_2022_06_05b_test/T176_2018_05_03/data.mat")
julia> max_abs_norm_diff_∇𝐸, abs_norm_diff_𝐸 = FHMDDM.check_∇logevidence(model; simulate=true)
julia> model = Model("/mnt/cup/labs/brody/tzluo/analysis_data/analysis_2022_06_05b_test/T176_2018_05_03/data.mat")
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
		FHMDDM.initializeparameters!(memory, model)
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
