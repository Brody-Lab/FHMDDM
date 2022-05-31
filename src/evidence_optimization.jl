"""
	maximizeevidence!(model)

Learn both the parameters and hyperparameters by maximizing the evidence

The optimization procedure alternately fixes the hyperparameters and learn the parameters by maximizing the posterior and fixes the parameters and learn the hyperparameters by maximing the evidence.

MODIFIED ARGUMENT
-`model`: structure containing the parameters, hyperparameters, and data
"""
function maximizeevidence!(model::Model)
    maximizeposterior!(model)
    𝐇 = ∇∇loglikelihood(model)
    𝛉ₘₐₚ = concatenateparameters(model)[1]
    𝐀₀ = copy(model.priorprecision)
	memory = Memoryforgradient(model)
    maximizeevidence!(model, 𝐇)
	return nothing
end

"""
	maximizeevidence!(model, ...)

Learn hyperparameters by fixing the parameters of the model and maximizing the evidence

MODIFIED ARGUMENT
-`model`: structure containing the parameters, hyperparameters, and data
"""
function maximizeevidence!(memory::Memoryforgradient,
						model::Model,
						𝐀₀::Diagonal,
						𝐇::Symmetric,
						𝐁₀𝛉ₘₐₚ::Vector{<:Real};
						optimizationoptions::Optim.Options=Optim.Options(),
						optimizer::Optim.FirstOrderOptimizer=LBFGS(linesearch=LineSearches.BackTracking()))
	𝐀₀ = copy(model.precisionmatrix)
	𝛉ₘₐₚ = concatenateparameters(model)[1]
    𝐇 = ∇∇loglikelihood(model)
	𝐁₀𝛉ₘₐₚ = (𝐀₀-𝐇)*𝛉ₘₐₚ
    f(𝛂) = -logevidence!(memory, model, 𝐀₀, 𝛂, 𝐇, 𝐁₀𝛉ₘₐₚ)
	g!(∇n𝐸, 𝛂) = ∇negativelogevidence!(∇n𝐸, memory, model, 𝐀₀, 𝛂, 𝐇, 𝐁₀𝛉ₘₐₚ)
	optimizationresults = Optim.optimize(f, g!, 𝐀₀.diag, optimizer, optimizationoptions)
end

"""
	logevidence!(memory, model, 𝛂, 𝐇, 𝐁₀𝛉ₘₐₚ)

Log of the marginal likelihood

MODIFIED ARGUMENT
-`memory`: a structure containing memory for in-place computation
-`model`: structure containing the parameters, hyperparameters, and data

UNMODIFIED ARGUMENT
-`𝐀₀`: precision matrix that was fixed and used to find the MAP values of the parameters
-`𝛂`: diagonal of the precision matrix; these are the values that are being learned
-`𝐇`: Hessian of the log-likelihood evaluated at the MAP values of the parameters
-`𝐁₀𝛉ₘₐₚ`: Hessian of the log-posterior evalued at the MAP values of the parameters multiplied by the MAP value of the parameters

RETURN
-log of the evidence
"""
function logevidence!(memory::Memoryforgradient,
					model::Model,
					𝛂::Vector{<:Real},
					𝐇::Symmetric,
					𝐁₀𝛉ₘₐₚ::Vector{<:Real})
	𝐀 = Diagonal(𝛂)
	𝐁 = 𝐀-𝐇
    𝐰 = 𝐁 \ 𝐁₀𝛉ₘₐₚ
	loglikelihood!(model, memory, 𝐰)
	logevidence(𝐀, 𝐇, memory.ℓ[1], 𝐰)
end

"""
	logevidence(𝐀, 𝐇, ℓ, 𝐰)

Evaluate the log-evidence

ARGUMENT
-`𝐀`: precision matrix
-`𝐇`: Hessian of the log-likelihood evaluated at the MAP values of the parameters
-`ℓ`: log-likelihood evaluated at the approximate posterior mode 𝐰
-`𝐰`: approximate posterior mode as a function of 𝐀

RETURN
-log evidence
"""
function logevidence(𝐀::Diagonal, 𝐇::Symmetric, ℓ::Real, 𝐰::Vector{<:Real})
	ℓ - 0.5𝐰'*𝐀*𝐰 - 0.5logdet(I - 𝐀^-1*𝐇)
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
-`𝐀₀`: precision matrix that was fixed and used to find the MAP values of the parameters
-`𝛂`: diagonal of the precision matrix; these are the values that are being learned
-`𝐇`: Hessian of the log-likelihood evaluated at the MAP values of the parameters
-`𝐁₀𝛉ₘₐₚ`: Hessian of the log-posterior evalued at the MAP values of the parameters multiplied by the MAP value of the parameters
"""
function ∇negativelogevidence!(∇n𝐸::Vector{<:Real},
								memory::Memoryforgradient,
								model::Model,
								𝛂::Vector{<:Real},
								𝐇::Symmetric,
								𝐁₀𝛉ₘₐₚ::Vector{<:Real})
	𝐀 = Diagonal(𝛂)
	𝐁 = 𝐀-𝐇
    𝐰 = 𝐁 \ 𝐁₀𝛉ₘₐₚ
	∇nℓ = ∇n𝐸 # reuse memory
	FHMDDM.∇negativeloglikelihood!(∇nℓ, memory, model, 𝐰)
	𝐉 = -𝐁 \ Diagonal(𝐰) #Jacobian matrix of the posterior mode 𝐰 with respect to the precisions 𝛂
	∇n𝐸 .= 𝐉'*(∇nℓ - 0.5𝐁₀𝛉ₘₐₚ + 𝐀*𝐰)
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
-absolute difference between the gradients
-absolute difference between the log-evidence functions

EXAMPLE
```julia-repl
julia> using FHMDDM
julia> model = Model("/mnt/cup/labs/brody/tzluo/analysis_data/analysis_2022_05_21_test/T176_2018_05_03/data.mat")
julia> absdiff∇𝐸, absidff𝐸 = FHMDDM.check_∇logevidence(model; simulate=true)
```
"""
function check_∇logevidence(model::Model; simulate::Bool=true)
	D = size(model.precisionmatrix,1)
	𝛂 = rand(D)
	if simulate
		𝐑 = 1 .- 2rand(D,D)
		𝐇 = Symmetric(Diagonal(𝛂) - transpose(𝐑)*𝐑)
		𝐀 = Diagonal(𝛂)
		𝐁 = 𝐀-𝐇
		𝐰 = 1 .- 2rand(D)
		𝐰 ./= norm(𝐰)
		𝐁₀𝛉ₘₐₚ = 𝐁*𝐰
		𝐀₀ = model.precisionmatrix
		𝛉ₘₐₚ = (𝐀₀-𝐇) \ 𝐁₀𝛉ₘₐₚ
		indexθ = FHMDDM.concatenateparameters(model)[2]
		FHMDDM.sortparameters!(model, 𝛉ₘₐₚ, indexθ)
		FHMDDM.real2native!(model.θnative, model.options, model.θreal)
	else
		FHMDDM.initializeparameters!(model)
		FHMDDM.maximizeposterior!(model)
		𝐇 = Symmetric(FHMDDM.∇∇loglikelihood(model))
		𝛉ₘₐₚ, indexθ = FHMDDM.concatenateparameters(model)
		𝐀₀ = model.precisionmatrix
		𝐁₀𝛉ₘₐₚ = (𝐀₀-𝐇)*𝛉ₘₐₚ
	end
	memory = FHMDDM.Memoryforgradient(model)
	handcoded_evidence = FHMDDM.logevidence!(memory, model, 𝛂, 𝐇, 𝐁₀𝛉ₘₐₚ)
	handcoded_gradient = fill(NaN,D)
	FHMDDM.∇negativelogevidence!(handcoded_gradient, memory, model, 𝛂, 𝐇, 𝐁₀𝛉ₘₐₚ)
    f(x) = FHMDDM.logevidence(x, 𝐁₀𝛉ₘₐₚ, 𝐇, indexθ, model)
	automatic_evidence = f(𝛂)
	automatic_gradient = ForwardDiff.gradient(f, 𝛂)
	return abs.(automatic_gradient .+ handcoded_gradient), abs(automatic_evidence-handcoded_evidence)
end

"""
	logevidence(𝛂, 𝐁₀𝛉ₘₐₚ, 𝐇, indexθ, model)

ForwardDiff-computation evaluation of the log-evidence

ARGUMENT
-`𝛂`: diagonal of the precision matrix; these are the values that are being learned
-`𝐁₀𝛉ₘₐₚ`: Hessian of the log-posterior evalued at the MAP values of the parameters multiplied by the MAP value of the parameters
-`𝐇`: Hessian of the log-likelihood evaluated at the MAP values of the parameters
-`indexθ`: index of the parameters
-`model`: structure containing the parameters, hyperparameters, and data

RETURN
-log of the marginal likelihood
"""
function logevidence(𝛂::Vector{<:Real},
					𝐁₀𝛉ₘₐₚ::Vector{<:Real},
					𝐇::Symmetric,
					indexθ::Indexθ,
					model::Model)
	𝐀 = Diagonal(𝛂)
	𝐁 = 𝐀-𝐇
    𝐰 = 𝐁 \ 𝐁₀𝛉ₘₐₚ
	ℓ = FHMDDM.loglikelihood(𝐰, indexθ, model)
	logevidence(𝐀, 𝐇, ℓ, 𝐰)
end
