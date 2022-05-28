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
						𝐀₀::Diagonal{<:Real, <:Vector{<:Real}},
						𝐇::Matrix{<:Real},
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
					𝐇::Matrix{<:Real},
					𝐁₀𝛉ₘₐₚ::Vector{<:Real})
	𝐀 = model.precisionmatrix
	if 𝛂 != 𝐀.diag
		𝐀.diag .= 𝛂
	end
	𝐁 = 𝐀-𝐇
    𝐰 = 𝐁 \ 𝐁₀𝛉ₘₐₚ
	FHMDDM.loglikelihood!(model, memory, 𝐰)
	memory.ℓ[1] - 0.5*(transpose(𝐰)*𝐀*𝐰) - 0.5*logdet(𝐀) + 0.5*logdet(𝐁)
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
								𝐇::Matrix{<:Real},
								𝐁₀𝛉ₘₐₚ::Vector{<:Real})
	𝐀 = model.precisionmatrix
	if 𝛂 != 𝐀.diag
		𝐀.diag .= 𝛂
	end
	𝐁 = 𝐀-𝐇
    𝐰 = 𝐁 \ 𝐁₀𝛉ₘₐₚ
	∇nℓ = ∇n𝐸 # reuse memory
	∇negativeloglikelihood!(∇nℓ, memory, model, 𝐰)
	J𝐰 = -𝐁*Diagonal(𝐁*𝐁₀𝛉ₘₐₚ) #Jacobian matrix of the posterior mode 𝐰 with respect to the precisions 𝛂
	∇n𝐸 .= J𝐰*(-∇nℓ) - 0.5*𝐰.^2 + J𝐰*𝐀*𝐰 - 0.5*(1.0./𝛂) + 0.5*diag(inv(𝐁))
	return nothing
end

"""
	check_∇logevidence(model)

Check whether the hand-coded gradient of the log-evidence matches the automatic gradient

ARGUMENT
-`model`: structure containing the parameters and hyperparameters

RETURN
-absolute difference between the gradients
-absolute difference between the log-evidence functions

EXAMPLE
```julia-repl
julia> using FHMDDM
julia> model = Model("/mnt/cup/labs/brody/tzluo/analysis_data/analysis_2022_05_21_test/T176_2018_05_03/data.mat")
julia> FHMDDM.initializemodel!(model)
julia> FHMDDM.maximizeposterior!(model)
julia> absdiff∇𝐸, absidff𝐸 = FHMDDM.check_∇logevidence(model)
```
"""
function check_∇logevidence(model::Model)
	memory = FHMDDM.Memoryforgradient(model)
	𝐀₀ = copy(model.precisionmatrix)
    𝐇 = FHMDDM.∇∇loglikelihood(model)
	𝛉ₘₐₚ, indexθ = FHMDDM.concatenateparameters(model)
	𝐁₀𝛉ₘₐₚ = (𝐀₀-𝐇)*𝛉ₘₐₚ
	𝛂 = ones(length(𝛉ₘₐₚ))
	handcoded_gradient = similar(𝛉ₘₐₚ)
	handcoded_evidence = FHMDDM.logevidence!(memory, model, 𝛂, 𝐇, 𝐁₀𝛉ₘₐₚ)
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
					𝐇::Matrix{<:Real},
					indexθ::Indexθ,
					model::Model)
	𝐀 = Diagonal(𝛂)
	𝐁 = 𝐀-𝐇
    𝐰 = 𝐁 \ 𝐁₀𝛉ₘₐₚ
	FHMDDM.loglikelihood(𝐰, indexθ, model) - 0.5*(transpose(𝐰)*𝐀*𝐰) - 0.5*logdet(𝐀) + 0.5*logdet(𝐁)
end
