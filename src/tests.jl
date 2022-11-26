"""
    test(datapath)

Run a number of tests on the model

ARGUMENT
-`datapath`: full path of the data file
"""
function test(datapath::String; maxabsdiff::Real=1e-8)
	println("testing `"*datapath*"`")
	printseparator()
	println("testing cross-validation and saving results in `cvresults.mat`")
	model = Model(datapath)
	cvresults = crossvalidate(2, model)
	printseparator()
	println("saving cross-validation results in `cvresults.mat`")
	save(cvresults, model.options)
    printseparator()
    println("testing the hessian of the expectation of the log-likelihood of one neuron's spike train")
    println("	- `expectation_of_∇∇loglikelihood(γ, mpGLM, x)`")
    println("	- used for parameter initialization")
    test_expectation_of_∇∇loglikelihood!(datapath; maxabsdiff=maxabsdiff)
    printseparator()
    println("testing the gradient of the expectation of the log-likelihood of one neuron's spike train")
    println("	- 'expectation_∇loglikelihood!(∇Q, γ, mpGLM)'")
    println("	- used for learning of the parameters of the full model")
    test_expectation_∇loglikelihood!(datapath; maxabsdiff=maxabsdiff)
    printseparator()
    println("testing the gradient of log-likelihood of all the data")
    println("	- '∇negativeloglikelihood!(∇nℓ, memory, model, concatenatedθ)'")
    test_∇negativeloglikelihood!(datapath; maxabsdiff=maxabsdiff)
	printseparator()
    println("testing the gradient of log-posterior conditioned on all the data")
	println("	- `∇negativelogposterior(model)`")
	test_∇negativelogposterior(datapath; maxabsdiff=maxabsdiff)
	printseparator()
	println("testing hessian of the log-likelihood of all the data")
	println("	- `∇∇loglikelihood(model)`")
	test_∇∇loglikelihood(datapath; maxabsdiff=maxabsdiff)
    printseparator()
    println("testing gradient of log evidence of all the data")
	println("	- `∇logevidence(model)`")
    test_∇logevidence(datapath; maxabsdiff=maxabsdiff, simulate=false)
    printseparator()
    println("testing the hessian of the log-likelihood of only the behavioral choices")
	println("	- `∇∇choiceLL(model)`")
    test_∇∇choiceLL(datapath; maxabsdiff=maxabsdiff)
	printseparator()
	println("testing parameter initialization")
	initializeparameters!(Model(datapath))
	printseparator()
	println("testing maximum likelihood estimation")
	maximizelikelihood!(Model(datapath), Optim.LBFGS(linesearch = LineSearches.BackTracking()))
	printseparator()
	println("testing maximum a posteriori estimation")
	maximizeposterior!(Model(datapath))
	printseparator()
	println("testing evidence optimization")
	maximizeevidence!(Model(datapath))
    printseparator()
    println("testing saving model summary and predictions in `test.mat`")
	analyzeandsave(Model(datapath); prefix="test")
	printseparator()
	println("tests completed")
    return nothing
end

"""
	printseparator()

Write a standardized separator
"""
printseparator() = print("\n---------\n---------\n\n")

"""
    test_expectation_of_∇∇loglikelihood(datapath)

Check the hessian and gradient of the expectation of the log-likelihood of one neuron's GLM.

The function being checked is used in parameter initialization.
"""
function test_expectation_of_∇∇loglikelihood!(datapath::String; maxabsdiff::Real=1e-8)
    model = Model(datapath)
    mpGLM = model.trialsets[1].mpGLMs[1]
    γ = FHMDDM.randomposterior(mpGLM; rng=MersenneTwister(1234))
    x₀ = concatenateparameters(mpGLM.θ; initialization=true)
    nparameters = length(x₀)
    fhand, ghand, hhand = fill(NaN,1), fill(NaN,nparameters),
    fill(NaN,nparameters,nparameters)
    FHMDDM.expectation_of_∇∇loglikelihood!(fhand, ghand, hhand, γ, mpGLM)
    f(x) = FHMDDM.expectation_of_loglikelihood(γ, mpGLM, x; initialization=true);
    fauto = f(x₀);
    gauto = ForwardDiff.gradient(f, x₀);
    hauto = ForwardDiff.hessian(f, x₀);
	absΔQ = abs(fauto - fhand[1])
	maxabsΔ∇Q = maximum(abs.(gauto .- ghand))
	maxabsΔ∇∇Q = maximum(abs.(hauto .- hhand))
    println("   |ΔQ|: ", absΔQ)
    println("   max(|Δgradient|): ", maxabsΔ∇Q)
    println("   max(|Δhessian|): ", maxabsΔ∇∇Q)
	if (absΔQ > maxabsdiff) || (maxabsΔ∇Q > maxabsdiff) || (maxabsΔ∇∇Q > maxabsdiff)
		error("Maxmimum absolute difference exceeded")
	end
	return nothing
end

"""
    test_expectation_∇loglikelihood(datapath)

Check the gradient of the expectation of the log-likelihood of one neuron's GLM.

The function being checked is used in computing the gradient of the log-likelihood of the entire model.
"""
function test_expectation_∇loglikelihood!(datapath::String; maxabsdiff::Real=1e-8)
    model = Model(datapath)
    mpGLM = model.trialsets[1].mpGLMs[1]
    if length(mpGLM.θ.b) > 0
        while abs(mpGLM.θ.b[1]) < 1e-4
            mpGLM.θ.b[1] = 1 - 2rand()
        end
    end
    γ = FHMDDM.randomposterior(mpGLM; rng=MersenneTwister(1234))
    ∇Q = FHMDDM.GLMθ(mpGLM.θ, eltype(mpGLM.θ.𝐮))
    FHMDDM.expectation_∇loglikelihood!(∇Q, γ, mpGLM)
    ghand = FHMDDM.concatenateparameters(∇Q)
    concatenatedθ = FHMDDM.concatenateparameters(mpGLM.θ)
    f(x) = FHMDDM.expectation_of_loglikelihood(γ, mpGLM, x)
    gauto = ForwardDiff.gradient(f, concatenatedθ)
	maxabsΔ∇Q = maximum(abs.(gauto .- ghand))
    println("   max(|Δgradient|): ", maxabsΔ∇Q)
	if maxabsΔ∇Q > maxabsdiff
		error("Maxmimum absolute difference exceeded")
	end
	return nothing
end

"""
    test_∇negativeloglikelihood!(datapath)

Check the gradient of the negative of the log-likelihood of the model
"""
function test_∇negativeloglikelihood!(datapath::String; maxabsdiff::Real=1e-8)
    model = Model(datapath)
    for trialset in model.trialsets
        for mpGLM in trialset.mpGLMs
            if length(mpGLM.θ.b) > 0
                while abs(mpGLM.θ.b[1]) < 1e-4
                    mpGLM.θ.b[1] = 1 - 2rand()
                end
            end
        end
    end
    concatenatedθ, indexθ = concatenateparameters(model)
    ∇nℓ = similar(concatenatedθ)
    memory = FHMDDM.Memoryforgradient(model)
    FHMDDM.∇negativeloglikelihood!(∇nℓ, memory, model, concatenatedθ)
    f(x) = -FHMDDM.loglikelihood(x, indexθ, model)
    ℓ_auto = f(concatenatedθ)
    ∇nℓ_auto = ForwardDiff.gradient(f, concatenatedθ)
	absΔℓ = abs(ℓ_auto + memory.ℓ[1])
	maxabsΔ∇ℓ = maximum(abs.(∇nℓ_auto .- ∇nℓ))
    println("   |Δℓ|: ", absΔℓ)
    println("   max(|Δ∇ℓ|): ", maxabsΔ∇ℓ)
	if (absΔℓ > maxabsdiff) || (maxabsΔ∇ℓ > maxabsdiff)
		error("Maxmimum absolute difference exceeded")
	end
end

"""
    test_∇∇loglikelihood(datapath)

Check the computation of the hessian of the log-likelihood
"""
function test_∇∇loglikelihood(datapath::String; maxabsdiff::Real=1e-8)
    model = Model(datapath)
    for trialset in model.trialsets
        for mpGLM in trialset.mpGLMs
            if length(mpGLM.θ.b) > 0
                while abs(mpGLM.θ.b[1]) < 1e-4
                    mpGLM.θ.b[1] = 1 - 2rand()
                end
            end
        end
    end
	concatenatedθ, indexθ = FHMDDM.concatenateparameters(model)
	ℓhand, ∇hand, ∇∇hand = FHMDDM.∇∇loglikelihood(model)
	f(x) = FHMDDM.loglikelihood(x, indexθ, model)
	ℓauto = f(concatenatedθ)
	∇auto = ForwardDiff.gradient(f, concatenatedθ)
	∇∇auto = ForwardDiff.hessian(f, concatenatedθ)
	absΔℓ = abs(ℓauto-ℓhand)
	maxabsΔ∇ℓ = maximum(abs.(∇auto .- ∇hand))
	maxabsΔ∇∇ℓ = maximum(abs.(∇∇auto .- ∇∇hand))
	println("   |Δℓ|: ", absΔℓ)
    println("   max(|Δ∇ℓ|): ", maxabsΔ∇ℓ)
    println("   max(|Δ∇∇ℓ|): ", maxabsΔ∇∇ℓ)
	if (absΔℓ > maxabsdiff) || (maxabsΔ∇ℓ > maxabsdiff) || (maxabsΔ∇∇ℓ > maxabsdiff)
		error("Maxmimum absolute difference exceeded")
	end
end

"""
	test_∇negativelogposterior(model)

Compare the hand-computed and automatically-differentiated gradients

ARGUMENT
-`model`: a structure containing the data, parameters, and hyperparameters of a factorial hidden-Markov drift-diffusion model
```
"""
function test_∇negativelogposterior(datapath::String; maxabsdiff::Real=1e-8)
	model = Model(datapath)
	for trialset in model.trialsets
        for mpGLM in trialset.mpGLMs
            if length(mpGLM.θ.b) > 0
                while abs(mpGLM.θ.b[1]) < 1e-4
                    mpGLM.θ.b[1] = 1 - 2rand()
                end
            end
        end
    end
	concatenatedθ, indexθ = FHMDDM.concatenateparameters(model)
	memory = FHMDDM.Memoryforgradient(model)
	ℓhand = FHMDDM.logposterior!(model, memory, concatenatedθ)
	∇hand = similar(concatenatedθ)
	FHMDDM.∇negativelogposterior!(∇hand, model, memory, concatenatedθ)
	f(x) = FHMDDM.logposterior(x, indexθ, model)
	ℓauto = f(concatenatedθ)
	∇auto = ForwardDiff.gradient(f, concatenatedθ)
	absΔ𝐿 = abs(ℓauto-ℓhand)
	maxabsΔ∇𝐿 = maximum(abs.(∇auto .+ ∇hand))
	println("   |Δ(𝐿)|: ", absΔ𝐿)
    println("   max(|Δ(∇𝐿)|): ", maxabsΔ∇𝐿)
	if (absΔ𝐿 > maxabsdiff) || (maxabsΔ∇𝐿 > maxabsdiff)
		error("Maxmimum absolute difference exceeded")
	end
end

"""
	test_∇logevidence(model)

Check whether the hand-coded gradient of the log-evidence matches the automatic gradient

ARGUMENT
-`datapath`: path to the file containing the data and model settings

OPTIONAL ARGUMENT
-`simulate`: whether to simulate Hessian and MAP solution. If not, the model is first fitted before a Hessian is computed

RETURN
-maximum absolute normalized difference between the gradients
-maximum absolute normalized difference between the log-evidence functions
"""
function test_∇logevidence(datapath::String; maxabsdiff::Real=1e-8, simulate::Bool=false)
    model = Model(datapath)
	@unpack 𝛂, index𝐀, index𝚽, 𝚽 = model.gaussianprior
	𝛉, index𝛉 = FHMDDM.concatenateparameters(model)
	𝐱 = 1 .- 2rand(length(𝛂))
	FHMDDM.real2native!(model.gaussianprior, 𝐱)
	FHMDDM.precisionmatrix!(model.gaussianprior)
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
		FHMDDM.initializeparameters!(model; show_trace=false)
		FHMDDM.maximizeposterior!(model; show_trace=false);
		𝐇 = FHMDDM.∇∇loglikelihood(model)[3][index𝚽, index𝚽]
		𝐰₀ = FHMDDM.concatenateparameters(model)[1][index𝚽]
		𝐁₀𝐰₀ = (𝚽-𝐇)*𝐰₀
		𝐱 = 1 .- 2rand(length(𝛂)) # only if we are not simulating because changing the hyperparameters might make the Hessian of the posterior not positive-definite
	end
	memory = FHMDDM.Memoryforgradient(model)
	handcoded_evidence = FHMDDM.logevidence!(memory, model, 𝛉, 𝐁₀𝐰₀, 𝐇, 𝐱)
	handcoded_gradient = fill(NaN,length(𝐱))
	∇nℓ = similar(𝛉)
	FHMDDM.∇negativelogevidence!(memory, model, handcoded_gradient, ∇nℓ, 𝛉, 𝐁₀𝐰₀, 𝐇, 𝐱)
    f(x) = FHMDDM.logevidence(𝐁₀𝐰₀, 𝐇, model, x)
	automatic_evidence = f(𝐱)
	automatic_gradient = ForwardDiff.gradient(f, 𝐱)
	absdiff∇ = abs.((automatic_gradient .+ handcoded_gradient))
	absreldiff∇ = abs.((automatic_gradient .+ handcoded_gradient))./automatic_gradient
	absΔ𝐸 = abs(automatic_evidence-handcoded_evidence)
	maxabsΔ∇𝐸 = maximum(min.(absdiff∇, absreldiff∇))
    println("   |Δ𝐸|: ", absΔ𝐸)
    println("   max(|Δ∇𝐸|): ", maxabsΔ∇𝐸)
	if (absΔ𝐸 > maxabsdiff) || (maxabsΔ∇𝐸 > maxabsdiff)
		error("Maxmimum absolute difference exceeded")
	end
end

"""
	test_∇∇choiceLL(model)

Compare the automatically computed and hand-coded gradients and hessians with respect to the parameters being fitted in their real space

ARGUMENT
-`datapath`: path to the file containing the data and model settings

RETURN
-`absdiffℓ`: absolute difference in the log-likelihood evaluted using the algorithm bein automatically differentiated and the hand-coded algorithm
-`absdiff∇`: absolute difference in the gradients
-`absdiff∇∇`: absolute difference in the hessians
```
"""
function test_∇∇choiceLL(datapath::String; maxabsdiff::Real=1e-8)
	model = Model(datapath)
	ℓhand, ∇hand, ∇∇hand = FHMDDM.∇∇choiceLL(model)
	concatenatedθ, indexθ = FHMDDM.concatenate_choice_related_parameters(model)
	if isempty(concatenatedθ)
		println("   drift-diffusion parameters are not being fitted")
	else
		index𝛂 = FHMDDM.choice_related_precisions(model)[2]
		f(x) = FHMDDM.choiceLL(x, indexθ.latentθ, model)
		ℓauto = f(concatenatedθ)
		∇auto = ForwardDiff.gradient(f, concatenatedθ)
		∇∇auto = ForwardDiff.hessian(f, concatenatedθ)
		absΔℓ = abs(ℓauto-ℓhand)
		maxabsΔ∇ℓ = maximum(abs.(∇auto .- ∇hand[index𝛂]))
		maxabsΔ∇∇ℓ = maximum(abs.(∇∇auto .- ∇∇hand[index𝛂,index𝛂]))
		println("   |Δ(ℓ)|: ", absΔℓ)
	    println("   max(|Δ(∇ℓ)|): ", maxabsΔ∇ℓ)
	    println("   max(|Δ(∇∇ℓ)|): ", maxabsΔ∇∇ℓ)
		if (absΔℓ > maxabsdiff) || (maxabsΔ∇ℓ > maxabsdiff) || (maxabsΔ∇∇ℓ > maxabsdiff)
			error("Maxmimum absolute difference exceeded")
		end
	end
end
