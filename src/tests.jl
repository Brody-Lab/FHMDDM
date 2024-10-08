"""
    test(csvpath)

Run a number of tests on the model

ARGUMENT
-`csvpath`: full path of the data file
"""
function test(csvpath::String, row::Integer; maxabsdiff::Real=1e-8)
	options = Options(csvpath, row)
	println("testing `"*options.datapath*"`")
    println("testing the hessian of the expectation of the log-likelihood of one neuron's spike train")
    println("	- `expectation_of_∇∇loglikelihood(γ, mpGLM, x)`")
    println("	- used for parameter initialization")
    test_expectation_of_∇∇loglikelihood!(csvpath, row; maxabsdiff=maxabsdiff)
    printseparator()
    println("testing the gradient of the expectation of the log-likelihood of one neuron's spike train")
    println("	- 'expectation_∇loglikelihood!(∇Q, γ, mpGLM)'")
    println("	- used for learning of the parameters of the full model")
    test_expectation_∇loglikelihood!(csvpath, row; maxabsdiff=maxabsdiff)
    printseparator()
    println("testing the gradient of log-likelihood of all the data")
    println("	- '∇negativeloglikelihood!(∇nℓ, memory, model, concatenatedθ)'")
    test_∇negativeloglikelihood!(csvpath, row; maxabsdiff=maxabsdiff)
	printseparator()
    println("testing the gradient of log-posterior conditioned on all the data")
	println("	- `∇negativelogposterior(model)`")
	test_∇negativelogposterior(csvpath, row; maxabsdiff=maxabsdiff)
	printseparator()
	println("testing hessian of the log-likelihood of all the data")
	println("	- `∇∇loglikelihood(model)`")
	test_∇∇loglikelihood(csvpath, row; maxabsdiff=maxabsdiff)
	printseparator()
    println("testing the hessian of the log-likelihood of only the behavioral choices")
	println("	- `∇∇choiceLL(model)`")
    test_∇∇choiceLL(csvpath, row; maxabsdiff=maxabsdiff)
	printseparator()
	println("saving model summary")
	model = Model(csvpath, row)
	testfolderpath = joinpath(model.options.outputpath, "test")
	save(ModelSummary(model), testfolderpath)
	printseparator()
	println("loading model parameters from a saved summary")
	model = Model(csvpath, row)
	sortparameters!(model, joinpath(testfolderpath, "modelsummary.mat"))
	printseparator()
	println("testing maximum likelihood estimation")
	model = Model(csvpath, row)
	initializeparameters!(model)
	maximizelikelihood!(model, Optim.LBFGS(linesearch = LineSearches.BackTracking()); iterations=50)
	printseparator()
	println("testing maximum a posteriori estimation")
	model = Model(csvpath, row)
	initializeparameters!(model)
	maximizeposterior!(model)
	if !model.options.fit_overdispersion
		printseparator()
		println("testing gradient of log evidence of all the data")
		println("	- `∇logevidence(model)`")
		test_∇logevidence(csvpath, row; maxabsdiff=maxabsdiff, simulate=false)
		printseparator()
		println("testing evidence optimization")
		model = Model(csvpath, row)
		initializeparameters!(model)
		maximizeevidence!(model)
	end
    printseparator()
	model = Model(csvpath, row)
	printseparator()
	println("saving model characterization")
	characterization = Characterization(model)
	save(characterization, testfolderpath)
	printseparator()
	println("post-stimulus time histograms")
	psthsets = poststereoclick_time_histogram_sets(characterization.expectedemissions, model)
	save(psthsets, testfolderpath)
	printseparator()
	println("simulating model")
	samplepaths = simulateandsave(Model(csvpath, row), 2)
	printseparator()
	println("loading simulated observations")
	simulation = Model(samplepaths[1])
    printseparator()
	println("testing cross-validation and saving results")
	model = Model(csvpath, row)
	cvfolderpath = joinpath(model.options.outputpath, "cvtest")
	cvresults = crossvalidate(2, model)
	save(cvresults, cvfolderpath)
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
    test_expectation_of_∇∇loglikelihood(csvpath, row)

Check the hessian and gradient of the expectation of the log-likelihood of one neuron's GLM.

The function being checked is used in parameter initialization.
"""
function test_expectation_of_∇∇loglikelihood!(csvpath::String, row::Integer; maxabsdiff::Real=1e-8)
    model = Model(csvpath, row)
    mpGLM = model.trialsets[1].mpGLMs[1]
    γ = FHMDDM.randomposterior(mpGLM; rng=MersenneTwister(1234))
    x₀ = concatenateparameters(mpGLM.θ; initialization=true)
    nparameters = length(x₀)
    fhand, ghand, hhand = fill(NaN,1), fill(NaN,nparameters),
    fill(NaN,nparameters,nparameters)
	glmderivatives = FHMDDM.GLMDerivatives(mpGLM)
    expectation_of_∇∇loglikelihood!(glmderivatives, fhand, ghand, hhand, γ, mpGLM)
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
    test_expectation_∇loglikelihood(csvpath, row)

Check the gradient of the expectation of the log-likelihood of one neuron's GLM.

The function being checked is used in computing the gradient of the log-likelihood of the entire model.
"""
function test_expectation_∇loglikelihood!(csvpath::String, row::Integer; maxabsdiff::Real=1e-8)
    model = Model(csvpath, row)
    mpGLM = model.trialsets[1].mpGLMs[1]
    if length(mpGLM.θ.b) > 0
        while abs(mpGLM.θ.b[1]) < 1e-4
            mpGLM.θ.b[1] = 1 - 2rand()
        end
    end
    γ = FHMDDM.randomposterior(mpGLM; rng=MersenneTwister(1234))
    ∇Q = FHMDDM.GLMθ(eltype(mpGLM.θ.𝐮), mpGLM.θ)
	glmderivatives = FHMDDM.GLMDerivatives(mpGLM)
    FHMDDM.expectation_∇loglikelihood!(∇Q, glmderivatives, γ, mpGLM)
    ghand = concatenateparameters(∇Q)
    concatenatedθ = concatenateparameters(mpGLM.θ)
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
    test_∇negativeloglikelihood!(csvpath, row)

Check the gradient of the negative of the log-likelihood of the model
"""
function test_∇negativeloglikelihood!(csvpath::String, row::Integer; maxabsdiff::Real=1e-8)
    model = Model(csvpath, row)
    for trialset in model.trialsets
        for mpGLM in trialset.mpGLMs
            if length(mpGLM.θ.b) > 0
                while abs(mpGLM.θ.b[1]) < 1e-4
                    mpGLM.θ.b[1] = 1 - 2rand()
                end
            end
        end
    end
    concatenatedθ = concatenateparameters(model)
	indexθ = indexparameters(model)
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
    test_∇∇loglikelihood(csvpath, row)

Check the computation of the hessian of the log-likelihood
"""
function test_∇∇loglikelihood(csvpath::String, row::Integer; maxabsdiff::Real=1e-8)
    model = Model(csvpath, row)
    for trialset in model.trialsets
        for mpGLM in trialset.mpGLMs
            if length(mpGLM.θ.b) > 0
                while abs(mpGLM.θ.b[1]) < 1e-4
                    mpGLM.θ.b[1] = 1 - 2rand()
                end
            end
        end
    end
	concatenatedθ = concatenateparameters(model)
	indexθ = indexparameters(model)
	ℓhand, ∇hand, ∇∇hand = FHMDDM.∇∇loglikelihood(model)
	f(x) = FHMDDM.loglikelihood(x, indexθ, model)
	ℓauto = f(concatenatedθ)
	∇auto = ForwardDiff.gradient(f, concatenatedθ)
	∇∇auto = ForwardDiff.hessian(f, concatenatedθ)
	absΔℓ = abs(ℓauto-ℓhand)
	maxabsΔ∇ℓ = maximum(abs.(∇auto .- ∇hand))
	absΔ∇∇ℓ = abs.(∇∇auto .- ∇∇hand)
	maxabsΔ∇∇ℓ = maximum(min.(absΔ∇∇ℓ, absΔ∇∇ℓ./abs.(∇∇hand)))
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
function test_∇negativelogposterior(csvpath::String, row::Integer; maxabsdiff::Real=1e-8)
	model = Model(csvpath, row)
	for trialset in model.trialsets
        for mpGLM in trialset.mpGLMs
            if length(mpGLM.θ.b) > 0
                while abs(mpGLM.θ.b[1]) < 1e-4
                    mpGLM.θ.b[1] = 1 - 2rand()
                end
            end
        end
    end
	concatenatedθ = concatenateparameters(model)
	indexθ = indexparameters(model)
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
-`csvpath`: absolute path of the CSV file
-`row`: row of the CSV file containing the fixed hyperparameters of this model

OPTIONAL ARGUMENT
-`simulate`: whether to simulate Hessian and MAP solution. If not, the model is first fitted before a Hessian is computed

RETURN
-maximum absolute normalized difference between the gradients
-maximum absolute normalized difference between the log-evidence functions
"""
function test_∇logevidence(csvpath::String, row::Integer; maxabsdiff::Real=1e-8, simulate::Bool=false)
    model = Model(csvpath, row)
	@unpack 𝛂, index𝐀, index𝚽, 𝚽 = model.gaussianprior
	𝛉 = concatenateparameters(model)
	index𝛉 = indexparameters(model)
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
		𝐰₀ = concatenateparameters(model)[index𝚽]
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
-`csvpath`: absolute path of the CSV file
-`row`: row of the CSV file containing the fixed hyperparameters of this model

RETURN
-`absdiffℓ`: absolute difference in the log-likelihood evaluted using the algorithm bein automatically differentiated and the hand-coded algorithm
-`absdiff∇`: absolute difference in the gradients
-`absdiff∇∇`: absolute difference in the hessians
```
"""
function test_∇∇choiceLL(csvpath::String, row::Integer; maxabsdiff::Real=1e-8)
	model = Model(csvpath, row)
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
