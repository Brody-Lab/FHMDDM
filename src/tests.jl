"""
    test(datapath)

Run a number of test on the model

ARGUMENT
-`datapath`: full path of the data file

EXAMPLE
```julia-repl
julia> using FHMDDM
julia> datapath = "/mnt/cup/labs/brody/tzluo/analysis_data/analysis_2022_08_08c_test/T176_2018_05_03/data.mat"
julia> test(datapath)
julia>
```
"""
function test(datapath::String)
    println(" ")
    println("---------")
    println("testing `expectation_of_∇∇loglikelihood(γ, mpGLM, x)`")
    test_expectation_of_∇∇loglikelihood!(datapath)
    println(" ")
    println("---------")
    println("testing 'expectation_∇loglikelihood!(∇Q, γ, mpGLM)'")
    test_expectation_∇loglikelihood!(datapath)
    println(" ")
    println("---------")
    println("testing '∇negativeloglikelihood!(∇nℓ, memory, model, concatenatedθ)'")
    test_∇negativeloglikelihood!(datapath)
    println(" ")
    println("---------")
    println("testing hessian of the log-likelihood of the model")
    test_∇∇loglikelihood(datapath)
    println(" ")
    println("---------")
    println("testing gradient of log evidence")
    max_abs_norm_diff_∇𝐸, abs_norm_diff_𝐸 = FHMDDM.check_∇logevidence(datapath; simulate=false)
    println("   max(|Δ𝐸|): ", abs_norm_diff_𝐸)
    println("   max(|Δ∇𝐸|): ", max_abs_norm_diff_∇𝐸)
    println(" ")
    println("---------")
    println("testing the hessian and gradient of the log-likelihood of choices")
    absdiffℓ, absdiff∇, absdiff∇∇ = FHMDDM.check_∇∇choiceLL(datapath)
    println("   max(|Δloss|): ", absdiffℓ)
    println("   max(|Δgradient|): ", maximum(absdiff∇))
    println("   max(|Δhessian|): ", maximum(absdiff∇∇))
    println(" ")
    println("---------")
    println("testing parameter learning")
    model = Model(datapath)
    learnparameters!(model)
    println(" ")
    println("---------")
    println("testing saving model parameters, hessian, and predictions in `test.mat`")
	analyzeandsave(model; prefix="test.mat")
    return nothing
end

"""
    test_expectation_of_∇∇loglikelihood(datapath)

Check the hessian and gradient of the expectation of the log-likelihood of one neuron's GLM.

The function being checked is used in parameter initialization.
"""
function test_expectation_of_∇∇loglikelihood!(datapath::String)
    model = Model(datapath)
    mpGLM = model.trialsets[1].mpGLMs[1]
    γ = FHMDDM.randomposterior(mpGLM; rng=MersenneTwister(1234))
    x₀ = concatenateparameters(mpGLM.θ; omitb=true)
    nparameters = length(x₀)
    fhand, ghand, hhand = fill(NaN,1), fill(NaN,nparameters),
    fill(NaN,nparameters,nparameters)
    FHMDDM.expectation_of_∇∇loglikelihood!(fhand, ghand, hhand, γ, mpGLM)
    f(x) = FHMDDM.expectation_of_loglikelihood(γ, mpGLM, x; omitb=true);
    fauto = f(x₀);
    gauto = ForwardDiff.gradient(f, x₀);
    hauto = ForwardDiff.hessian(f, x₀);
    println("   |ΔQ|: ", abs(fauto - fhand[1]))
    println("   max(|Δgradient|): ", maximum(abs.(gauto .- ghand)))
    println("   max(|Δhessian|): ", maximum(abs.(hauto .- hhand)))
end

"""
    test_expectation_∇loglikelihood(datapath)

Check the gradient of the expectation of the log-likelihood of one neuron's GLM.

The function being checked is used in computing the gradient of the log-likelihood of the entire model.
"""
function test_expectation_∇loglikelihood!(datapath::String)
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
    println("   max(|Δgradient|): ", maximum(abs.(gauto .- ghand)))
end

"""
    test_∇negativeloglikelihood!(datapath)

Check the gradient of the negative of the log-likelihood of the model
"""
function test_∇negativeloglikelihood!(datapath::String)
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
    println("   max(|Δloss|): ", abs(ℓ_auto + memory.ℓ[1]))
    println("   max(|Δgradient|): ", maximum(abs.(∇nℓ_auto .- ∇nℓ)))
end

"""
    test_∇∇loglikelihood(datapath)

Check the computation of the hessian of the log-likelihood
"""
function test_∇∇loglikelihood(datapath::String)
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
    absdiffℓ, absdiff∇, absdiff∇∇ = FHMDDM.check_∇∇loglikelihood(model)
    println("   max(|Δloss|): ", absdiffℓ)
    println("   max(|Δgradient|): ", maximum(absdiff∇))
    println("   max(|Δhessian|): ", maximum(absdiff∇∇))
end

"""
	check_∇∇loglikelihood(model)

Compare the automatically computed and hand-coded gradients and hessians with respect to the parameters being fitted in their real space

ARGUMENT
-`model`: a structure containing the data, parameters, and hyperparameters of a factorial hidden-Markov drift-diffusion model

RETURN
-`absdiffℓ`: absolute difference in the log-likelihood evaluted using the algorithm bein automatically differentiated and the hand-coded algorithm
-`absdiff∇`: absolute difference in the gradients
-`absdiff∇∇`: absolute difference in the hessians
"""
function check_∇∇loglikelihood(model::Model)
	concatenatedθ, indexθ = FHMDDM.concatenateparameters(model)
	ℓhand, ∇hand, ∇∇hand = FHMDDM.∇∇loglikelihood(model)
	f(x) = FHMDDM.loglikelihood(x, indexθ, model)
	ℓauto = f(concatenatedθ)
	∇auto = ForwardDiff.gradient(f, concatenatedθ)
	∇∇auto = ForwardDiff.hessian(f, concatenatedθ)
	return abs(ℓauto-ℓhand), abs.(∇auto .- ∇hand), abs.(∇∇auto .- ∇∇hand)
end

"""
	check_∇logevidence(model)

Check whether the hand-coded gradient of the log-evidence matches the automatic gradient

ARGUMENT
-`datapath`: path to the file containing the data and model settings

OPTIONAL ARGUMENT
-`simulate`: whether to simulate Hessian and MAP solution. If not, the model is first fitted before a Hessian is computed

RETURN
-maximum absolute normalized difference between the gradients
-absolute normalized difference between the log-evidence functions
"""
function check_∇logevidence(datapath::String; simulate::Bool=false)
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
		FHMDDM.initializeparameters!(model; show_trace=false, verbose=false)
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
	return maximum(min.(absdiff∇, absreldiff∇)), abs((automatic_evidence-handcoded_evidence))
end

"""
	check_∇∇choiceLL(model)

Compare the automatically computed and hand-coded gradients and hessians with respect to the parameters being fitted in their real space

ARGUMENT
-`datapath`: path to the file containing the data and model settings

RETURN
-`absdiffℓ`: absolute difference in the log-likelihood evaluted using the algorithm bein automatically differentiated and the hand-coded algorithm
-`absdiff∇`: absolute difference in the gradients
-`absdiff∇∇`: absolute difference in the hessians
```
"""
function check_∇∇choiceLL(datapath::String)
	model = Model(datapath)
	concatenatedθ, indexθ = FHMDDM.concatenate_choice_related_parameters(model)
	ℓhand, ∇hand, ∇∇hand = FHMDDM.∇∇choiceLL(model)
	index𝛂 = FHMDDM.choice_related_precisions(model)[2]
	f(x) = FHMDDM.choiceLL(x, indexθ.latentθ, model)
	ℓauto = f(concatenatedθ)
	∇auto = ForwardDiff.gradient(f, concatenatedθ)
	∇∇auto = ForwardDiff.hessian(f, concatenatedθ)
	return abs(ℓauto-ℓhand), abs.(∇auto .- ∇hand[index𝛂]), abs.(∇∇auto .- ∇∇hand[index𝛂,index𝛂])
end
