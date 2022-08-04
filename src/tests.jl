"""
    test(datapath)

Run a number of test on the model

ARGUMENT
-`datapath`: full path of the data file
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
    println("testing saving model as `test.mat`")
    model = Model(datapath)
    save(model; filename="test.mat")
    return nothing
end

"""
"""
function test_expectation_of_∇∇loglikelihood!(datapath::String)
    model = Model(datapath)
    mpGLM = model.trialsets[1].mpGLMs[1]
    γ = randomposterior(mpGLM; rng=MersenneTwister(1234))
    x₀ = concatenateparameters(mpGLM.θ)
    nparameters = length(x₀)
    fhand, ghand, hhand = fill(NaN,1), fill(NaN,nparameters),
    fill(NaN,nparameters,nparameters)
    expectation_of_∇∇loglikelihood!(fhand, ghand, hhand, γ, mpGLM)
    f(x) = expectation_of_loglikelihood(γ, mpGLM, x);
    fauto = f(x₀);
    gauto = ForwardDiff.gradient(f, x₀);
    hauto = ForwardDiff.hessian(f, x₀);
    println("   |ΔQ|: ", abs(fauto - fhand[1]))
    println("   max(|Δgradient|): ", maximum(abs.(gauto .- ghand)))
    println("   max(|Δhessian|): ", maximum(abs.(hauto .- hhand)))
end

"""
"""
function test_expectation_∇loglikelihood!(datapath::String)
    model = Model(datapath)
    mpGLM = model.trialsets[1].mpGLMs[1]
    γ = randomposterior(mpGLM; rng=MersenneTwister(1234))
    ∇Q = GLMθ(mpGLM.θ, eltype(mpGLM.θ.𝐮))
    expectation_∇loglikelihood!(∇Q, γ, mpGLM)
    ghand = concatenateparameters(∇Q)
    concatenatedθ = concatenateparameters(mpGLM.θ)
    f(x) = expectation_of_loglikelihood(γ, mpGLM, x)
    gauto = ForwardDiff.gradient(f, concatenatedθ)
    println("   max(|Δgradient|): ", maximum(abs.(gauto .- ghand)))
end

"""
    test_∇negativeloglikelihood!(datapath)

Check the gradient of the negative of the log-likelihood of the model
"""
function test_∇negativeloglikelihood!(datapath::String)
    model = Model(datapath)
    concatenatedθ, indexθ = concatenateparameters(model)
    ∇nℓ = similar(concatenatedθ)
    memory = Memoryforgradient(model)
    ∇negativeloglikelihood!(∇nℓ, memory, model, concatenatedθ)
    f(x) = -loglikelihood(x, indexθ, model)
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
    absdiffℓ, absdiff∇, absdiff∇∇ = check_∇∇loglikelihood(model)
    println("   max(|Δloss|): ", absdiffℓ)
    println("   max(|Δgradient|): ", maximum(absdiff∇))
    println("   max(|Δhessian|): ", maximum(absdiff∇∇))
end
