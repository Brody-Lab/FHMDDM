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
    println("---------")
    println("testing gradient of log evidence")
    model = Model(datapath)
    max_abs_norm_diff_∇𝐸, abs_norm_diff_𝐸 = FHMDDM.check_∇logevidence(model; simulate=false)
    println("   max(|Δ𝐸|): ", abs_norm_diff_𝐸)
    println("   max(|Δ∇𝐸|): ", max_abs_norm_diff_∇𝐸)
    println("---------")
    println("testing parameter learning")
    model = Model(datapath)
    learnparameters!(model)
    println(" ")
    println("---------")
    println("testing saving model parameters, hessian, and predictions in `test.mat`")
    ∇∇ℓ = ∇∇loglikelihood(model)[3]
    λΔt, pchoice = expectedemissions(model)
    save(∇∇ℓ, λΔt, model, pchoice; filename="test.mat")
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
