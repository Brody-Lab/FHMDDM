"""
    learn_𝛃_from_sample(model; niterations, tolerance, startfromgenerative)

Learn of the GLM filters 𝛃 from a sample of the factorial hidden-Markov drift-diffusion model

During learning, other parameters of the model are fixed to the values used to generate the sample of the results

ARGUMENT
-`generativemodel`: an existing model whose data provide the organizational basis of the simulated data, whose options are copied, and whose parameters are used as the generative parameters

OPTIONAL ARGUMENT
-`niterations`: number of iterations of the expectation-maximization algorithm
-`tolerance`: the smallest change in the objective value of the expectation-maximization algorithm before convergence is considered to be successful
-`startfromgenerative`: whether the initial values of the learning process is set the values used in model sampling. This is useful for testing whether a local minimum might be an issue
"""
function learn_𝛃_from_sample(generativemodel::FHMDDM;
                            niterations::Integer=100,
                            startfromgenerative::Bool=false,
                            tolerance::AbstractFloat = 1e-9)
    model = sample(generativemodel)
    @unpack data, mpGLM, options, θ = model
    @unpack Aᶻ, 𝛃, πᶻ, ψ, θₐ = θ
    if !startfromgenerative
        for i in eachindex(mpGLM)
            for n in eachindex(mpGLM[i])
                mpGLM[i][n].𝐮 .= 10 .- 20rand(size(mpGLM[i][n].𝐔,2))
                for k in eachindex(mpGLM[i][n].𝐯)
                    mpGLM[i][n].𝐯[k] .= 10 .- 20rand(size(mpGLM[i][n].𝐕[1],2))
                end
                𝛃[i][n] .= vcat(mpGLM[i][n].𝐮, collect(Iterators.flatten(mpGLM[i][n].𝐯)))
                options.θ₀.𝛃[i][n] .= 𝛃[i][n]
            end
        end
    end
    quantities = EMquantities(data, options)
    @unpack γ = quantities
    𝒬_𝛃_previous = 0.0
    for j = 1:niterations
        E_step!(quantities, data, mpGLM, options, Aᶻ, θₐ, πᶻ, ψ)
        estimatefilters!(mpGLM, γ)
        𝒬_𝛃 = expectation(mpGLM, γ)
        println("Iteration ", j)
        println("𝒬_𝛃 = ", 𝒬_𝛃)
        println("𝛃[1][1] = ", regression_coefficients(mpGLM)[1][1])
        if j > 1 && abs(𝒬_𝛃 - 𝒬_𝛃_previous)/(1+abs(𝒬_𝛃_previous)) < tolerance
            break
        else
            𝒬_𝛃_previous = 𝒬_𝛃
        end
    end
    return model
end

"""
    learn_θₐ_from_sample(model; niterations, startfromgenerative, tolerance)

Learn of the parameters controlling the drift-diffusion dynamics (θₐ) from a sample of the factorial hidden-Markov drift-diffusion model

During learning, other parameters of the model are fixed to the values used to generate the sample of the results

ARGUMENT
-`generativemodel`: an existing model whose data provide the organizational basis of the simulated data, whose options are copied, and whose parameters are used as the generative parameters

OPTIONAL ARGUMENT
-`compute_posterior_with_generative`: whether to compute the posterior probabilities using the parameter values used to generate the data
-`niterations`: number of iterations of the expectation-maximization algorithm
-`startfromgenerative`: whether the initial values of the learning process is set the values used in model sampling. This is useful for testing whether a local minimum might be an issue
-`tolerance`: the smallest change in the objective value of the expectation-maximization algorithm before convergence is considered to be successful
"""
function learn_θₐ_from_sample(generativemodel::FHMDDM;
                              compute_posterior_with_generative::Bool=false,
                              niterations::Integer=100,
                              startfromgenerative::Bool=false,
                              tolerance::AbstractFloat = 1e-9)
    model = sample(generativemodel)
    if !startfromgenerative
        θₐ₀ = Θₐ(model.options.θ₀.θₐ,
                 model.options.θₐisfit,
                 model.options.θₐlowerbound,
                 model.options.θₐupperbound)
        θ₀ = θFHMDDM(Aᶻ=model.θ.Aᶻ,
                     𝛃=model.θ.𝛃,
                     πᶻ=model.θ.πᶻ,
                     ψ=model.θ.ψ,
                     θₐ=θₐ₀)
    else
        θ₀ = model.θ
    end
    optionsdict = Dict(model.options)
    if haskey(optionsdict, "theta_0")
        optionsdict["theta_0"] = Dict(θ₀)
    else
        optionsdict = merge(optionsdict, Dict("theta_0" => Dict(θ₀)))
    end
    model = FHMDDM(data=model.data,
                   mpGLM=model.mpGLM,
                   options=FHMDDMoptions(optionsdict),
                   θ=θ₀)
    @unpack data, mpGLM, options, θ = model
    @unpack Aᶻ, 𝛃, πᶻ, ψ, θₐ = θ
    quantities = EMquantities(data, options)
    @unpack Aᵃ, χᵃ, χᵃnb, χᶻ, fb, γ, γᵃ, γᶻ, p𝐘𝑑 = quantities
    ℓprevious = NaN
    println("Initial θₐ = ", θₐ)
    for j = 1:niterations
        if compute_posterior_with_generative
            E_step!(quantities, data, mpGLM, options, Aᶻ, generativemodel.θ.θₐ, πᶻ, ψ)
        else
            E_step!(quantities, data, mpGLM, options, Aᶻ, θₐ, πᶻ, ψ)
        end
        θₐ = estimateθₐ(χᵃnb, data, γᵃ, options, θₐ; minAᵃ=eps())
        ℓ = loglikelihood!(quantities, model, ψ, θₐ)
        println("Iteration ", j, ": log-likelihood = ", ℓ)
        println("θₐ = ", θₐ)
        if j > 1 &&
           (ℓ - ℓprevious)/abs(ℓprevious) < tolerance # stops if the log-likelihood is no longer being maximized
           if ℓ < ℓprevious
               println("The log-likelihood decreased from the previous iteration!")
           end
            break
        else
            ℓprevious = ℓ
        end
    end
    θ = θFHMDDM(Aᶻ=Aᶻ, 𝛃 =𝛃, πᶻ=πᶻ, ψ =ψ, θₐ=θₐ)
    FHMDDM(data=data, mpGLM=mpGLM, options=options, θ=θ)
end

"""
    learn_𝛃_θₐ_from_sample(generativemodel; niterations, startfromgenerative, tolerance)


Learn the linear filters (𝛃) and the parameters controlling the drift-diffusion dynamics (θₐ) from a sample of the factorial hidden-Markov drift-diffusion model

During learning, other parameters of the model are fixed to the values used to generate the sample of the results

ARGUMENT
-`generativemodel`: an existing model whose data provide the organizational basis of the simulated data, whose options are copied, and whose parameters are used as the generative parameters

OPTIONAL ARGUMENT
-`niterations`: number of iterations of the expectation-maximization algorithm
-`startfromgenerative`: whether the initial values of the learning process is set the values used in model sampling. This is useful for testing whether a local minimum might be an issue
-`tolerance`: the smallest change in the objective value of the expectation-maximization algorithm before convergence is considered to be successful
"""
function learn_𝛃_θₐ_from_sample(generativemodel::FHMDDM;
                              niterations::Integer=100,
                              startfromgenerative::Bool=false,
                              tolerance::AbstractFloat = 1e-9)
    model = initialize_𝛃_θₐ(sample(generativemodel))
    @unpack data, mpGLM, options, θ = model
    @unpack Aᶻ, 𝛃, πᶻ, ψ, θₐ = θ
    quantities = EMquantities(data, options)
    @unpack γᵃ, χᵃnb, γ = quantities
    𝒬previous = 0.
    𝒬 = 0.
    println("Initial θₐ = ", θₐ)
    println("Initial 𝛃[1][1] = ", regression_coefficients(mpGLM)[1][1])
    for j = 1:niterations
        E_step!(quantities, data, mpGLM, options, Aᶻ, θₐ, πᶻ, ψ)
        estimatefilters!(mpGLM, γ)
        𝒬_𝛃 = expectation(mpGLM, γ)
        θₐ, 𝒬_θₐ = estimateθₐ(χᵃnb, data, γᵃ, options, θₐ)
        𝒬 = 𝒬_𝛃 + 𝒬_θₐ
        println("Iteration ", j, ": 𝒬 = ", 𝒬)
        println("θₐ = ", θₐ)
        println("𝛃[1][1] = ", regression_coefficients(mpGLM)[1][1])
        if j > 1 && abs(𝒬 - 𝒬previous)/(1+abs(𝒬previous)) < tolerance
            break
        else
            𝒬previous = 𝒬
        end
    end
    θ = θFHMDDM(Aᶻ=Aᶻ, 𝛃=regression_coefficients(mpGLM), πᶻ=πᶻ, ψ =ψ, θₐ=θₐ)
    FHMDDM(data=data, mpGLM=mpGLM, options=options, θ=θ)
end

"""
    initialize_𝛃_θₐ(model; startfromgenerative)

Initialize the linear filters (𝛃) and the parameters controlling the drift-diffusion dynamics (θₐ) and document them in the model's options

ARGUMENT
-`model`: an instance of the factorial hidden Markov drift-diffusion model

OPTIONAL ARGUMENT
-`startfromgenerative`: whether the initial values of the learning process is set the values used in model sampling. This is useful for testing whether a local minimum might be an issue

RETURN
-an instance of the factorial hidden Markov drift-diffusion model with the parameters in 𝛃 and θₐ initialized
"""
function initialize_𝛃_θₐ(model::FHMDDM; startfromgenerative::Bool=false)
    @unpack data, mpGLM, options, θ = model
    @unpack Aᶻ, 𝛃, πᶻ, ψ, θₐ = θ
    if !startfromgenerative
        θₐ₀ = Θₐ(options.θ₀.θₐ,
                 options.θₐisfit,
                 options.θₐlowerbound,
                 options.θₐupperbound)
        for i in eachindex(mpGLM)
            for n in eachindex(mpGLM[i])
                mpGLM[i][n].𝐮 .= 10 .- 20rand(size(mpGLM[i][n].𝐔,2))
                    for k in eachindex(mpGLM[i][n].𝐯)
                        mpGLM[i][n].𝐯[k] .= 10 .- 20rand(size(mpGLM[i][n].𝐕[1],2))
                    end
                𝛃[i][n] .= vcat(mpGLM[i][n].𝐮, collect(Iterators.flatten(mpGLM[i][n].𝐯)))
            end
        end
        θ₀ = θFHMDDM(Aᶻ=model.θ.Aᶻ,
                     𝛃=𝛃,
                     πᶻ=model.θ.πᶻ,
                     ψ=model.θ.ψ,
                     θₐ=θₐ₀)
    else
        θ₀ = model.θ
    end
    optionsdict = Dict(model.options)
    if haskey(optionsdict, "theta_0")
        optionsdict["theta_0"] = Dict(θ₀)
    else
        optionsdict = merge(optionsdict, Dict("theta_0" => Dict(θ₀)))
    end
    FHMDDM(data=data,
           mpGLM=mpGLM,
           options=FHMDDMoptions(optionsdict),
           θ=θ₀)
end

"""
    assess_mpGLM_iterative_optimization(model; nrepeats)

Check whether the learning of the linear filters of the mixture of Poisson generalized linear model (mpGLM) is correct, when the posteriors of the latent variable are computed from both the generative parameters and previous values of the linear filters.

ARGUMENT
-`generativemodel`: an existing model whose data provide the organizational basis of the simulated data, whose options are copied, and whose parameters are used as the generative parameters

OPTIONAL ARGUMENT

SAVED TO DISK
-the simulated data are saved in the file `dirname(model.options.datapath)*"/simulateddata.mat"`
-for the `i`-th simulation, a file `dirname(model.options.resultspath)*"/simulatedresults.mat"`

RETURN
-`nothing`
"""
function assess_mpGLM_optimization(generativemodel::FHMDDM;
                                   niterations::Integer=100,
                                   nrepeats::Integer=2,
                                   tolerance::AbstractFloat = 1e-9,
                                   generativeposterior::Bool=false,
                                   startfromgenerative::Bool=false)
    model = sample(generativemodel)
    𝐲generative= predict_expected_spiketrain(generativemodel, p𝐚, p𝐳) # the spike rate used for the simulations
    optionsdict = Dict(generativemodel.options)
    optionsdict["datapath"] = dirname(generativemodel.options.datapath)*"/simulateddata.mat"
    optionsdict = merge(optionsdict, Dict("theta_generative" => Dict(generativemodel.θ)))
    dict = Dict("data" => map(trialset->Dict(trialset), simulateddata),
                "options" => optionsdict,
                "ygenerative" => 𝐲generative)
    matwrite(optionsdict["datapath"], dict)

    for i = 1:nrepeats
        if startfromgenerative || generativeposterior
            mpGLM = MixturePoissonGLM(generativemodel.θ.𝛃, simulateddata, FHMDDMoptions(optionsdict))
        else
            mpGLM = MixturePoissonGLM(simulateddata, FHMDDMoptions(optionsdict)) # initialize with random coefficients
        end
        𝛃₀ = regression_coefficients(mpGLM)
        θ₀ = θFHMDDM(Aᶻ= copy(generativemodel.θ.Aᶻ),
                     𝛃 = 𝛃₀,
                     πᶻ= copy(generativemodel.θ.πᶻ),
                     ψ = copy(generativemodel.θ.ψ),
                     θₐ= Θₐ(pulse_input_DDM.flatten(generativemodel.θ.θₐ)...))
        if haskey(optionsdict, "theta_0")
            optionsdict["theta_0"] = Dict(θ₀)
        else
            optionsdict = merge(optionsdict, Dict("theta_0" => Dict(θ₀)))
        end
        optionsdict["resultspath"] = dirname(generativemodel.options.resultspath)*"/simulatedresults"*string(i)*".mat"
        options = FHMDDMoptions(optionsdict)

        𝒬_𝛃_previous = 0.0
        for j = 1:niterations
            quantities = EMquantities(simulateddata, options)
            @unpack γ = quantities
            println("Repeat ", i, " iteration ", j)
            println("beginning of loop")
            println("𝛃[1][1] = ", regression_coefficients(mpGLM)[1][1])
            if generativeposterior
                mpGLM_Estep = mpGLM_generative
            else
                mpGLM_Estep = mpGLM
            end
            E_step!(quantities,
                    simulateddata,
                    mpGLM_Estep,
                    options,
                    generativemodel.θ.Aᶻ,
                    generativemodel.θ.θₐ,
                    generativemodel.θ.πᶻ,
                    generativemodel.θ.ψ)
            estimatefilters!(mpGLM, γ)
            𝒬_𝛃 = expectation(mpGLM, γ)
            println("end of loop")
            println("𝒬_𝛃 = ", 𝒬_𝛃)
            println("𝛃[1][1] = ", regression_coefficients(mpGLM)[1][1])
            if j > 1 && abs(𝒬_𝛃 - 𝒬_𝛃_previous)/(1+abs(𝒬_𝛃_previous)) < tolerance
                break
            else
                𝒬_𝛃_previous = 𝒬_𝛃
            end
        end

        θ = θFHMDDM(Aᶻ= copy(generativemodel.θ.Aᶻ),
                    𝛃 = regression_coefficients(mpGLM), # only parameters that are learned
                    πᶻ= copy(generativemodel.θ.πᶻ),
                    ψ = copy(generativemodel.θ.ψ),
                    θₐ= Θₐ(pulse_input_DDM.flatten(generativemodel.θ.θₐ)...))
        simulatedmodel = FHMDDM(data=simulateddata,
                                mpGLM=mpGLM,
                                options=options,
                                θ=θ)
        save(simulatedmodel)
    end
    return nothing
end

"""
    assess_θₐ_optimization(model; nrepeats)

Check whether the learning of the drift-diffusion parameters is correct

ARGUMENT
-`generativemodel`: an existing model whose data provide the organizational basis of the simulated data, whose options are copied, and whose parameters are used as the generative parameters

OPTIONAL ARGUMENT
-`niterations`: number of iterations in the learning process. If `niterations` is less than 1, then `θₐ` are learned using posteriors computed with the generative `θₐ`
-`nrepeats`: number of repeated simulations starting from random initial values

SAVED TO DISK
-the simulated data are saved in the file `dirname(model.options.datapath)*"/simulateddata.mat"`
-for the `i`-th simulation, a file `dirname(model.options.resultspath)*"/simulatedresults.mat"`

RETURN
-`nothing`
"""
function assess_θₐ_optimization(generativemodel::FHMDDM; niterations::Integer=100, nrepeats::Integer=2, tolerance::AbstractFloat = 1e-9)
    data, p𝐚, p𝐳 = simulatedata(generativemodel)
    𝐲generative= predict_expected_spiketrain(generativemodel, p𝐚, p𝐳) # the spike rate used for the simulations
    optionsdict = Dict(generativemodel.options)
    optionsdict["datapath"] = dirname(generativemodel.options.datapath)*"/simulateddata.mat"
    optionsdict = merge(optionsdict, Dict("theta_generative" => Dict(generativemodel.θ)))
    dict = Dict("data" => map(trialset->Dict(trialset), data),
                "options" => optionsdict,
                "ygenerative" => 𝐲generative)
    matwrite(optionsdict["datapath"], dict)
    mpGLM_generative = MixturePoissonGLM(generativemodel.θ.𝛃, data, FHMDDMoptions(optionsdict))
    for i = 1:nrepeats
        θₐ = Θₐ(generativemodel.options.θ₀.θₐ,
                generativemodel.options.θₐisfit,
                generativemodel.options.θₐlowerbound,
                generativemodel.options.θₐupperbound)
        println("randomly initialized θₐ = ", θₐ)
        θ₀ = θFHMDDM(Aᶻ= generativemodel.options.θ₀.Aᶻ,
                     𝛃 = generativemodel.options.θ₀.𝛃,
                     πᶻ= generativemodel.options.θ₀.πᶻ,
                     ψ = generativemodel.options.θ₀.ψ,
                     θₐ= θₐ)
        if haskey(optionsdict, "theta_0")
            optionsdict["theta_0"] = Dict(θ₀)
        else
            optionsdict = merge(optionsdict, Dict("theta_0" => Dict(θ₀)))
        end
        optionsdict["resultspath"] = dirname(generativemodel.options.resultspath)*"/simulatedresults"*string(i)*".mat"
        options = FHMDDMoptions(optionsdict)

        quantities = EMquantities(data, options)
        @unpack γᵃ, χᵃnb = quantities
        if niterations < 1
            E_step!(quantities,
                    data,
                    mpGLM_generative,
                    options,
                    generativemodel.θ.Aᶻ,
                    generativemodel.θ.θₐ,
                    generativemodel.θ.πᶻ,
                    generativemodel.θ.ψ)
            θₐ, 𝒬 = estimateθₐ(χᵃnb, data, γᵃ, options, θₐ)
            println("θₐ = ", θₐ)
        end
        𝒬previous = 0.
        for j = 1:niterations
            E_step!(quantities,
                    data,
                    mpGLM_generative,
                    options,
                    generativemodel.θ.Aᶻ,
                    θₐ,
                    generativemodel.θ.πᶻ,
                    generativemodel.θ.ψ)
            θₐ, 𝒬 = estimateθₐ(χᵃnb, data, γᵃ, options, θₐ)
            println("Repetition ", i, " iteration ", j, ": 𝒬 = ", 𝒬)
            println("θₐ = ", θₐ)
            if j > 1
                if abs(𝒬 - 𝒬previous)/(1+abs(𝒬previous)) < tolerance
                    break
                else
                    𝒬previous = 𝒬
                end
            else
                𝒬previous = 𝒬
            end
        end
        θ = θFHMDDM(Aᶻ= generativemodel.θ.Aᶻ,
                    𝛃 = generativemodel.θ.𝛃,
                    πᶻ= generativemodel.θ.πᶻ,
                    ψ = generativemodel.θ.ψ,
                    θₐ= θₐ) # only parameters that are learned
        simulatedmodel = FHMDDM(data=data,
                                mpGLM=mpGLM_generative,
                                options=options,
                                θ=θ)
        save(simulatedmodel)
    end
    return nothing
end

"""
    loglikelihood(model)

Compute the (incomplete-data) log-likelihood log p(𝐘, d ∣ θ)

INPUT
-`model`: an instance of FHM-DDM

OPTIONAL INPUT
-`quantities`: a structure containing quantities used in computing the posterior probabilities

RETURN
-a scalar representing the log-likelihood of the model
"""
function loglikelihood(model::FHMDDM;
                       quantities=EMquantities(model.data, model.options))
    @unpack data, mpGLM, options, θ = model
    @unpack Aᶻ, θₐ, πᶻ, ψ = θ
    @unpack Aᵃ, Aᵃsilent, D, f, p𝒂, p𝐘𝑑 = quantities
    accumulatortransitions!(Aᵃ, Aᵃsilent, data, options, θₐ; minAᵃ=0.)
    emissionslikelihood!(p𝐘𝑑, data, mpGLM, ψ) # `p𝐘𝑑` is the conditional likelihood p(𝐘ₜ, d ∣ aₜ, zₜ)
    ℓ = 0.0 # log-likelihood
    for i in eachindex(data)
       for m in eachindex(data[i].trials)
           forward!(f[i][m], D[i][m], Aᵃ[i][m], Aᶻ, πᶻ, p𝒂, p𝐘𝑑[i][m])
           ∏D = 1.0
           for t in eachindex(f[i][m])
               ∏D *= D[i][m][t]
               f_unscaled = f[i][m][t]*∏D # for t < T, f_unscaled = p(𝐘ₜ, aₜ, zₜ), and for t=T, f_unscaled = p(𝐘ₜ, d, aₜ, zₜ)
               ℓ += log(sum(f_unscaled))
           end
       end
    end
    return ℓ
end

"""
"""
function testf1(X)
    pmap(x->x⋅x, X)
end

"""
"""
function testf2(X)
    pmap(t->X[t]⋅X[t], 1:length(X))
end

"""
"""
function testf3(X)
    map(x->x⋅x, X)
end
