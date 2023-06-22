"""
    EKCrossValidation(trialsets)

Return an object containing cross-validated exponential kernel models

ARGUMENT
-`trialsets`: A vector whose each element is an instance of `Trialset`

OPTIONAL ARGUMENT
-`Δt`: duration, in seconds, of each time step
-`kfold`: number of cross-validation folds
-`latency`: number of time steps the clicks are shifted
-`T`: number of time steps
"""
function EKCrossValidation(trialsets::Vector{<:Trialset}; Δt::AbstractFloat=0.01, kfold::Integer=10, T::Integer=75, latency::Integer=1)
    trials = collect((trialset.trials for trialset in trialsets)...)
    stimulus_duration_s = map(trials) do trial
        if isempty(trial.clicks.time)
            0
        else
            maximum(trial.clicks.time)
        end
    end
    min_stimulus_duration_s = T*Δt
    trials = trials[stimulus_duration_s.>=min_stimulus_duration_s]
    𝐝 = BitArray(collect(trial.choice for trial in trials))
    𝛌 = collect(FHMDDM.gamma2lambda(trial.γ) for trial in trials)
    𝐄 = map(𝛌,trials) do λ, trial
            leftpulses = collect(length(left) for left in trial.clicks.left)
            rightpulses = collect(length(right) for right in trial.clicks.right)
            𝐜 = rightpulses .- leftpulses
            𝐜[1+latency:T+latency] .- λ*Δt
        end
    testingtrials, trainingtrials = cvpartition(kfold, length(trials))
    testingmodels = map(testingtrials) do testingtrials
                        ExponentialKernelModel(𝐝 = 𝐝[testingtrials],
                                            𝐄 = 𝐄[testingtrials],
                                            Δt = Δt,
                                            𝛌=𝛌[testingtrials],
                                            T=T)
                    end
    trainingmodels = map(trainingtrials) do trainingtrials
                        ExponentialKernelModel(𝐝 = 𝐝[trainingtrials],
                                            𝐄 = 𝐄[trainingtrials],
                                            Δt = Δt,
                                            𝛌=𝛌[trainingtrials],
                                            T=T)
                    end
    EKCrossValidation(𝐝=𝐝,
                    kfold=kfold,
                    𝛌=𝛌,
                    testingmodels=testingmodels,
                    trainingmodels=trainingmodels,
                    testingtrials=testingtrials,
                    trainingtrials=trainingtrials)
end

"""
    gamma2lambda(γ)

Transform the log-odds of the right and left pulse input rate (γ's) to the difference between the input rates

OPTIONAL ARGUMENT
-`f`: total pulse rate per second
"""
function gamma2lambda(γ::Real; f::Real=40)
    f * (exp(γ)-exp(-γ)) / (2+exp(γ)+exp(-γ))
end


"""
    crossvalidate!(ekcv)

Make out-of-sample predictions
"""
function crossvalidate!(ekcv::EKCrossValidation)
    for trainingmodel in ekcv.trainingmodels
        maximizeposterior!(trainingmodel)
    end
    for k = 1:ekcv.kfold
        ekcv.testingmodels[k].𝛃 .= ekcv.trainingmodels[k].𝛃
    end
    FHMDDM.predict!(ekcv)
    return nothing
end

"""
    maximizeposterior!(ekmodel)

Find the maximum-a-posteriori parameters of an exponential kernel model.

The optimization algorithm is done using the quasi-Newton algorithm L-BFGS. The gradient of the log-likelihood is computed using automatic differentiation.
"""
function maximizeposterior!(ekmodel::ExponentialKernelModel; α::AbstractFloat=0.5, nrepeat::Integer=5, show_trace::Bool=false)
    f(𝛃) = -loglikelihood(𝛃, ekmodel) + α*dot(𝛃,𝛃)
    minloss = Inf
    𝛃ML = copy(ekmodel.𝛃)
    for i = 1:nrepeat
        𝛃₀ = rand(length(ekmodel.𝛃))
        od = OnceDifferentiable(f, 𝛃₀; autodiff = :forward);
        optimizer = LBFGS(linesearch = LineSearches.BackTracking())
        options = Optim.Options(show_trace=show_trace)
        optimizationresults = optimize(od, 𝛃₀, optimizer, options)
        show_trace && println(optimizationresults)
        thisloss = Optim.minimum(optimizationresults)
        if thisloss < minloss
            𝛃ML = Optim.minimizer(optimizationresults)
            minloss = thisloss
        end
    end
    ekmodel.𝛃 .= 𝛃ML
    return nothing
end
"""
    loglikelihood(𝛃, ekmodel)

Log-likelihood of the parameters `𝛃` given the data contained in the object `ekmodel`
"""
function loglikelihood(𝛃::Vector{<:Real}, ekmodel::ExponentialKernelModel)
    @unpack 𝐝, Δt, 𝐄, 𝛌, 𝛌abs, T, 𝛕 = ekmodel
    TΔt = T*Δt
    ℓ = 0
    for i = 1:length(𝐝)
        p = probabilityright(𝛃, 𝐄[i], TΔt, 𝛌[i], 𝛌abs[i], 𝛕)
        ℓ += 𝐝[i] ? log(p) : log(1-p)
    end
    return ℓ
end

"""
    probabilityright(𝛃, TΔt, λ, λabs, 𝛕)

Probability of a right choice

ARGUMENT
-`𝛃`: parameters
-`𝐞`: excess pulse input
-`TΔt`: stimulus duration, in seconds
-`λ`: expected input rate
-`λabs`: absolute value of the expected input rate
-`𝛕`: time, in seconds, of each tiem step from stimulus onset
"""
function probabilityright(𝛃::Vector{<:Real}, 𝐞::Vector{<:Real}, TΔt::Real, λ::Real, λabs::Real, 𝛕::Vector{<:Real})
    k = 𝛃[4]*λabs
    y = dot(exp.(k.*𝛕),𝐞)
    b⁻¹ = k/(exp(k*TΔt)-1)
    x = 𝛃[1] + 𝛃[2]TΔt*λ + 𝛃[3]*y*b⁻¹
    return logistic(x)
end

"""
    predict!(ekcv)

Make out-of-sample predictions
"""
function predict!(ekcv::EKCrossValidation)
    𝐩right = map(ekcv.testingmodels) do testingmodel
                @unpack 𝛃, 𝐝, Δt, 𝐄, 𝛌, 𝛌abs, T, 𝛕 = testingmodel
                TΔt = T*Δt
                collect(probabilityright(𝛃, 𝐞, TΔt, λ, λabs, 𝛕) for (𝐞, λ, λabs) in zip(𝐄, 𝛌,𝛌abs))
            end
    𝐩right = vcat(𝐩right...)
    sortindices = sortperm(vcat(ekcv.testingtrials...))
    ekcv.𝐩right .= 𝐩right[sortindices]
    for i = 1:length(ekcv.ℓ)
        p = ekcv.𝐩right[i]
        ekcv.ℓ[i] = ekcv.𝐝[i] ? log(p) : log(1-p)
    end
    ℓbernoulli = map(ekcv.testingmodels, ekcv.trainingmodels) do testingmodel, trainingmodel
                    p = mean(trainingmodel.𝐝)
                    collect(d ? log(p) : log(1-p) for d in testingmodel.𝐝)
                end
    ℓbernoulli = vcat(ℓbernoulli...)
    ekcv.ℓbernoulli .= ℓbernoulli[sortindices]
    for k = 1:ekcv.kfold
        ekcv.trainingmodels[k].hessian .= hessian(ekcv.trainingmodels[k])
    end
    return nothing
end

"""
    save(ekcv, filepath)

Save the cross-validation results of an exponential kernel model
"""
function save(ekcv::EKCrossValidation, filepath::String)
    dict = Dict("betas"=>collect(trainingmodel.𝛃 for trainingmodel in ekcv.trainingmodels),
                "lambda"=>ekcv.𝛌,
                "choice"=>ekcv.𝐝,
                "pright"=>ekcv.𝐩right,
                "loglikelihood"=>ekcv.ℓ,
                "loglikelihood_bernoulli"=>ekcv.ℓbernoulli,
                "times_s"=>ekcv.trainingmodels[1].𝛕,
                "hessians"=>collect(trainingmodel.hessian for trainingmodel in ekcv.trainingmodels))
    matwrite(filepath, dict)
end

"""
    hessian(ekmodel)

Hessian matrix of the exponential kernel model
"""
function hessian(ekmodel::ExponentialKernelModel)
    f(𝛃) = -loglikelihood(𝛃, ekmodel)
    ForwardDiff.hessian(f, ekmodel.𝛃)
end
