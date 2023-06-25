"""
    PKMCrossValidation(trials)

Return an object containing cross-validated models for estimating the psychophysical kernel

ARGUMENT
-`trial`: A vector whose elements are of the type `Trial`

OPTIONAL ARGUMENT
-`α`: maximum lapse rate
-`Δt`: duration, in seconds, of each time step
-`kfold`: number of cross-validation folds
-`latency`: number of time steps the clicks are shifted
-`T`: number of time steps
"""
function PKMCrossValidation(nfunctions::Integer, trials::Vector{<:Trial}; α=0.2, Δt::AbstractFloat=0.01, kfold::Integer=10, latency::Integer=2, T::Integer=75)
    stimulus_duration_s = map(trials) do trial
        if isempty(trial.clicks.time)
            0
        else
            maximum(trial.clicks.time)
        end
    end
    min_stimulus_duration_s = [T+latency]*Δt
    trials = trials[stimulus_duration_s.>=min_stimulus_duration_s]
    𝐝 = BitArray(collect(trial.choice for trial in trials))
    timesteps = latency .+ (1:T)
    𝐂 = map(trials) do trial
            leftpulses = collect(length(left) for left in trial.clicks.left)
            rightpulses = collect(length(right) for right in trial.clicks.right)
            𝐜 = rightpulses .- leftpulses
            𝐜[timesteps]
        end
    𝛌Δt = collect(gamma2lambda(trial.γ)*Δt for trial in trials)
    𝐂 = vcat(collect(𝐜' for 𝐜 in 𝐂)...)
    𝐄 = 𝐂 .- 𝛌Δt
    M = length(trials)
    𝐗 = hcat(ones(M), 𝛌Δt.*T) # I think of `𝛌Δt.*T` = ones(M,T)*(𝛌Δt.*ones(T))`, which is conceptually `Φ*𝚲Δt`
    if nfunctions > 0
        if nfunctions > 1
            begins0=false
            ends0=false
            stretch=1e-6
            scalefactor = 1.0
            Φ = temporal_basis_functions(begins0, ends0, nfunctions, T, scalefactor, stretch; orthogonal_to_ones=false)
        else nfunctions == 1
            Φ = ones(T,1)./T
        end
        𝐗 = hcat(𝐗, 𝐄*Φ)
    else
        Φ = zeros(T,0)
    end
    testingtrials, trainingtrials = cvpartition(kfold, length(trials))
    testingmodels = map(testingtrials) do testingtrials
                        PsychophysicalKernelModel(α=α,
                                            𝐝 = 𝐝[testingtrials],
                                            𝐗 = 𝐗[testingtrials,:],
                                            Δt = Δt,
                                            Φ=Φ)
                    end
    trainingmodels = map(trainingtrials) do trainingtrials
                        PsychophysicalKernelModel(α=α,
                                            𝐝 = 𝐝[trainingtrials],
                                            𝐗 = 𝐗[trainingtrials,:],
                                            Δt = Δt,
                                            Φ=Φ)
                    end
    PKMCrossValidation(𝐂=𝐂,
                    𝐝=𝐝,
                    Δt=Δt,
                    𝐄=𝐄,
                    kfold=kfold,
                    𝛌Δt=𝛌Δt,
                    T=T,
                    testingmodels=testingmodels,
                    trainingmodels=trainingmodels,
                    testingtrials=testingtrials,
                    trainingtrials=trainingtrials,
                    𝐗=𝐗)
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
    crossvalidate!(cv)

Make out-of-sample predictions

OPTIONAL ARGUMENT
-`α`: hyperparameter for L2-regularization
"""
function crossvalidate!(cv::PKMCrossValidation; α::AbstractFloat=0.0)
    for trainingmodel in cv.trainingmodels
        maximizeposterior!(trainingmodel; α=α)
    end
    for k = 1:cv.kfold
        cv.testingmodels[k].𝛃 .= cv.trainingmodels[k].𝛃
    end
    predict!(cv)
    return nothing
end

"""
    maximizeposterior!(model)

Find the maximum-a-posteriori parameters of an psychophysical kernel model.

The optimization algorithm is done using the quasi-Newton algorithm L-BFGS. The gradient of the log-likelihood is computed using automatic differentiation.

OPTIONAL ARGUMENT
-`α`: hyperparameter for L2-regularization
-`nrepeat`: number of repeats
"""
function maximizeposterior!(model::PsychophysicalKernelModel; α::AbstractFloat=0.0, nrepeat::Integer=5, show_trace::Bool=false)
    f(𝛃) = -loglikelihood(𝛃, model) + α*dot(𝛃,𝛃)
    minloss = Inf
    𝛃ML = copy(model.𝛃)
    for i = 1:nrepeat
        𝛃₀ = 1.0 .- 2.0.*rand(length(model.𝛃))
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
    model.𝛃 .= 𝛃ML
    return nothing
end

"""
    loglikelihood(𝛃, model)

Log-likelihood of the parameters of a psychophysical kernel model given the data
"""
function loglikelihood(𝛃::Vector{<:Real}, model::PsychophysicalKernelModel)
    @unpack α, 𝐝, 𝐗 = model
    𝐋 = 𝐗*𝛃[2:end]
    ℓ = 0
    ψ = α*logistic(𝛃[1])
    pright_lapse = logistic(𝛃[2])
    for i = 1:length(𝐝)
        pright = ψ*pright_lapse + (1-ψ)*logistic(𝐋[i])
        ℓ += 𝐝[i] ? log(pright) : log(1-pright)
    end
    return ℓ
end

"""
    predict!(cv)

Make out-of-sample predictions
"""
function predict!(cv::PKMCrossValidation)
    𝐩right =   map(cv.testingmodels) do testingmodel
                    @unpack α, 𝛃, 𝐝, 𝐗 = testingmodel
                    𝐋 = 𝐗*𝛃[2:end]
                    ψ = α*logistic(𝛃[1])
                    pright_lapse = logistic(𝛃[2])
                    (ψ*pright_lapse) .+ (1-ψ).*logistic.(𝐋)
                end
    𝐩right = vcat(𝐩right...)
    sortindices = sortperm(vcat(cv.testingtrials...))
    cv.𝐩right .= 𝐩right[sortindices]
    for i = 1:length(cv.ℓ)
        p = cv.𝐩right[i]
        cv.ℓ[i] = cv.𝐝[i] ? log(p) : log(1-p)
    end
    ℓbernoulli = map(cv.testingmodels, cv.trainingmodels) do testingmodel, trainingmodel
                    p = mean(trainingmodel.𝐝)
                    collect(d ? log(p) : log(1-p) for d in testingmodel.𝐝)
                end
    ℓbernoulli = vcat(ℓbernoulli...)
    cv.ℓbernoulli .= ℓbernoulli[sortindices]
    for k = 1:cv.kfold
        cv.trainingmodels[k].hessian .= hessian(cv.trainingmodels[k])
    end
    if size(cv.testingmodels[1].Φ,2) > 0
        for k = 1:cv.kfold
            @unpack Φ, 𝛃 = cv.testingmodels[k]
            cv.psychophysicalkernel[k] .= Φ*𝛃[4:end]
        end
    end
    return nothing
end

"""
    hessian(model)

Hessian matrix of the psychophysical kernel model
"""
function hessian(model::PsychophysicalKernelModel)
    f(𝛃) = -loglikelihood(𝛃, model)
    ForwardDiff.hessian(f, model.𝛃)
end

"""
    save(cv, filepath)

Save the cross-validation results of an psychophysical kernel model
"""
function save(cv::PKMCrossValidation, filepath::String)
    dict = Dict("betas"=>collect(trainingmodel.𝛃 for trainingmodel in cv.trainingmodels),
                "choice"=>cv.𝐝,
                "clickdifference_hz"=>vec(mean(cv.𝐂, dims=2))./cv.Δt,
                "pright"=>cv.𝐩right,
                "loglikelihood"=>cv.ℓ,
                "loglikelihood_bernoulli"=>cv.ℓbernoulli,
                "psychophysicalkernel"=>cv.psychophysicalkernel,
                "temporal_basis_functions"=>cv.testingmodels[1].Φ,
                "times_s"=>collect(cv.Δt.*(1:cv.T)),
                "hessians"=>collect(trainingmodel.hessian for trainingmodel in cv.trainingmodels))
    matwrite(filepath, dict)
end
